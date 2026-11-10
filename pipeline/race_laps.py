"""Seleciona as voltas de corrida, marca as voltas limpas e calcula o alvo e as variáveis.

Entrada: data/interim/laps.parquet (build_laps.py). Saída: data/processed/race_laps.parquet,
uma linha por (corrida, piloto, volta), com:

- referência do piloto no fim de semana: a melhor volta em pneu de pista seca no quali ou, se o
  quali foi molhado, nos treinos livres. É conhecida antes da largada, então pode ser usada ao vivo
  sem vazar informação do futuro;
- alvo: pace = tempo de volta / referência − 1 (ritmo relativo);
- flags de limpeza e a máscara `clean`, que define em quais voltas a perda é calculada;
- variáveis por volta: setores relativos, speed traps relativos, gaps, pneu, clima e progresso;
- dureza nominal do composto (C1–C6) e a nominação do fim de semana, de reference/compounds.csv.

Toda variável de entrada usa só informação disponível até a volta corrente (é calculável ao vivo).
As exceções são `outlier` e `clean`, que usam a mediana da corrida inteira e por isso entram apenas
na máscara da perda e nas métricas, nunca como entrada; a versão ao vivo é `clean_live`.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m pipeline.race_laps
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAPS = ROOT / "data" / "interim" / "laps.parquet"
CIRCUITS = ROOT / "data" / "reference" / "circuits.csv"
COMPOUNDS = ROOT / "data" / "reference" / "compounds.csv"
OUT = ROOT / "data" / "processed" / "race_laps.parquet"

DRY = ["SOFT", "MEDIUM", "HARD"]
WET = ["INTERMEDIATE", "WET"]
COMPOUND_REL = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
NEUTRALISED = "[4567]"           # 4 safety car, 5 bandeira vermelha, 6 VSC, 7 fim do VSC
OUTLIER_RATIO = 1.07             # volta > 107% da mediana limpa do piloto na corrida
LIVE_PACE_MAX = 0.30             # teto de ritmo de uma volta limpa ao vivo (30% acima da referência)
WET_RACE_SHARE = 0.10            # corrida "molhada" se > 10% das voltas foram com pneu de chuva
WET_QUALI_SHARE = 0.50           # quali "molhado" se > 50% das voltas foram com pneu de chuva
PACE_CLIP = (-0.05, 0.60)        # limites do ritmo usado como entrada (voltas de SC/pit são muito lentas)
KEYS = ["race_id", "driver"]


def regulation(year: int) -> str:
    return "2026" if year >= 2026 else "2022-25"


def weekend_reference(laps: pd.DataFrame) -> pd.DataFrame:
    """Volta e setores de referência do piloto no fim de semana, todos em pneu de pista seca.

    Usa a melhor volta seca do quali. Se o quali foi molhado (mais da metade das voltas com pneu de
    chuva) ou se o piloto não tem volta seca no quali, usa a melhor volta seca dos treinos livres.
    Sem volta seca nos treinos (fim de semana todo molhado, como Spa 2023), usa as voltas com pneu
    liso do quali molhado e, por fim, a mediana das referências do evento.
    O critério é da sessão, não da volta: num quali úmido, as poucas voltas com pneu liso são
    lentas (Canadá 2023: a volta "seca" do quali ficou ~9% mais lenta que a dos treinos).
    Os setores de referência vêm da mesma fonte que a volta.
    """
    keys = ["year", "event", "driver"]
    quali = laps[laps["stype"] == "Q"]
    wet_share = quali["compound"].isin(WET).groupby([quali["year"], quali["event"]]).mean()
    wet_quali = wet_share[wet_share > WET_QUALI_SHARE].index
    in_wet_quali = pd.Series(pd.MultiIndex.from_frame(laps[["year", "event"]]).isin(wet_quali), index=laps.index)
    dry = laps[laps["compound"].isin(DRY) & (laps["deleted"] != True) & laps["lap_time"].notna()]
    is_q = dry["stype"] == "Q"
    wet_q = is_q & in_wet_quali[dry.index]
    sources = [("Q", is_q & ~wet_q),           # quali seco
               ("FP", dry["stype"] == "FP"),    # treinos livres
               ("Q_wet", wet_q)]                # voltas lisas de um quali molhado (último recurso)
    parts = []
    for source, rows in sources:
        ref = dry[rows].groupby(keys).agg(
            ref_lap=("lap_time", "min"), ref_s1=("s1", "min"), ref_s2=("s2", "min"), ref_s3=("s3", "min"))
        parts.append(ref.assign(ref_source=source))
    # Prioridade na ordem acima: cada piloto usa a primeira fonte em que tem volta seca
    ref = pd.concat(parts)
    ref = ref[~ref.index.duplicated(keep="first")]
    ref["ref_from_quali"] = ref["ref_source"] == "Q"
    return ref.reset_index()


def add_reference(race: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    race = race.merge(ref, on=["year", "event", "driver"], how="left")
    # Piloto sem volta seca no fim de semana: usa a mediana das referências do evento
    race["ref_fallback"] = race["ref_lap"].isna()
    race["ref_source"] = race["ref_source"].fillna("event_median")
    race["ref_from_quali"] = race["ref_from_quali"].fillna(False).astype(bool)
    for c in ["ref_lap", "ref_s1", "ref_s2", "ref_s3"]:
        race[c] = race[c].fillna(race.groupby("race_id")[c].transform("median"))
    return race


def add_flags(race: pd.DataFrame) -> pd.DataFrame:
    status = race["track_status"].fillna("")
    race["has_time"] = race["lap_time"].notna()
    race["lap1"] = race["lap"] == 1
    race["neutralised"] = status.str.contains(NEUTRALISED, regex=True)
    race["yellow"] = status.str.contains("2", regex=False)
    race["wet_tyre"] = race["compound"].isin(WET)
    race["dry_tyre"] = race["compound"].isin(DRY)
    race["deleted"] = race["deleted"].fillna(False).astype(bool)
    race["generated"] = race["generated"].fillna(False).astype(bool)
    candidate = (race["has_time"] & ~race["lap1"] & ~race["pit_in"] & ~race["pit_out"]
                 & ~race["neutralised"] & ~race["deleted"] & ~race["generated"] & race["dry_tyre"])
    median = race["lap_time"].where(candidate).groupby([race[k] for k in KEYS]).transform("median")
    race["candidate"] = candidate
    race["outlier"] = candidate & (race["lap_time"] > OUTLIER_RATIO * median)
    race["clean"] = candidate & ~race["outlier"]
    wet_share = race.groupby("race_id")["wet_tyre"].transform("mean")
    race["race_wet"] = wet_share > WET_RACE_SHARE
    return race


def add_compounds(race: pd.DataFrame, compounds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dureza nominal do composto de cada volta e a nominação do fim de semana (reference/compounds.csv).

    `compound_c` é o número do composto da volta (C1 = 1 … C6 = 6), vazio em pneu de chuva e em GP
    sem nominação conhecida. `c_hard`, `c_medium` e `c_soft` descrevem a nominação inteira do fim de
    semana. A numeração muda de sentido entre temporadas (o C1 de 2023 é um composto novo, e o C1 de
    2022 virou C0), por isso a dureza só se compara dentro de um ano; ver reference/tyre_specs.csv.
    """
    if compounds is None:
        compounds = pd.read_csv(COMPOUNDS)
    nominated = compounds[["year", "circuit"]].copy()
    for k in ("hard", "medium", "soft"):
        nominated[f"c_{k}"] = pd.to_numeric(compounds[k].str[1:], errors="coerce").astype("Int8")
    race = race.drop(columns=[c for c in ("c_hard", "c_medium", "c_soft", "compound_c") if c in race])
    race = race.merge(nominated, on=["year", "circuit"], how="left")
    race["compound_c"] = pd.Series(pd.NA, index=race.index, dtype="Int8")
    for compound, col in (("HARD", "c_hard"), ("MEDIUM", "c_medium"), ("SOFT", "c_soft")):
        rows = race["compound"] == compound
        race.loc[rows, "compound_c"] = race.loc[rows, col]
    return race


def add_gaps(race: pd.DataFrame) -> pd.DataFrame:
    """Gap ao carro da frente e ao líder na linha de chegada de cada volta.

    Usa o tempo de sessão ao fim da volta (session_time) dos carros na mesma volta, ordenados pela
    posição. Equivale ao IntervalToPositionAhead / GapToLeader do feed ao vivo.
    """
    r = race[["race_id", "lap", "position", "session_time"]].dropna(subset=["position"])
    r = r.sort_values(["race_id", "lap", "position"])
    g = r.groupby(["race_id", "lap"])["session_time"]
    race["gap_ahead"] = (r["session_time"] - g.shift(1)).reindex(race.index)
    race["gap_leader"] = (r["session_time"] - g.transform("first")).reindex(race.index)
    return race


def past_median(s: pd.Series, by: list[pd.Series], valid: pd.Series) -> pd.Series:
    """Mediana das voltas válidas anteriores do mesmo piloto (só usa o passado)."""
    return s.where(valid).groupby(by).transform(lambda x: x.expanding().median().shift(1))


def causal_relative(s: pd.Series, by: list[pd.Series], valid: pd.Series) -> pd.Series:
    """s / (mediana das voltas válidas anteriores do mesmo piloto) − 1."""
    return s / past_median(s, by, valid) - 1


def add_features(race: pd.DataFrame) -> pd.DataFrame:
    race = race.sort_values(KEYS + ["lap"]).reset_index(drop=True)
    by = [race[k] for k in KEYS]
    # Volta limpa calculável ao vivo: o corte de 107% usa a mediana das voltas anteriores, não a da
    # corrida inteira. `clean` (com a mediana da corrida) fica só para a máscara da perda e as métricas;
    # as entradas da rede e os baselines usam `clean_live`.
    # O teto absoluto de ritmo cobre o começo da corrida, quando ainda não há voltas anteriores para
    # comparar (voltas de relargada depois de bandeira vermelha chegam a 2× o tempo normal).
    race["pace"] = race["lap_time"] / race["ref_lap"] - 1
    past = past_median(race["lap_time"], by, race["candidate"])
    race["clean_live"] = (race["candidate"] & (race["pace"] <= LIVE_PACE_MAX)
                          & ~(race["lap_time"] > OUTLIER_RATIO * past))
    race["pace_in"] = race["pace"].clip(*PACE_CLIP)
    for k in (1, 2, 3):
        race[f"s{k}_pace"] = (race[f"s{k}"] / race[f"ref_s{k}"] - 1).clip(*PACE_CLIP)
    for c in ["speed_i1", "speed_i2", "speed_fl", "speed_st"]:
        race[f"{c}_rel"] = causal_relative(race[c], by, race["clean_live"]).clip(-0.2, 0.2)

    race["laps_total"] = race.groupby("race_id")["lap"].transform("max")
    race["laps_remaining"] = race["laps_total"] - race["lap"]
    race["race_progress"] = race["lap"] / race["laps_total"]
    race["stint_lap"] = race.groupby(KEYS + ["stint"]).cumcount() + 1
    race["compound_rel"] = race["compound"].map(COMPOUND_REL).astype("Int8")
    race["fresh_tyre"] = race["fresh_tyre"].fillna(False).astype(bool)
    race["traffic"] = race["gap_ahead"] < 1.5
    race["regulation"] = race["year"].map(regulation)
    return race


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    laps = pd.read_parquet(LAPS)
    circuits = pd.read_csv(CIRCUITS)[["year", "event", "circuit"]]
    race = laps[laps["stype"] == "R"].copy()
    # As corridas são identificadas pelo circuito, não pelo nome do GP (que muda entre anos)
    race = race.merge(circuits, on=["year", "event"], how="left")
    if race["circuit"].isna().any():
        raise SystemExit(f"GPs sem circuito em {CIRCUITS.name}: "
                         f"{race.loc[race['circuit'].isna(), ['year', 'event']].drop_duplicates().values.tolist()}")
    race["race_label"] = race["year"].astype(str) + " " + race["circuit"]
    race = add_reference(race, weekend_reference(laps))
    race = add_flags(race)
    race = add_gaps(race)
    race = add_features(race)
    race = add_compounds(race)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    race.to_parquet(OUT, index=False)
    n_races = race["race_id"].nunique()
    dry = race[~race["race_wet"]]
    missing = race.loc[race["dry_tyre"] & race["compound_c"].isna(), "race_label"].unique()
    print(f"voltas em pneu liso com dureza C conhecida: {race.loc[race['dry_tyre'], 'compound_c'].notna().mean():.1%}"
          + (f" (sem nominação: {', '.join(missing)})" if len(missing) else ""))
    print(f"{len(race):,} voltas de corrida, {n_races} corridas ({race.loc[race['race_wet'], 'race_id'].nunique()} molhadas)"
          f" -> {OUT.relative_to(ROOT)}")
    print(f"voltas limpas em corridas secas: {dry['clean'].sum():,} de {len(dry):,} ({dry['clean'].mean():.1%})")
    print(f"pilotos-corrida sem referência própria: {race.loc[race['ref_fallback'], KEYS].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
