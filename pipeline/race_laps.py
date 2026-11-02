"""Seleciona as voltas de corrida, marca as voltas limpas e calcula o alvo e as variáveis.

Entrada: data/interim/laps.parquet (build_laps.py). Saída: data/processed/race_laps.parquet,
uma linha por (corrida, piloto, volta), com:

- referência do piloto no fim de semana: a melhor volta em pneu de pista seca nas sessões
  anteriores à corrida (treinos, quali, quali da sprint e sprint). É conhecida antes da largada,
  então pode ser usada ao vivo sem vazar informação do futuro;
- alvo: pace = tempo de volta / referência − 1 (ritmo relativo);
- flags de limpeza e a máscara `clean`, que define em quais voltas a perda é calculada;
- variáveis por volta: setores relativos, speed traps relativos, gaps, pneu, clima e progresso.

Toda variável de entrada usa só informação disponível até a volta corrente (é calculável ao vivo).
A única exceção é `outlier`, que entra apenas na máscara da perda, nunca como entrada.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m pipeline.race_laps
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAPS = ROOT / "data" / "interim" / "laps.parquet"
CIRCUITS = ROOT / "data" / "reference" / "circuits.csv"
OUT = ROOT / "data" / "processed" / "race_laps.parquet"

DRY = ["SOFT", "MEDIUM", "HARD"]
WET = ["INTERMEDIATE", "WET"]
COMPOUND_REL = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
NEUTRALISED = "[4567]"           # 4 safety car, 5 bandeira vermelha, 6 VSC, 7 fim do VSC
OUTLIER_RATIO = 1.07             # volta > 107% da mediana limpa do piloto na corrida
WET_RACE_SHARE = 0.10            # corrida "molhada" se > 10% das voltas foram com pneu de chuva
PACE_CLIP = (-0.05, 0.60)        # limites do ritmo usado como entrada (voltas de SC/pit são muito lentas)
KEYS = ["race_id", "driver"]


def regulation(year: int) -> str:
    return "2026" if year >= 2026 else "2022-25"


def weekend_reference(laps: pd.DataFrame) -> pd.DataFrame:
    """Melhor volta e melhores setores do piloto antes da corrida, em pneu de pista seca."""
    pre = laps[(laps["stype"] != "R") & laps["compound"].isin(DRY) & (laps["deleted"] != True)]
    keys = ["year", "event", "driver"]
    ref = pre.groupby(keys).agg(
        ref_lap=("lap_time", "min"), ref_s1=("s1", "min"), ref_s2=("s2", "min"), ref_s3=("s3", "min"))
    # De qual tipo de sessão veio a melhor volta. Se o quali foi molhado, ela vem dos treinos e
    # é mais lenta que uma volta de quali seca, o que desloca o ritmo relativo da corrida para baixo.
    best = pre.dropna(subset=["lap_time"]).sort_values("lap_time").drop_duplicates(keys)
    ref = ref.join(best.set_index(keys)["stype"].rename("ref_stype"))
    ref["ref_from_quali"] = ref["ref_stype"].isin(["Q", "SQ"])
    return ref.reset_index()


def add_reference(race: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    race = race.merge(ref, on=["year", "event", "driver"], how="left")
    # Piloto sem volta seca no fim de semana: usa a mediana das referências do evento
    race["ref_fallback"] = race["ref_lap"].isna()
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
    race["outlier"] = candidate & (race["lap_time"] > OUTLIER_RATIO * median)
    race["clean"] = candidate & ~race["outlier"]
    wet_share = race.groupby("race_id")["wet_tyre"].transform("mean")
    race["race_wet"] = wet_share > WET_RACE_SHARE
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


def causal_relative(s: pd.Series, by: list[pd.Series], valid: pd.Series) -> pd.Series:
    """s / (mediana das voltas válidas anteriores do mesmo piloto) − 1; só usa o passado."""
    past = s.where(valid).groupby(by).transform(lambda x: x.expanding().median().shift(1))
    return s / past - 1


def add_features(race: pd.DataFrame) -> pd.DataFrame:
    race = race.sort_values(KEYS + ["lap"]).reset_index(drop=True)
    by = [race[k] for k in KEYS]
    race["pace"] = race["lap_time"] / race["ref_lap"] - 1
    race["pace_in"] = race["pace"].clip(*PACE_CLIP)
    for k in (1, 2, 3):
        race[f"s{k}_pace"] = (race[f"s{k}"] / race[f"ref_s{k}"] - 1).clip(*PACE_CLIP)
    for c in ["speed_i1", "speed_i2", "speed_fl", "speed_st"]:
        race[f"{c}_rel"] = causal_relative(race[c], by, race["clean"]).clip(-0.2, 0.2)

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
    race = race.merge(circuits, on=["year", "event"], how="left")
    if race["circuit"].isna().any():
        raise SystemExit(f"GPs sem circuito em {CIRCUITS.name}: "
                         f"{race.loc[race['circuit'].isna(), ['year', 'event']].drop_duplicates().values.tolist()}")
    race = add_reference(race, weekend_reference(laps))
    race = add_flags(race)
    race = add_gaps(race)
    race = add_features(race)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    race.to_parquet(OUT, index=False)
    n_races = race["race_id"].nunique()
    dry = race[~race["race_wet"]]
    print(f"{len(race):,} voltas de corrida, {n_races} corridas ({race.loc[race['race_wet'], 'race_id'].nunique()} molhadas)"
          f" -> {OUT.relative_to(ROOT)}")
    print(f"voltas limpas em corridas secas: {dry['clean'].sum():,} de {len(dry):,} ({dry['clean'].mean():.1%})")
    print(f"pilotos-corrida sem referência própria: {race.loc[race['ref_fallback'], KEYS].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
