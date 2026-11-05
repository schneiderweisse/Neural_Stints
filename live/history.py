"""Histórico de voltas ao vivo -> as mesmas variáveis do treino.

O live.reader grava as voltas como chegam do feed. Aqui elas viram a tabela que a rede consome,
usando as próprias funções do pipeline (pipeline/race_laps.py), para que a definição de cada
variável seja idêntica no treino e ao vivo:

- referência do piloto: melhor volta seca no quali; treinos livres se o quali foi molhado
  (`reference`, a mesma regra de race_laps.weekend_reference);
- flags de limpeza (`clean_live` como entrada), ritmo relativo, setores e speed traps relativos;
- progresso da corrida pelo total de voltas anunciado no feed (LapCount), não pela maior volta vista;
- voltas que faltam no histórico (o leitor estava parado) entram como voltas sem tempo, para a
  sequência de cada piloto continuar contínua, como no treino.

Os gaps vêm direto do feed (IntervalToPositionAhead e GapToLeader), que é o que o treino calcula
pelos tempos acumulados na linha de chegada.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import race_laps as rl
from pipeline.build_laps import read_driver_laps
from pipeline.sessions import SESSION_TYPES, iter_sessions

LIVE = Path(__file__).resolve().parents[1] / "data" / "live"
# sufixo da pasta ao vivo (live.laps.session_key) -> tipo de sessão
LIVE_STYPE = {"practice_1": "FP", "practice_2": "FP", "practice_3": "FP", "qualifying": "Q",
              "sprint_qualifying": "SQ", "sprint_shootout": "SQ", "sprint": "S", "race": "R"}

RAW_COLUMNS = ["lap", "lap_time", "s1", "s2", "s3", "speed_i1", "speed_i2", "speed_fl", "speed_st", "compound",
               "stint", "tyre_life", "fresh_tyre", "position", "gap_ahead", "gap_leader", "track_status",
               "pit_in", "pit_out", "air_temp", "track_temp", "humidity", "pressure", "rain"]


def session_laps_frame(laps: pd.DataFrame, stype: str, year: int, event: str) -> pd.DataFrame:
    """Voltas de uma sessão (ao vivo ou do TracingInsights) no formato que weekend_reference espera."""
    df = laps.copy()
    df["stype"], df["year"], df["event"] = stype, year, event
    if "deleted" not in df:
        df["deleted"] = False
    return df


def reference(weekend: pd.DataFrame) -> pd.DataFrame:
    """Volta e setores de referência por piloto (mesma regra do treino)."""
    return rl.weekend_reference(weekend)


def weekend_from_tracinginsights(year: int, event: str) -> pd.DataFrame:
    """Voltas de treinos e quali de um GP no TracingInsights (rode antes o fetch_tracinginsights)."""
    frames = []
    for y, ev, session, folder in iter_sessions(years=[year]):
        if ev != event or SESSION_TYPES[session] == "R":
            continue
        for f in folder.glob("*/laptimes.json"):
            df = read_driver_laps(f)
            if not df.empty:
                frames.append(session_laps_frame(df, SESSION_TYPES[session], year, event))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ["lap_time", "s1", "s2", "s3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["compound"] = df["compound"].astype("string").str.upper()
    df["deleted"] = df["deleted"].fillna(False).astype(bool)
    return df


def weekend_from_live(year: int, circuit_slug: str, event: str, root: Path = LIVE) -> pd.DataFrame:
    """Voltas de treinos e quali gravadas ao vivo pelo live.reader (deletadas vêm do race control)."""
    frames = []
    for folder in sorted(root.glob(f"{year}_{circuit_slug}_*")):
        stype = LIVE_STYPE.get(folder.name.removeprefix(f"{year}_{circuit_slug}_"))
        if stype in (None, "R") or not (folder / "laps.parquet").exists():
            continue
        frames.append(session_laps_frame(pd.read_parquet(folder / "laps.parquet"), stype, year, event))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def event_reference(year: int, event: str, circuit_slug: str) -> tuple[pd.DataFrame, str]:
    """Referência do fim de semana: TracingInsights se já publicou o quali, senão as gravações ao vivo."""
    ti = weekend_from_tracinginsights(year, event)
    if not ti.empty and (ti["stype"] == "Q").any():
        return reference(ti), "tracinginsights"
    live = weekend_from_live(year, circuit_slug, event)
    if not live.empty:
        live["deleted"] = live["deleted"].fillna(False).astype(bool) if "deleted" in live else False
        return reference(live), "live"
    return pd.DataFrame(columns=["year", "event", "driver", "ref_lap", "ref_s1", "ref_s2", "ref_s3",
                                 "ref_source", "ref_from_quali"]), "none"


def fill_tyre(full: pd.DataFrame, missing: pd.Series) -> None:
    """Pneu das voltas ausentes, reconstruído pela volta conhecida mais próxima.

    Pela próxima volta conhecida (idade L na volta k, jogo montado com S voltas de uso), as voltas
    j com L − (k − j) > S ainda são daquele jogo, com idade L − (k − j). As anteriores continuam o jogo
    da volta conhecida anterior, contando uma volta a mais por volta. Assim uma parada no meio do
    buraco cai no lugar certo.
    """
    tyre = ["compound", "stint", "fresh_tyre"]
    if "tyre_start_laps" not in full:
        full["tyre_start_laps"] = 0.0
    known = full.loc[~missing, tyre + ["tyre_life", "tyre_start_laps"]]
    for lap in full.index[missing]:
        nxt, prv = known[known.index > lap], known[known.index < lap]
        start = (nxt["tyre_start_laps"].fillna(0).iloc[0] if len(nxt) else 0)
        if len(nxt) and nxt["tyre_life"].iloc[0] - (nxt.index[0] - lap) > start:
            src, life = nxt.iloc[0], nxt["tyre_life"].iloc[0] - (nxt.index[0] - lap)
        elif len(prv):
            src, life = prv.iloc[-1], prv["tyre_life"].iloc[-1] + (lap - prv.index[-1])
        else:
            continue
        full.loc[lap, tyre] = src[tyre].to_numpy()
        full.loc[lap, "tyre_life"] = life


def fill_missing_laps(race: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por volta de 1 até a última de cada piloto; voltas ausentes ficam sem tempo."""
    out = []
    for drv, g in race.groupby("driver", sort=False):
        g = g.drop_duplicates("lap", keep="last").set_index("lap").sort_index()
        full = g.reindex(range(1, int(g.index.max()) + 1))
        missing = pd.Series(~full.index.isin(g.index), index=full.index)
        full = full.astype({c: object for c in ["compound", "fresh_tyre"] if c in full})
        fill_tyre(full, missing)
        for c in ["driver", "driver_number", "team", "laps_total"]:
            if c in full:
                full[c] = full[c].ffill().bfill()
        for c in ["pit_in", "pit_out", "deleted"]:
            if c in full:
                full[c] = full[c].fillna(False).astype(bool)
        full["track_status"] = full["track_status"].fillna("")
        full["missing"] = missing
        out.append(full.reset_index().rename(columns={"index": "lap"}))
    return pd.concat(out, ignore_index=True)


def race_features(live: pd.DataFrame, ref: pd.DataFrame, year: int, event: str, circuit: str,
                  race_id: str = "live") -> pd.DataFrame:
    """Voltas ao vivo de uma corrida com as colunas de data/processed/race_laps.parquet."""
    race = fill_missing_laps(live)
    race["year"], race["event"], race["circuit"], race["race_id"] = year, event, circuit, race_id
    race["race_label"] = f"{year} {circuit}"
    race["deleted"] = race["deleted"].fillna(False).astype(bool) if "deleted" in race else False
    race["generated"] = False
    race = rl.add_reference(race, ref)
    race = rl.add_flags(race)
    race = rl.add_features(race)
    # progresso pela distância anunciada no feed (a corrida ainda não acabou)
    total = race["laps_total"].astype(float)
    race["laps_total"] = total
    race["laps_remaining"] = total - race["lap"]
    race["race_progress"] = race["lap"] / total
    race["stint_lap"] = race.groupby(rl.KEYS + ["stint"]).cumcount() + 1
    return race


def parity_check(race_id: str = "2026_19") -> pd.DataFrame:
    """Refaz uma corrida do treino pelo caminho ao vivo e compara as variáveis com o pipeline.

    Devolve, para cada variável, a fração de voltas em que os dois caminhos concordam.
    """
    processed = pd.read_parquet(rl.OUT)
    r = processed[processed["race_id"] == race_id].copy()
    # `deleted` ao vivo vem das mensagens do race control (live/laps.py), as mesmas do TracingInsights
    live = r[RAW_COLUMNS + ["driver", "driver_number", "team", "laps_total", "deleted"]].copy()
    ref = r.groupby(["year", "event", "driver"])[["ref_lap", "ref_s1", "ref_s2", "ref_s3", "ref_source",
                                                   "ref_from_quali"]].first().reset_index()
    year, event, circuit = int(r["year"].iloc[0]), r["event"].iloc[0], r["circuit"].iloc[0]
    feats = race_features(live, ref, year, event, circuit, race_id)
    cols = ["pace", "pace_in", "s1_pace", "s2_pace", "s3_pace", "speed_i1_rel", "speed_st_rel", "candidate",
            "clean", "clean_live", "neutralised", "yellow", "traffic", "race_progress", "laps_remaining",
            "stint_lap", "compound_rel"]
    m = feats.merge(r[["driver", "lap"] + cols], on=["driver", "lap"], suffixes=("_live", "_pipe"))
    agree = {}
    for c in cols:
        a, b = m[f"{c}_live"], m[f"{c}_pipe"]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            agree[c] = float(np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-9).mean())
        else:
            agree[c] = float((a.astype(str) == b.astype(str)).mean())
    return pd.Series(agree, name=f"concordância {race_id}").to_frame()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(parity_check(sys.argv[1] if len(sys.argv) > 1 else "2026_19").to_string())
