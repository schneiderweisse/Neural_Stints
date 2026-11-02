"""Monta a tabela de voltas de todas as sessões baixadas do TracingInsights.

Lê <ano>/<GP>/<sessão>/<piloto>/laptimes.json (via sessions.iter_sessions) e grava uma linha por
volta em data/interim/laps.parquet, com nomes de coluna legíveis e tipos corretos. Inclui treinos,
quali, sprint e corrida; a seleção das voltas de corrida e a limpeza ficam em race_laps.py.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m pipeline.build_laps
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.sessions import SESSION_TYPES, iter_sessions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "laps.parquet"

# Campo do laptimes.json -> coluna da tabela
RENAME = {
    "lap": "lap", "time": "lap_time", "s1": "s1", "s2": "s2", "s3": "s3",
    "compound": "compound", "stint": "stint", "life": "tyre_life", "fresh": "fresh_tyre",
    "pos": "position", "status": "track_status", "sesT": "session_time", "lST": "lap_start_time",
    "lSD": "lap_start_date", "pin": "pit_in_time", "pout": "pit_out_time",
    "vi1": "speed_i1", "vi2": "speed_i2", "vfl": "speed_fl", "vst": "speed_st",
    "del": "deleted", "ff1G": "generated", "iacc": "accurate", "pb": "personal_best",
    "drv": "driver", "dNum": "driver_number", "team": "team",
    "wAT": "air_temp", "wTT": "track_temp", "wH": "humidity", "wP": "pressure", "wR": "rain",
    "wWD": "wind_direction", "wWS": "wind_speed",
}
NUMERIC = ["lap", "lap_time", "s1", "s2", "s3", "stint", "tyre_life", "position", "session_time",
           "lap_start_time", "pit_in_time", "pit_out_time", "speed_i1", "speed_i2", "speed_fl",
           "speed_st", "air_temp", "track_temp", "humidity", "pressure", "wind_direction", "wind_speed"]
BOOLEAN = ["fresh_tyre", "deleted", "generated", "accurate", "personal_best", "rain"]
TEXT = ["compound", "track_status", "driver", "driver_number", "team", "lap_start_date"]


def clean_value(v):
    """O TracingInsights grava valores ausentes como None ou como o texto 'None'/'nan'."""
    return None if v is None or (isinstance(v, str) and v in ("None", "nan", "NaT", "")) else v


def read_driver_laps(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    n = len(raw["lap"])
    cols = {}
    for src, dst in RENAME.items():
        values = raw.get(src)
        cols[dst] = [clean_value(v) for v in values] if values is not None and len(values) == n else [None] * n
    return pd.DataFrame(cols)


def build() -> pd.DataFrame:
    frames = []
    for year, event, session, folder in iter_sessions():
        for f in sorted(folder.glob("*/laptimes.json")):
            df = read_driver_laps(f)
            if df.empty:
                continue
            df.insert(0, "year", year)
            df.insert(1, "event", event)
            df.insert(2, "session", session)
            df.insert(3, "stype", SESSION_TYPES[session])
            frames.append(df)
    laps = pd.concat(frames, ignore_index=True)

    for c in NUMERIC:
        laps[c] = pd.to_numeric(laps[c], errors="coerce")
    for c in BOOLEAN:
        laps[c] = laps[c].map(lambda v: None if v is None else bool(v)).astype("boolean")
    for c in TEXT:
        laps[c] = laps[c].astype("string")
    laps["lap"] = laps["lap"].astype("Int16")
    laps["lap_start_date"] = pd.to_datetime(laps["lap_start_date"], errors="coerce")
    laps["compound"] = laps["compound"].str.upper()
    laps["pit_in"] = laps["pit_in_time"].notna()
    laps["pit_out"] = laps["pit_out_time"].notna()

    # Data da sessão (primeira volta) e ordem dos GPs dentro da temporada
    start = laps.groupby(["year", "event", "session"])["lap_start_date"].transform("min")
    laps["session_date"] = start.dt.normalize()
    race_dates = (laps[laps["stype"] == "R"].groupby(["year", "event"])["lap_start_date"].min()
                  .rename("race_date").reset_index())
    race_dates["round"] = race_dates.groupby("year")["race_date"].rank(method="first").astype("Int16")
    laps = laps.merge(race_dates[["year", "event", "round"]], on=["year", "event"], how="left")
    laps["race_id"] = (laps["year"].astype(str) + "_" + laps["round"].astype(str).str.zfill(2))

    order = ["year", "round", "race_id", "event", "session", "stype", "driver", "driver_number", "team"]
    laps = laps[order + [c for c in laps.columns if c not in order]]
    return laps.sort_values(["year", "round", "session", "driver", "lap"], ignore_index=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    laps = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    laps.to_parquet(OUT, index=False)
    print(f"{len(laps):,} voltas -> {OUT.relative_to(ROOT)}")
    print(laps.groupby(["year", "stype"]).size().unstack(fill_value=0).to_string())
    missing = laps["round"].isna().sum()
    if missing:
        print(f"atenção: {missing} voltas de GPs sem corrida (round ausente)")


if __name__ == "__main__":
    main()
