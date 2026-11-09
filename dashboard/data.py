"""Carregamento de dados do painel, com cache do Streamlit."""
import gzip
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from live.feed import HttpSource, ReplaySource

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LIVE = ROOT / "data" / "live"
RACE_LAPS = ROOT / "data" / "processed" / "race_laps.parquet"


# ---------------------------------------------------------------- ao vivo

def fetch_live_state(url: str) -> dict | None:
    """Estado atual do f1-dash (sem cache: muda a cada leitura)."""
    return HttpSource(url).fetch()


REPLAY_FILE = "replay.json.gz"   # estados a cada 30 s de uma gravação, feito por demo/build.py


def recordings() -> list[Path]:
    """Gravações em data/live, da mais recente para a mais antiga; na demo, que não leva o fluxo bruto, as
    reproduções já calculadas (replay.json.gz)."""
    return sorted(LIVE.glob("*/recording*.jsonl"), reverse=True) or sorted(LIVE.glob(f"*/{REPLAY_FILE}"), reverse=True)


@st.cache_resource(show_spinner="Reproduzindo a gravação…")
def recording_snapshots(path: str, every: float = 30.0) -> list[tuple[float, dict]]:
    """Estados da sessão gravada a cada `every` segundos do relógio da gravação."""
    if path.endswith(REPLAY_FILE):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [(clock, state) for clock, state in json.load(f)]
    return [(clock, json.loads(json.dumps(state))) for clock, state in ReplaySource(path, interval=every)]


def read_parquet_fresh(path: Path) -> pd.DataFrame | None:
    """Lê um parquet que outro processo regrava (sem cache, para ver a versão mais nova)."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:                      # arquivo sendo trocado neste instante: tenta na próxima atualização
        return None


# ---------------------------------------------------------------- modelos

def runs(prefix: str) -> list[str]:
    found = [p for p in RUNS.glob(f"{prefix}_*") if (p / "metrics.json").exists()]
    return [p.name for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]


@st.cache_data
def run_files(run_id: str) -> dict:
    d = RUNS / run_id
    out = {"metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
           "config": json.loads((d / "config.json").read_text(encoding="utf-8"))}
    if (d / "history.csv").exists():
        out["history"] = pd.read_csv(d / "history.csv")
    return out


@st.cache_data
def run_predictions(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS / run_id / "predictions.parquet")


@st.cache_data
def race_laps(race_label: str) -> pd.DataFrame:
    cols = ["race_label", "driver", "team", "lap", "lap_time", "clean", "pit_in", "pit_out", "neutralised", "position",
            "compound", "stint", "tyre_life", "fresh_tyre"]
    df = pd.read_parquet(RACE_LAPS, columns=cols)
    return df[df["race_label"] == race_label]
