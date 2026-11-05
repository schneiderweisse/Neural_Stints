"""Lê o live timing (f1-dash ou gravação) e mantém o histórico de voltas da sessão em disco.

A cada volta completada, grava data/live/<ano>_<circuito>_<sessão>/laps.parquet (escrita atômica:
o painel nunca lê o arquivo pela metade). Uma sessão nova (outro SessionInfo) abre outra pasta. Se o
arquivo da sessão já existe (o leitor foi reiniciado), o histórico continua a partir dele.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m live.reader                              # f1-dash em localhost:4000
    .venv/Scripts/python -m live.reader --replay <gravação> [--speed 30]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from live.feed import DEFAULT_URL, HttpSource, ReplaySource
from live.laps import LapTracker
from modeling.results import write_atomic

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"


def save(tracker: LapTracker, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "laps.parquet"
    df = pd.DataFrame(tracker.rows)
    df = df.sort_values(["driver_number", "lap"], key=lambda s: pd.to_numeric(s) if s.name == "driver_number" else s)
    write_atomic(path, lambda p: df.to_parquet(p, index=False))
    return path


def save_session_info(state: dict, out_dir: Path) -> None:
    """session.json: o SessionInfo do feed (GP, circuito, tipo e data), usado pelo live.predict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = dict(state.get("SessionInfo") or {})
    info["TotalLaps"] = (state.get("LapCount") or {}).get("TotalLaps")
    write_atomic(out_dir / "session.json",
                 lambda p: p.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"))


def load_laps(out_root: Path, session: str) -> list[dict]:
    """Voltas já gravadas da sessão (para retomar depois de um reinício do leitor)."""
    path = out_root / session / "laps.parquet"
    if not path.exists():
        return []
    rows = pd.read_parquet(path).to_dict("records")
    print(f"retomando {session}: {len(rows)} voltas já gravadas")
    return rows


def run(source, out_root: Path = LIVE, quiet: bool = False) -> LapTracker:
    tracker = LapTracker(load=lambda session: load_laps(out_root, session))
    waiting = False
    for clock, state in source:
        if state is None:
            if not waiting and not quiet:
                print(f"{datetime.now():%H:%M:%S} sem sessão no f1-dash (estado vazio); aguardando…")
            waiting = True
            continue
        waiting = False
        before = tracker.session
        new = tracker.update(state, clock)
        if tracker.session != before and tracker.session:
            save_session_info(state, out_root / tracker.session)
            if not quiet:
                print(f"sessão: {tracker.session}")
        if new and tracker.session:
            path = save(tracker, out_root / tracker.session)
            if not quiet:
                lap = max(r["lap"] for r in new)
                times = " ".join(f"{r['driver']}:{r['lap_time']:.3f}" for r in new[:6] if r["lap_time"])
                print(f"  volta {lap:>2}: {len(new)} carros · {times}{' …' if len(new) > 6 else ''}")
    if tracker.rows and tracker.session:
        save(tracker, out_root / tracker.session)
    return tracker


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--replay", help="gravação no formato do simulador do f1-dash")
    ap.add_argument("--speed", type=float, default=0.0, help="replay: 0 = o mais rápido possível, 1 = tempo real")
    ap.add_argument("--interval", type=float, default=3.0, help="segundos entre leituras do estado")
    ap.add_argument("--out", type=Path, default=LIVE)
    args = ap.parse_args()
    source = ReplaySource(args.replay, args.interval, args.speed) if args.replay else HttpSource(args.url, args.interval)
    try:
        tracker = run(source, args.out)
    except KeyboardInterrupt:
        print("interrompido")
        return
    print(f"{len(tracker.rows)} voltas registradas em {tracker.session}")


if __name__ == "__main__":
    main()
