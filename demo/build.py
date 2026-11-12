"""Monta a demo do painel para o GitHub Pages (roda no navegador com stlite, sem servidor).

Dois passos:

    .venv/Scripts/python demo/build.py export      # dados enxutos -> demo/files/ (commitado)
    python demo/build.py site demo/_site           # site estático -> demo/_site/ (o GitHub Actions faz isto)

`export` precisa dos dados do projeto, que ficam fora do git (runs/, data/processed, data/live), e por
isso roda na máquina de quem treinou. Ele copia o run da GRU e o do ingênuo mostrados na demo, as
voltas das corridas de validação e teste e calcula a reprodução das corridas gravadas a cada 30 s, só
com o que o painel usa (`replay.json.gz`, no lugar do fluxo bruto). Quando a gravação tem previsões
feitas ao vivo, elas vão junto.

`site` só usa a biblioteca padrão: junta o index.html e um app.zip com o código das abas
(dashboard/, live/feed.py, live/laps.py, modeling/figures.py), a entrada da demo (demo/app.py) e
demo/files/. O stlite descompacta o zip no sistema de arquivos do Pyodide e roda demo_app.py.
"""
import argparse
import gzip
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
FILES = DEMO / "files"
CODE = ["dashboard/__init__.py", "dashboard/data.py", "dashboard/layout.py", "dashboard/live_tab.py",
        "dashboard/model_tab.py", "dashboard/tasks_tab.py", "dashboard/duels_view.py", "live/__init__.py", "live/feed.py",
        "live/laps.py", "modeling/__init__.py", "modeling/duels.py", "modeling/figures.py"]
RUN_FILES = ["config.json", "metrics.json", "history.csv", "predictions.parquet"]
REPLAY_SESSIONS = ["2026_interlagos_race", "2019_hockenheim_race"]
PREDICTION_COLUMNS = ["driver", "t", "h", "lap", "time_p10", "time_p50", "time_p90", "time_real", "clean", "gru_run"]
REPLAY_EVERY = 30.0
RACE_LAPS_COLUMNS = ["race_label", "driver", "team", "lap", "lap_time", "clean", "pit_in", "pit_out", "neutralised",
                     "position", "compound", "stint", "tyre_life", "fresh_tyre"]

# o que a aba Ao Vivo lê de cada estado (dashboard/live_tab.py); o resto do feed fica de fora
STATE_KEYS = ["SessionInfo", "SessionStatus", "TrackStatus", "LapCount", "WeatherData", "ExtrapolatedClock", "Heartbeat",
              "DriverList", "TimingData", "TimingAppData"]
LINE_KEYS = ["Line", "Position", "BestLapTime", "LastLapTime", "GapToLeader", "IntervalToPositionAhead",
             "TimeDiffToFastest", "TimeDiffToPositionAhead", "Sectors", "NumberOfLaps", "NumberOfPitStops", "Retired",
             "KnockedOut", "Stopped", "InPit", "PitOut"]
SECTOR_KEYS = ["Value", "PreviousValue", "OverallFastest", "PersonalFastest", "Segments"]
DRIVER_KEYS = ["Tla", "TeamName", "TeamColour", "Line"]
STINT_KEYS = ["Compound", "TotalLaps", "New", "StartLaps"]


# ---------------------------------------------------------------- export

def items(x):
    return x.items() if isinstance(x, dict) else enumerate(x or [])


def slim_state(state: dict) -> dict:
    """Só as partes do estado que o painel mostra."""
    out = {k: state[k] for k in STATE_KEYS if k in state}
    out["DriverList"] = {num: {k: d[k] for k in DRIVER_KEYS if k in d}
                         for num, d in (state.get("DriverList") or {}).items() if isinstance(d, dict)}
    lines = {}
    for num, ln in ((state.get("TimingData") or {}).get("Lines") or {}).items():
        line = {k: ln[k] for k in LINE_KEYS if k in ln}
        if "Sectors" in line:
            line["Sectors"] = {str(i): {k: s[k] for k in SECTOR_KEYS if k in s}
                               for i, s in items(line["Sectors"]) if isinstance(s, dict)}
        lines[num] = line
    out["TimingData"] = {"Lines": lines}
    app = {}
    for num, ln in ((state.get("TimingAppData") or {}).get("Lines") or {}).items():
        stints = {str(i): {k: s[k] for k in STINT_KEYS if k in s} for i, s in items(ln.get("Stints")) if isinstance(s, dict)}
        app[num] = {"Stints": stints}
    out["TimingAppData"] = {"Lines": app}
    return out


def newest(prefix: str) -> Path:
    runs = sorted((ROOT / "runs").glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"nenhum run {prefix}_* em runs/")
    return runs[-1]


def export(gru: Path | None, naive: Path | None) -> None:
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from live.feed import ReplaySource

    shutil.rmtree(FILES, ignore_errors=True)
    for run in (gru or newest("gru"), naive or newest("naive")):
        dest = FILES / "runs" / run.name
        dest.mkdir(parents=True)
        for name in RUN_FILES:
            if (run / name).exists():
                shutil.copy2(run / name, dest / name)
        print(f"run {run.name}: {sum(f.stat().st_size for f in dest.iterdir()) / 1e6:.1f} MB")

    pred = pd.read_parquet(FILES / "runs" / (gru or newest("gru")).name / "predictions.parquet", columns=["race_label"])
    races = pred["race_label"].unique()
    laps = pd.read_parquet(ROOT / "data" / "processed" / "race_laps.parquet", columns=RACE_LAPS_COLUMNS)
    laps = laps[laps["race_label"].isin(races)]
    out = FILES / "data" / "processed" / "race_laps.parquet"
    out.parent.mkdir(parents=True)
    laps.to_parquet(out, index=False)
    print(f"race_laps: {len(laps):,} voltas de {laps['race_label'].nunique()} corridas · {out.stat().st_size / 1e6:.1f} MB")

    for session in REPLAY_SESSIONS:
        src, dest = ROOT / "data" / "live" / session, FILES / "data" / "live" / session
        dest.mkdir(parents=True)
        for name in ("session.json", "laps.parquet"):
            shutil.copy2(src / name, dest / name)
        if (src / "predictions.parquet").exists():
            pd.read_parquet(src / "predictions.parquet", columns=PREDICTION_COLUMNS).to_parquet(dest / "predictions.parquet", index=False)
        # o ReplaySource altera o mesmo estado a cada mensagem: cada momento precisa de uma cópia própria
        snaps = [[clock, slim_state(json.loads(json.dumps(state)))]
                 for clock, state in ReplaySource(str(src / "recording.jsonl"), interval=REPLAY_EVERY)]
        with gzip.open(dest / "replay.json.gz", "wt", encoding="utf-8", compresslevel=9) as f:
            json.dump(snaps, f, ensure_ascii=False, separators=(",", ":"))
        print(f"reprodução {session}: {len(snaps)} estados ({snaps[-1][0] / 60:.0f} min) · "
              f"{(dest / 'replay.json.gz').stat().st_size / 1e6:.1f} MB")

    assets = FILES / "assets"
    assets.mkdir()
    shutil.copy2(ROOT / "dashboard" / "assets" / "unb_logo.png", assets / "unb_logo.png")
    total = sum(f.stat().st_size for f in FILES.rglob("*") if f.is_file())
    print(f"demo/files: {total / 1e6:.1f} MB no total")


# ---------------------------------------------------------------- site

def site(out: Path) -> None:
    """index.html + app.zip (código das abas, entrada da demo e demo/files) + .nojekyll."""
    if not FILES.exists():
        sys.exit("demo/files não existe: rode antes `python demo/build.py export`")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy2(DEMO / "index.html", out / "index.html")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    with zipfile.ZipFile(out / "app.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(DEMO / "app.py", "demo_app.py")
        for rel in CODE:
            z.write(ROOT / rel, rel)
        for f in sorted(FILES.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(FILES).as_posix())
    print(f"site em {out}: app.zip com {len(zipfile.ZipFile(out / 'app.zip').namelist())} arquivos, "
          f"{(out / 'app.zip').stat().st_size / 1e6:.1f} MB")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="copia e enxuga os dados da demo para demo/files")
    e.add_argument("--gru", type=Path, help="pasta do run da GRU (padrão: a mais recente em runs/)")
    e.add_argument("--naive", type=Path, help="pasta do run do ingênuo (padrão: o mais recente)")
    s = sub.add_parser("site", help="monta o site estático")
    s.add_argument("out", type=Path, nargs="?", default=DEMO / "_site")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.gru, args.naive)
    else:
        site(args.out)


if __name__ == "__main__":
    main()
