"""Bateria LSTM × GRU: a mesma rede com a célula trocada, várias sementes, escolha pela validação.

A GRU já tem a sua bateria (runs/sweep_20261106). Aqui entram as LSTMs, com os mesmos dados (divisão
de 3 corridas), a mesma taxa de aprendizado (0,001) e o mesmo treino (modeling.gru --cell lstm):
- estado 32, dropout 0,1: a configuração escolhida para a GRU. Mesmo tamanho de estado; a LSTM tem
  uma porta a mais e fica com 34% mais parâmetros;
- estado 26, dropout 0,1: o mesmo número de parâmetros da GRU escolhida (~15 mil);
- estado 64, dropout 0,1 e estado 32, dropout 0,2: vizinhas, para a LSTM não ficar presa à
  configuração que é a melhor para a GRU.
Cinco sementes em cada. A LSTM é escolhida só pela validação (média da perda pinball), o mesmo
critério da bateria da GRU; o teste só entra no fim, para as duas escolhidas.

Etapas:
1. tempo por época, com um treino de cada vez (3 épocas da GRU 32, da LSTM 32 e da LSTM 26);
2. a bateria, com vários treinos em paralelo (--workers, 2 threads cada); rodar de novo pula os
   treinos prontos, então dá para interromper e retomar;
3. o ensemble das 5 sementes da LSTM escolhida (runs/lstm_<data>_ens5), comparado com o da GRU
   (gru_20261106_ens5) nas mesmas voltas, com o bootstrap por piloto:corrida de modeling.compare;
4. summary.csv e report.md em runs/cells_<data>/.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.cells --workers 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import numpy as np
import pandas as pd

from live.power import keep_awake
from modeling.compare import block_bootstrap, common_rows
from modeling.ensemble import build
from modeling.results import RUNS, latest_run, sum5_error, summarize

PY = sys.executable
GRU_SWEEP, GRU_ENS = "sweep_20261106", "gru_20261106_ens5"
GRU_CHOSEN = (32, 0.1)
LSTM_CONFIGS = [(32, 0.1), (26, 0.1), (64, 0.1), (32, 0.2)]
SEEDS = (1, 2, 3, 4, 5)
LR = 0.001
TIMING = [("gru", 32), ("lstm", 32), ("lstm", 26)]
TIMING_EPOCHS = 3


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def rid(cell: str, hidden: int, dropout: float, seed: int, split: str = "") -> str:
    return f"{cell}_{split}h{hidden}_d{dropout:g}_lr{LR:g}_s{seed}"


def train(root, cell, hidden, dropout, seed, run_id, threads=None, extra=()) -> dict | None:
    if (root / run_id / "metrics.json").exists():
        return None
    env = os.environ | ({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)} if threads else {})
    cmd = [PY, "-m", "modeling.gru", "--cell", cell, "--hidden", str(hidden), "--dropout", str(dropout),
           "--lr", str(LR), "--seed", str(seed), "--out", str(root), "--run-id", run_id, *extra]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        log(f"FALHOU {run_id}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
        return None
    m = json.loads((root / run_id / "metrics.json").read_text(encoding="utf-8"))["splits"]["val"]["overall"]
    log(f"{run_id}: validação pinball {m['pinball_s']:.4f} · MAE {m['mae_s']:.3f} s · {(time.time() - t0) / 60:.1f} min")
    return m


def run_row(d) -> dict:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
    hp, val, test = c["hyperparameters"], m["splits"]["val"], m["splits"]["test"]
    return {"run_id": d.name, "cell": c["model"], "hidden": hp["hidden"], "dropout": hp["dropout"], "seed": c["seed"],
            "n_params": c["n_params"], "best_epoch": c["best_epoch"],
            "val_pinball": val["overall"]["pinball_s"], "val_mae": val["overall"]["mae_s"],
            "val_coverage": val["overall"]["coverage_80"],
            "test_pinball": test["overall"]["pinball_s"], "test_mae": test["overall"]["mae_s"],
            "test_coverage": test["overall"]["coverage_80"],
            "test_skill": test.get("skill_vs_reference", {}).get("overall"),
            "test_after_pit_mae": test["by_same_stint"].get("False", {}).get("mae_s")}


def summary(root) -> pd.DataFrame:
    gru = [RUNS / GRU_SWEEP / rid("gru", h, d, s, "split3_") for h, d in [GRU_CHOSEN, (64, 0.1), (32, 0.2)]
           for s in SEEDS]
    lstm = [root / rid("lstm", h, d, s) for h, d in LSTM_CONFIGS for s in SEEDS]
    return pd.DataFrame([run_row(p) for p in gru + lstm if (p / "metrics.json").exists()])


def ms(x: pd.Series, fmt: str) -> str:
    """média ± desvio entre sementes, com vírgula decimal."""
    if fmt == "pct":
        out = f"{x.mean():.1%} ± {x.std() * 100:.1f}"
    else:
        out = f"{x.mean():{fmt}} ± {x.std():{fmt}}"
    return out.replace(".", ",")


def report(root, s: pd.DataFrame, timing: dict, ens: dict | None, chosen: tuple) -> str:
    g = s.groupby(["cell", "hidden", "dropout"])
    agg = g.agg(n=("seed", "size"), params=("n_params", "first"), pinball=("val_pinball", "mean")).reset_index()
    agg = agg.sort_values("pinball")
    lines = [f"# LSTM × GRU ({root.name})", "",
             "A mesma rede encoder–decoder com a célula trocada: mesmos dados (divisão de 3 corridas), taxa 0,001, "
             "5 sementes por configuração. Valores como média ± desvio-padrão entre sementes. As GRUs vêm da bateria "
             f"`{GRU_SWEEP}`.", "",
             "## Validação: a escolha", "",
             "Ordenado pela perda pinball média na validação, o critério de escolha. O teste não é usado aqui.", "",
             "| Célula | Estado | Dropout | Parâmetros | Sementes | Pinball validação (s) | MAE validação (s) | "
             "Cobertura validação | Melhor época |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, a in agg.iterrows():
        r = s[(s["cell"] == a["cell"]) & (s["hidden"] == a["hidden"]) & (s["dropout"] == a["dropout"])]
        key = (a["cell"], int(a["hidden"]), float(a["dropout"]))
        mark = " **escolhida**" if key in [("gru", *GRU_CHOSEN), ("lstm", *chosen)] else ""
        drop = f"{a['dropout']:g}".replace(".", ",")
        params = f"{int(a['params']):,}".replace(",", ".")
        epoch = f"{r['best_epoch'].mean():.1f}".replace(".", ",")
        lines.append(f"| {a['cell'].upper()} | {int(a['hidden'])} | {drop} | {params} | {int(a['n'])} | "
                     f"{ms(r['val_pinball'], '.3f')}{mark} | {ms(r['val_mae'], '.3f')} | {ms(r['val_coverage'], 'pct')} | "
                     f"{epoch} |")
    lines += ["", "## Teste: sementes", "",
              "| Configuração | Sementes | MAE teste (s) | Skill vs. ingênuo | Cobertura teste | Pinball teste (s) | "
              "Depois de parada (s) |", "|---|---|---|---|---|---|---|"]
    shown = [("gru", *GRU_CHOSEN, "GRU escolhida")] + [("lstm", *chosen, "LSTM escolhida")]
    for cfg, label in [((32, 0.1), "LSTM com o mesmo estado da GRU"), ((26, 0.1), "LSTM com os mesmos parâmetros da GRU")]:
        if cfg != chosen:
            shown.append(("lstm", *cfg, label))
    for cell, h, d, label in shown:
        r = s[(s["cell"] == cell) & (s["hidden"] == h) & (s["dropout"] == d)]
        if r.empty:
            continue
        lines.append(f"| {label} (estado {h}, dropout {d:g}) | {len(r)} | {ms(r['test_mae'], '.3f')} | "
                     f"{ms(r['test_skill'], 'pct')} | {ms(r['test_coverage'], 'pct')} | {ms(r['test_pinball'], '.3f')} | "
                     f"{ms(r['test_after_pit_mae'], '.3f')} |".replace("dropout 0.", "dropout 0,"))
    a = s[(s["cell"] == "gru") & (s["hidden"] == GRU_CHOSEN[0]) & (s["dropout"] == GRU_CHOSEN[1])]
    b = s[(s["cell"] == "lstm") & (s["hidden"] == chosen[0]) & (s["dropout"] == chosen[1])]
    if len(a) > 1 and len(b) > 1:
        for split in ("val", "test"):
            diff = b[f"{split}_mae"].mean() - a[f"{split}_mae"].mean()
            se = np.sqrt(b[f"{split}_mae"].var() / len(b) + a[f"{split}_mae"].var() / len(a))
            num = lambda v, f: f"{v:{f}}".replace(".", ",")
            lines.append(f"\nMAE {'validação' if split == 'val' else 'teste'}, LSTM escolhida − GRU escolhida: "
                         f"{num(diff, '+.3f')} s, erro-padrão entre sementes {num(se, '.3f')} s "
                         f"(|diferença| / erro-padrão = {num(abs(diff) / se, '.1f')}).")
    if ens:
        lines += ["", "## Ensembles de 5 sementes, nas mesmas voltas", "",
                  "Média dos quantis das 5 sementes de cada célula. Intervalo da diferença por bootstrap em blocos "
                  "piloto:corrida (modeling.compare).", "",
                  "| Divisão | Modelo | MAE (s) | Pinball (s) | Cobertura 80% | Erro em 5 voltas (s) | MAE − GRU (IC 95%) |",
                  "|---|---|---|---|---|---|---|"]
        for split, name in (("val", "validação"), ("test", "teste")):
            for run, m in ens[split].items():
                o, b5 = m["overall"], m["sum5"]
                bt = m.get("boot")
                ci = (f"{bt['diff_s']:+.3f} ({bt['ci95_s'][0]:+.3f} a {bt['ci95_s'][1]:+.3f})" if bt else "—")
                lines.append(f"| {name} | `{run}` | {o['mae_s']:.3f} | {o['pinball_s']:.3f} | {o['coverage_80']:.1%} | "
                             f"{b5['mae_s']:.2f} | {ci} |".replace(".", ","))
    if timing:
        lines += ["", "## Tempo por época", "",
                  f"Um treino de cada vez, {TIMING_EPOCHS} épocas, na mesma máquina (CPU). A LSTM anda volta a volta no "
                  "encoder para guardar o estado de célula de cada volta.", "",
                  "| Célula | Estado | Parâmetros | Segundos por época |", "|---|---|---|---|"]
        for (cell, h), (sec, params) in timing.items():
            lines.append(f"| {cell.upper()} | {h} | {f'{params:,}'.replace(',', '.')} | {f'{sec:.1f}'.replace('.', ',')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"cells_{date.today():%Y%m%d}")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2, help="threads de cada treino em paralelo")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    root = RUNS / args.name
    root.mkdir(parents=True, exist_ok=True)
    keep_awake()

    timing = {}
    if not args.skip_timing:
        for cell, h in TIMING:
            run_id = f"timing_{cell}_h{h}"
            train(root / "timing", cell, h, 0.1, 1, run_id, extra=["--max-epochs", str(TIMING_EPOCHS)])
            hist = pd.read_csv(root / "timing" / run_id / "history.csv")
            cfg = json.loads((root / "timing" / run_id / "config.json").read_text(encoding="utf-8"))
            timing[(cell, h)] = (float(hist["seconds"].mean()), int(cfg["n_params"]))
            log(f"tempo {cell} {h}: {timing[(cell, h)][0]:.1f} s por época")

    jobs = [(h, d, s) for s in SEEDS for h, d in LSTM_CONFIGS]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda j: train(root, "lstm", j[0], j[1], j[2], rid("lstm", *j), threads=args.threads), jobs))

    s = summary(root)
    s.to_csv(root / "summary.csv", index=False)
    lstm = s[s["cell"] == "lstm"].groupby(["hidden", "dropout"]).agg(n=("seed", "size"), p=("val_pinball", "mean"))
    lstm = lstm[lstm["n"] == len(SEEDS)].sort_values("p")
    chosen = (int(lstm.index[0][0]), float(lstm.index[0][1]))
    log(f"LSTM escolhida pela validação: estado {chosen[0]}, dropout {chosen[1]:g}")

    members = [root / rid("lstm", *chosen, seed) for seed in SEEDS]
    ens_id = f"lstm_{date.today():%Y%m%d}_ens{len(SEEDS)}"
    build(members, ens_id, latest_run("naive"), RUNS)
    naive = latest_run("naive").name
    data = common_rows([naive, ens_id, GRU_ENS])
    ens = {}
    for split in ("val", "test"):
        boot = block_bootstrap(data, split, GRU_ENS)
        ens[split] = {}
        for run in (ens_id, GRU_ENS):
            g = data[run][data[run]["split"] == split]
            ens[split][run] = {"overall": summarize(g), "sum5": sum5_error(g), "boot": boot.get(run)}
    (root / "ensembles.json").write_text(json.dumps(ens, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(root, s, timing, ens, chosen)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
