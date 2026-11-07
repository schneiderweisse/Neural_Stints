"""Bateria de treinos da GRU: sementes, divisões e uma busca pequena de hiperparâmetros.

Fase 1 · divisões: a configuração atual (estado 64, dropout 0,1, taxa 0,001) com 5 sementes em cada
         divisão: teste com as 3 últimas corridas (dataset) e com as 5 últimas (dataset_split5).
         A comparação é feita nas corridas de teste comuns às duas (Singapura, Austin e México).
Fase 2 · busca: estado {32, 64, 128} × dropout {0,1; 0,2} × taxa {0,001; 0,0003}, 3 sementes cada,
         na divisão de 3 corridas. A escolha usa só a validação (média da perda pinball).
Fase 3 · final: a configuração escolhida com 5 sementes, avaliada no teste.

Cada treino é um processo separado (modeling.gru) gravado em runs/<bateria>/<run_id>/. Rodar de novo
pula os treinos já terminados, então dá para interromper e retomar. Com --deadline, um treino só
começa se a duração estimada couber no prazo, e a busca vai em largura (todas as configurações com
uma semente, depois a próxima), para um corte levar sementes e não configurações. No fim grava
summary.csv e report.md na pasta da bateria.

Precisa da divisão alternativa: .venv/Scripts/python -m pipeline.sequences --n-test 5 --name dataset_split5

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.sweep --name sweep_20261106 --deadline 09:00
    .venv/Scripts/python -m modeling.sweep --name sweep_teste --smoke     # teste rápido (1 época)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from live.power import keep_awake
from modeling.results import RUNS

PY = sys.executable
SPLITS = {"split3": "dataset", "split5": "dataset_split5"}
BASE = (64, 0.1, 1e-3)
GRID = [(h, d, lr) for h in (32, 64, 128) for d in (0.1, 0.2) for lr in (1e-3, 3e-4)]
COMMON_TEST = ["2026 Singapore", "2026 Austin", "2026 Mexico City"]


def rid(split: str, cfg: tuple, seed: int) -> str:
    h, d, lr = cfg
    return f"gru_{split}_h{h}_d{d:g}_lr{lr:g}_s{seed}"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


class Sweep:
    def __init__(self, root: Path, extra: list[str], deadline: float | None = None):
        self.root, self.extra, self.deadline = root, extra, deadline
        self.minutes: dict[int, list[float]] = {}          # estado da GRU -> durações observadas (min)
        root.mkdir(parents=True, exist_ok=True)

    def estimate(self, hidden: int) -> float:
        """Duração esperada de um treino (min): a maior já vista com esse estado; sem histórico, escala pelo tamanho."""
        if self.minutes.get(hidden):
            return max(self.minutes[hidden])
        seen = [m for v in self.minutes.values() for m in v]
        base = max(seen) if seen else 10.0
        return base * max(1.0, hidden / 64)

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else (self.deadline - time.time()) / 60

    def done(self, run_id: str) -> bool:
        return (self.root / run_id / "metrics.json").exists()

    def naive(self, split: str) -> None:
        run_id = f"naive_{split}"
        if not self.done(run_id):
            subprocess.run([PY, "-m", "modeling.naive", "--dataset", SPLITS[split], "--out", str(self.root),
                            "--run-id", run_id], check=True, capture_output=True)
            log(f"ingênuo {split} pronto")

    def train(self, split: str, cfg: tuple, seed: int) -> None:
        run_id = rid(split, cfg, seed)
        if self.done(run_id):
            return
        h, d, lr = cfg
        if self.estimate(h) > self.time_left() - 3:              # margem de 3 min para o relatório
            log(f"pulado {run_id}: não termina até o prazo (estimativa {self.estimate(h):.0f} min, "
                f"restam {self.time_left():.0f} min)")
            return
        t0 = time.time()
        cmd = [PY, "-m", "modeling.gru", "--hidden", str(h), "--dropout", str(d), "--lr", str(lr), "--seed", str(seed),
               "--dataset", SPLITS[split], "--out", str(self.root), "--run-id", run_id,
               "--reference", str(self.root / f"naive_{split}")] + self.extra
        limit = None if self.deadline is None else max(60.0, (self.time_left() - 2) * 60)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=limit)
        except subprocess.TimeoutExpired:
            log(f"interrompido {run_id}: chegou ao prazo")
            return
        if r.returncode != 0:
            log(f"FALHOU {run_id}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
            return
        self.minutes.setdefault(h, []).append((time.time() - t0) / 60)
        m = json.loads((self.root / run_id / "metrics.json").read_text(encoding="utf-8"))
        log(f"{run_id}: validação pinball {m['splits']['val']['overall']['pinball_s']:.4f} · "
            f"MAE {m['splits']['val']['overall']['mae_s']:.3f} s · {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------ resumo
    def summary(self) -> pd.DataFrame:
        rows = []
        for d in sorted(self.root.glob("gru_*")):
            if not (d / "metrics.json").exists():
                continue
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            hp, val, test = c["hyperparameters"], m["splits"]["val"], m["splits"]["test"]
            common = [test["by_race"][r] for r in COMMON_TEST if r in test["by_race"]]
            n = sum(r["n"] for r in common)
            rows.append({
                "run_id": d.name, "split": d.name.split("_")[1], "hidden": hp["hidden"], "dropout": hp["dropout"],
                "lr": hp["lr"], "seed": c["seed"], "best_epoch": c["best_epoch"], "minutes": c["train_seconds"] / 60,
                "n_params": c["n_params"],
                "val_pinball": val["overall"]["pinball_s"], "val_mae": val["overall"]["mae_s"],
                "val_coverage": val["overall"]["coverage_80"],
                "test_mae": test["overall"]["mae_s"], "test_coverage": test["overall"]["coverage_80"],
                "test_skill": test.get("skill_vs_reference", {}).get("overall"),
                "test_after_pit_mae": test["by_same_stint"].get("False", {}).get("mae_s"),
                "common3_mae": sum(r["mae_s"] * r["n"] for r in common) / n if n else np.nan,
                "common3_coverage": sum(r["coverage_80"] * r["n"] for r in common) / n if n else np.nan,
            })
        return pd.DataFrame(rows)

    def best_config(self, s: pd.DataFrame, min_seeds: int) -> tuple:
        """Configuração com a menor perda pinball média na validação, entre as que têm todas as sementes."""
        g = s[s["split"] == "split3"].groupby(["hidden", "dropout", "lr"])
        agg = g.agg(n=("seed", "size"), val_pinball=("val_pinball", "mean")).reset_index()
        if agg.empty:
            return BASE
        need = min(min_seeds, int(agg["n"].max()))
        agg = agg[agg["n"] >= max(need, 1)].sort_values("val_pinball")
        r = agg.iloc[0]
        return (int(r["hidden"]), float(r["dropout"]), float(r["lr"]))

    def report(self, s: pd.DataFrame, best: tuple) -> str:
        ms = lambda x: f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
        pct = lambda x: f"{100 * x.mean():.1f}% ± {100 * x.std(ddof=1):.1f}" if len(x) > 1 else f"{100 * x.mean():.1f}%"
        base = s[(s["hidden"] == BASE[0]) & (s["dropout"] == BASE[1]) & (s["lr"] == BASE[2])]
        lines = [f"# Bateria de treinos da GRU ({self.root.name})", "",
                 f"{len(s)} treinos. Valores como média ± desvio-padrão entre sementes.", "",
                 "## Fase 1 · divisão de 3 corridas × divisão de 5 corridas (configuração atual)", "",
                 "Erro nas corridas de teste comuns às duas divisões (Singapura, Austin e México).", "",
                 "| Divisão | Sementes | MAE nas 3 corridas (s) | Cobertura p10–p90 | Melhor época |",
                 "|---|---|---|---|---|"]
        for split, g in base.groupby("split"):
            lines.append(f"| {split} | {len(g)} | {ms(g['common3_mae'])} | {pct(g['common3_coverage'])} | "
                         f"{g['best_epoch'].mean():.1f} |")
        g3, g5 = base[base["split"] == "split3"]["common3_mae"], base[base["split"] == "split5"]["common3_mae"]
        if len(g3) > 1 and len(g5) > 1:
            diff = g3.mean() - g5.mean()
            se = np.sqrt(g3.var(ddof=1) / len(g3) + g5.var(ddof=1) / len(g5))
            lines += ["", f"Diferença (split3 − split5): {diff:+.3f} s, erro-padrão {se:.3f} s "
                      f"(|diferença| / erro-padrão = {abs(diff) / se:.1f})."]
        grid = s[s["split"] == "split3"].groupby(["hidden", "dropout", "lr"])
        lines += ["", "## Fase 2 · busca na validação (divisão de 3 corridas)", "",
                  "Ordenado pela perda pinball média na validação, o critério de escolha. O teste não é usado aqui.", "",
                  "| Estado | Dropout | Taxa | Sementes | Pinball validação (s) | MAE validação (s) | Cobertura validação | "
                  "Melhor época | Minutos | Parâmetros |", "|---|---|---|---|---|---|---|---|---|---|"]
        order = grid["val_pinball"].mean().sort_values().index
        for key in order:
            g = grid.get_group(key)
            mark = " **escolhida**" if tuple(key) == best else ""
            lines.append(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {len(g)} | {ms(g['val_pinball'])}{mark} | {ms(g['val_mae'])} | "
                         f"{pct(g['val_coverage'])} | {g['best_epoch'].mean():.1f} | {g['minutes'].mean():.1f} | "
                         f"{int(g['n_params'].iloc[0]):,} |".replace(",", "."))
        fin = s[(s["split"] == "split3") & (s["hidden"] == best[0]) & (s["dropout"] == best[1]) & (s["lr"] == best[2])]
        b3 = base[base["split"] == "split3"]
        lines += ["", "## Fase 3 · teste (Singapura, Austin e México)", "",
                  "| Configuração | Sementes | MAE teste (s) | Melhora sobre o ingênuo | Cobertura p10–p90 | Depois de parada (s) |",
                  "|---|---|---|---|---|---|"]
        for name, g in ((f"escolhida (estado {best[0]}, dropout {best[1]:g}, taxa {best[2]:g})", fin),
                        ("atual (estado 64, dropout 0,1, taxa 0,001)", b3)):
            lines.append(f"| {name} | {len(g)} | {ms(g['test_mae'])} | {pct(g['test_skill'])} | {pct(g['test_coverage'])} | "
                         f"{ms(g['test_after_pit_mae'])} |")
        if fin.empty:
            return "\n".join(lines) + "\n"
        members = " ".join(f"runs/{self.root.name}/{r}" for r in fin.sort_values("seed")["run_id"])
        lines += ["", "## Modelo sugerido para o ao vivo", "",
                  f"A média dos quantis das {len(fin)} sementes da configuração escolhida (ensemble). Para criá-lo:", "",
                  "```bash", f".venv/Scripts/python -m modeling.ensemble {members}", "```", "",
                  "O comando mostra o erro do ensemble na validação e no teste. O previsor e o painel usam a GRU mais "
                  "recente em runs/, então o ensemble passa a ser o usado."]
        return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"sweep_{datetime.now():%Y%m%d}")
    ap.add_argument("--smoke", action="store_true", help="teste rápido: 1 época, 1 semente, 2 configurações")
    ap.add_argument("--deadline", help="horário limite HH:MM (hoje, ou amanhã se já passou): nenhum treino começa "
                                       "se não der para terminar antes")
    args = ap.parse_args()
    extra = ["--max-epochs", "1"] if args.smoke else []
    seeds5, seeds3 = ([1], [1]) if args.smoke else ([1, 2, 3, 4, 5], [1, 2, 3])
    grid = [BASE, (32, 0.1, 1e-3)] if args.smoke else GRID
    deadline = None
    if args.deadline:
        hh, mm = map(int, args.deadline.split(":"))
        target = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= datetime.now():
            target += timedelta(days=1)
        deadline = target.timestamp()
    sw = Sweep(RUNS / args.name, extra, deadline)
    keep_awake()
    log(f"bateria em {sw.root}" + (f" · prazo {datetime.fromtimestamp(deadline):%d/%m %H:%M}" if deadline else ""))
    for split in SPLITS:
        sw.naive(split)

    log("fase 1: divisões")
    for split in SPLITS:
        for seed in seeds5:
            sw.train(split, BASE, seed)

    log("fase 2: busca na validação (em largura: todas as configurações por semente)")
    for seed in seeds3:
        for cfg in grid:
            sw.train("split3", cfg, seed)

    s = sw.summary()
    if s.empty:
        log("nenhum treino terminou; nada a resumir")
        return
    # a escolha é gravada quando a fase 2 está completa: retomar a bateria não escolhe de novo (senão as
    # sementes extras da fase 3 mudariam a escolha a cada execução)
    choice = sw.root / "phase3.json"
    if choice.exists():
        best = tuple(json.loads(choice.read_text(encoding="utf-8"))["config"])
        log(f"fase 3: configuração já escolhida {best} ({choice.name})")
    else:
        best = sw.best_config(s, len(seeds3))
        complete = {rid("split3", cfg, seed) for cfg in grid for seed in seeds3} <= set(s["run_id"])
        if complete:
            choice.write_text(json.dumps({"config": list(best), "chosen_at": datetime.now().isoformat(timespec="seconds")}),
                              encoding="utf-8")
        log(f"fase 3: configuração escolhida pela validação {best}" + ("" if complete else " (provisória: fase 2 incompleta)"))
    for seed in seeds5:
        sw.train("split3", best, seed)

    s = sw.summary()
    s.to_csv(sw.root / "summary.csv", index=False)
    (sw.root / "report.md").write_text(sw.report(s, best), encoding="utf-8")
    log(f"fim: {len(s)} treinos · relatório em {sw.root / 'report.md'}")


if __name__ == "__main__":
    main()
