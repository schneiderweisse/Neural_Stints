"""Experimento das eras de regulamento: como usar as temporadas 2022–25 para prever 2026.

A validação e o teste são corridas de 2026, o primeiro ano do regulamento novo. Todas as estratégias
usam a GRU escolhida na bateria (estado 32, dropout 0,1, taxa 0,001), o mesmo dataset e 5 sementes:

- pooled: todas as eras juntas, com o embedding de regulamento (o modelo em uso; bateria sweep_20261106);
- pooled_noreg: todas as eras, sem o embedding de regulamento. Mede se o condicionamento ajuda;
- 2026_only: só as 11 corridas de 2026 do treino;
- pre_2022_25: só 2022–25, sem nenhuma corrida de 2026 (transferência direta para a era nova);
- pretrain_finetune: parte do pre_2022_25 da mesma semente e continua só em 2026;
- pooled_finetune: parte do pooled da mesma semente e continua só em 2026.

Categorias que um run não viu no treino (circuitos, pilotos, equipes-ano, regulamento) viram <unk>,
como ao vivo (modeling.gru). Com poucas sequências (2026_only e os fine-tunings, 241), cada época tem
só 8 lotes, então a paciência sobe para 20 épocas e o limite para 300; no fine-tuning a taxa cai para
0,0003. Os dois fine-tunings rodam também com taxas de 0,0001 e 0,00003, para ver se o resultado depende
da taxa. A estratégia é escolhida pela média da perda pinball na validação; o teste só descreve.

Grava runs/eras_<data>/: os treinos, um ensemble de 5 sementes por estratégia (média dos quantis),
summary.csv, finetune_lr.csv, ensembles.json e report.md. Rodar de novo pula os treinos prontos.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.eras --workers 3
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from live.power import keep_awake
from modeling.cells import GRU_ENS, GRU_SWEEP, SEEDS, log, ms, rid, train
from modeling.compare import block_bootstrap, common_rows
from modeling.ensemble import build
from modeling.results import RUNS, latest_run, sum5_error, summarize

HIDDEN, DROPOUT = 32, 0.1
SMALL = ["--max-epochs", "300", "--patience", "20"]
STRATEGIES = {
    "pooled": "Todas as eras, com embedding de regulamento (em uso)",
    "pooled_noreg": "Todas as eras, sem embedding de regulamento",
    "2026_only": "Só 2026",
    "pre_2022_25": "Só 2022–25 (nenhuma corrida de 2026)",
    "pretrain_finetune": "Pré-treino em 2022–25 + fine-tuning em 2026",
    "pooled_finetune": "Todas as eras + fine-tuning em 2026",
}
FINETUNES = {"pretrain_finetune": "pre_2022_25", "pooled_finetune": "pooled"}   # fine-tuning -> ponto de partida
FT_LRS = ("0.0003", "0.0001", "0.00003")       # a primeira é a das estratégias; as outras só checam a taxa


def run_dir(root, strategy: str, seed: int, lr: str = FT_LRS[0]):
    if strategy == "pooled":
        return RUNS / GRU_SWEEP / rid("gru", HIDDEN, DROPOUT, seed, "split3_")
    return root / (f"{strategy}_s{seed}" if lr == FT_LRS[0] else f"{strategy}_lr{lr}_s{seed}")


def extra_args(root, strategy: str, seed: int, lr: str = FT_LRS[0]) -> list[str]:
    base = ["--eras-strategy", strategy]
    if strategy == "pooled_noreg":
        return base + ["--zero-cat", "regulation"]
    if strategy == "2026_only":
        return base + ["--train-years", "2026"] + SMALL
    if strategy == "pre_2022_25":
        return base + ["--train-years", "2022", "2023", "2024", "2025"]
    start = run_dir(root, FINETUNES[strategy], seed)
    return base + ["--train-years", "2026", "--init", str(start), "--lr", lr] + SMALL


def row(strategy: str, seed: int, d) -> dict:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
    val, test = m["splits"]["val"], m["splits"]["test"]
    return {"strategy": strategy, "seed": seed, "run": d.relative_to(RUNS).as_posix(),
            "n_train_sequences": c["hyperparameters"].get("n_train_sequences"), "best_epoch": c["best_epoch"],
            "minutes": c["train_seconds"] / 60,
            "val_pinball": val["overall"]["pinball_s"], "val_mae": val["overall"]["mae_s"],
            "val_coverage": val["overall"]["coverage_80"],
            "test_pinball": test["overall"]["pinball_s"], "test_mae": test["overall"]["mae_s"],
            "test_coverage": test["overall"]["coverage_80"],
            "test_after_pit_mae": test["by_same_stint"].get("False", {}).get("mae_s")}


def report(root, s: pd.DataFrame, ft: pd.DataFrame, ens: dict, chosen: str) -> str:
    num = lambda v, f: f"{v:{f}}".replace(".", ",")
    order = s.groupby("strategy")["val_pinball"].mean().sort_values().index
    lines = [f"# Eras de regulamento ({root.name})", "",
             f"GRU de estado {HIDDEN} e dropout {num(DROPOUT, 'g')}, 5 sementes por estratégia; validação e teste "
             "são corridas de 2026. Valores como média ± desvio-padrão entre sementes.", "",
             "## Validação: a escolha", "",
             "| Estratégia | Sequências de treino | Pinball validação (s) | MAE validação (s) | Cobertura validação | "
             "Melhor época | Minutos |", "|---|---|---|---|---|---|---|"]
    for st in order:
        r = s[s["strategy"] == st]
        n = r["n_train_sequences"].iloc[0]
        mark = " **escolhida**" if st == chosen else ""
        minutes = r["minutes"].mean() + (s[s["strategy"] == "pre_2022_25"]["minutes"].mean()
                                        if st == "pretrain_finetune" else 0)
        lines.append(f"| {STRATEGIES[st]} | {'1.835' if pd.isna(n) else f'{int(n):,}'.replace(',', '.')} | "
                     f"{ms(r['val_pinball'], '.3f')}{mark} | {ms(r['val_mae'], '.3f')} | {ms(r['val_coverage'], 'pct')} | "
                     f"{num(r['best_epoch'].mean(), '.1f')} | {num(minutes, '.1f')} |")
    lines += ["", "Nos fine-tunings, a melhor época e as sequências são as da etapa em 2026; os minutos do "
              "pré-treino em 2022–25 + fine-tuning somam as duas etapas.", "",
              "## Teste, por semente", "",
              "| Estratégia | MAE teste (s) | Cobertura teste | Pinball teste (s) | Depois de parada (s) |",
              "|---|---|---|---|---|"]
    for st in order:
        r = s[s["strategy"] == st]
        lines.append(f"| {STRATEGIES[st]} | {ms(r['test_mae'], '.3f')} | {ms(r['test_coverage'], 'pct')} | "
                     f"{ms(r['test_pinball'], '.3f')} | {ms(r['test_after_pit_mae'], '.3f')} |")
    lines += ["", "## Ensembles de 5 sementes, nas mesmas voltas", "",
              "Média dos quantis das 5 sementes. Diferença de MAE para o pooled com intervalo de 95% por bootstrap "
              "em blocos piloto:corrida (modeling.compare); o bootstrap não reamostra sementes.", "",
              "| Estratégia | MAE val. (s) | MAE − pooled val. (IC 95%) | MAE teste (s) | MAE − pooled teste (IC 95%) | "
              "Cobertura val. / teste |", "|---|---|---|---|---|---|"]
    for st in order:
        v, t = ens["val"][st], ens["test"][st]
        ci = lambda m: ("—" if not m.get("boot") else
                        f"{num(m['boot']['diff_s'], '+.3f')} ({num(m['boot']['ci95_s'][0], '+.3f')} a "
                        f"{num(m['boot']['ci95_s'][1], '+.3f')})")
        lines.append(f"| {STRATEGIES[st]} | {num(v['overall']['mae_s'], '.3f')} | {ci(v)} | "
                     f"{num(t['overall']['mae_s'], '.3f')} | {ci(t)} | "
                     f"{num(v['overall']['coverage_80'], '.1%')} / {num(t['overall']['coverage_80'], '.1%')} |")
    lines += ["", "## Fine-tuning: a taxa de aprendizado", "",
              "Cada linha parte dos mesmos 5 runs; “sem fine-tuning” é o próprio ponto de partida. A melhor época "
              "é a do fine-tuning: o laço de treino começa na época 1 e não avalia o ponto de partida.", "",
              "| Ponto de partida | Taxa do fine-tuning | Pinball validação (s) | MAE validação (s) | MAE teste (s) | "
              "Melhor época |", "|---|---|---|---|---|---|"]
    for st, start in FINETUNES.items():
        r0 = s[s["strategy"] == start]
        lines.append(f"| {STRATEGIES[start]} | sem fine-tuning | {ms(r0['val_pinball'], '.3f')} | "
                     f"{ms(r0['val_mae'], '.3f')} | {ms(r0['test_mae'], '.3f')} | — |")
        for lr in FT_LRS:
            r = ft[(ft["strategy"] == st) & (ft["lr"] == lr)]
            if len(r):
                lines.append(f"| | {lr.replace('.', ',')} | {ms(r['val_pinball'], '.3f')} | {ms(r['val_mae'], '.3f')} | "
                             f"{ms(r['test_mae'], '.3f')} | {num(r['best_epoch'].mean(), '.1f')} |")
    races = {sp: list(next(iter(ens[sp].values()))["by_race"]) for sp in ("val", "test")}
    lines += ["", "## MAE por corrida dos ensembles (s)", "",
              "| Estratégia | " + " | ".join(races["val"] + races["test"]) + " |",
              "|---|" + "---|" * (len(races["val"]) + len(races["test"]))]
    for st in order:
        vals = [ens[sp][st]["by_race"][r] for sp in ("val", "test") for r in races[sp]]
        lines.append(f"| {STRATEGIES[st]} | " + " | ".join(num(x, ".3f") for x in vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"eras_{date.today():%Y%m%d}")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2, help="threads de cada treino em paralelo")
    args = ap.parse_args()
    root = RUNS / args.name
    root.mkdir(parents=True, exist_ok=True)
    keep_awake()

    def job(item):
        st, seed, lr = (*item, FT_LRS[0])[:3]
        d = run_dir(root, st, seed, lr)
        return train(root, "gru", HIDDEN, DROPOUT, seed, d.name, threads=args.threads,
                     extra=extra_args(root, st, seed, lr))

    first = [(st, s) for s in SEEDS for st in ("pre_2022_25", "pooled_noreg", "2026_only", "pooled_finetune")]
    with ThreadPoolExecutor(args.workers) as pool:              # o pré-treino sai antes, para liberar o fine-tuning
        list(pool.map(job, first))
        list(pool.map(job, [("pretrain_finetune", s) for s in SEEDS]))
        list(pool.map(job, [(st, s, lr) for lr in FT_LRS[1:] for st in FINETUNES for s in SEEDS]))

    s = pd.DataFrame([row(st, seed, run_dir(root, st, seed)) for st in STRATEGIES for seed in SEEDS
                      if (run_dir(root, st, seed) / "metrics.json").exists()])
    s.to_csv(root / "summary.csv", index=False)
    ft = pd.DataFrame([row(st, seed, run_dir(root, st, seed, lr)) | {"lr": lr}
                       for st in FINETUNES for lr in FT_LRS for seed in SEEDS
                       if (run_dir(root, st, seed, lr) / "metrics.json").exists()])
    ft.to_csv(root / "finetune_lr.csv", index=False)
    full = s.groupby("strategy").filter(lambda g: len(g) == len(SEEDS))
    chosen = full.groupby("strategy")["val_pinball"].mean().idxmin()
    log(f"estratégia escolhida pela validação: {chosen}")

    naive = latest_run("naive")
    names = {}
    for st in STRATEGIES:
        if st == "pooled":
            names[st] = GRU_ENS
            continue
        members = [run_dir(root, st, seed) for seed in SEEDS]
        if all((m / "metrics.json").exists() for m in members):
            build(members, f"{st}_ens{len(SEEDS)}", naive, root)
            names[st] = f"{root.name}/{st}_ens{len(SEEDS)}"
    data = common_rows([naive.name] + [names[st] for st in names if st != "pooled"] + [GRU_ENS])
    ens = {}
    for split in ("val", "test"):
        boot = block_bootstrap(data, split, GRU_ENS)
        ens[split] = {}
        for st, run in names.items():
            g = data[run][data[run]["split"] == split]
            ae = (g["time_p50"] - g["time_real"]).abs()
            ens[split][st] = {"run": run, "overall": summarize(g), "sum5": sum5_error(g), "boot": boot.get(run),
                              "by_race": {k: float(v) for k, v in ae.groupby(g["race_label"], sort=False).mean().items()}}
    (root / "ensembles.json").write_text(json.dumps(ens, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(root, s, ft, ens, chosen)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
