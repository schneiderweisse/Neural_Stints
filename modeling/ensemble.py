"""Ensemble de GRUs: média dos quantis de várias sementes já treinadas.

Os membros são runs da GRU treinados no mesmo dataset (por exemplo as 5 sementes de uma configuração
da bateria). A previsão do ensemble, volta a volta, é a média dos p10, p50 e p90 dos membros; a média
de quantis em ordem continua em ordem. O run gravado tem o formato padrão de docs/resultados.md
(previsões de validação e teste, métricas e as curvas de treino de cada membro) e um model.pt com os
pesos de todos os membros, que o live.predict usa como um modelo só.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.ensemble runs/sweep_20261106/gru_split3_h64_d0.1_lr0.001_s{1,2,3,4,5}
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import torch

from modeling.gru import latest_run
from modeling.results import (RUNS, compute_metrics, finish_predictions, git_commit, now_utc, print_summary,
                              write_atomic, write_run)

KEYS = ["split", "race_id", "driver", "t", "h"]
Q = ["pace_p10", "pace_p50", "pace_p90"]


def average(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Média dos quantis dos membros, nas mesmas linhas da tabela de alvos."""
    if any(len(p) != len(preds[0]) for p in preds):
        raise ValueError("os membros não têm as mesmas linhas de previsão (datasets diferentes?)")
    stack = pd.concat([p.set_index(KEYS)[Q] for p in preds])
    mean = stack.groupby(level=KEYS).mean().reset_index()
    base = preds[0].drop(columns=Q + [q.replace("pace", "time") for q in Q])
    return base.merge(mean, on=KEYS, how="left", validate="one_to_one")


def build(members: list[Path], run_id: str, reference: Path | None, out_root: Path) -> Path:
    cks = [torch.load(m / "model.pt", weights_only=False) for m in members]
    metas = {json.dumps(ck["dataset_meta"], sort_keys=True) for ck in cks}
    if len(metas) > 1:
        raise ValueError("os membros usam normalizações ou vocabulários diferentes; precisam do mesmo dataset")
    configs = [ck["config"] for ck in cks]

    pred = finish_predictions(average([pd.read_parquet(m / "predictions.parquet") for m in members]), run_id, "gru")
    ref = pd.read_parquet(reference / "predictions.parquet") if reference else None
    metrics = compute_metrics(pred, ref)
    if reference:
        metrics["reference_run"] = reference.name

    first = configs[0]
    same_hp = all(c["hyperparameters"] == first["hyperparameters"] for c in configs)
    config = {
        "run_id": run_id, "model": "gru", "seed": None,
        "ensemble": {"method": "média dos quantis", "members": [m.name for m in members],
                     "member_paths": [m.resolve().relative_to(RUNS.resolve()).as_posix()
                                      if m.resolve().is_relative_to(RUNS.resolve()) else m.as_posix() for m in members],
                     "seeds": [c["seed"] for c in configs], "best_epochs": [c["best_epoch"] for c in configs],
                     "n_params_member": first["n_params"]},
        "hyperparameters": first["hyperparameters"] if same_hp else None,
        "dataset": first.get("dataset", "dataset"),
        "eras_strategy": first["eras_strategy"], "features": first["features"], "horizon": first["horizon"],
        "quantiles": first["quantiles"], "data": first["data"],
        "n_params": sum(c["n_params"] for c in configs),
        "train_seconds": round(sum(c["train_seconds"] for c in configs), 1),
        "best_epoch": None, "device": first["device"], "git_commit": git_commit(), "created_at": now_utc(),
    }
    history = pd.concat([pd.read_csv(m / "history.csv").assign(seed=c["seed"])
                         for m, c in zip(members, configs) if (m / "history.csv").exists()], ignore_index=True)
    out = write_run(run_id, config, pred, metrics, history if len(history) else None, root=out_root)
    checkpoint = {"config": config, "dataset_meta": cks[0]["dataset_meta"],
                  "members": [{"state_dict": ck["state_dict"], "model_args": ck["model_args"], "config": ck["config"]}
                              for ck in cks]}
    write_atomic(out / "model.pt", lambda p: torch.save(checkpoint, p))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+", type=Path, help="pastas dos runs da GRU que entram na média")
    ap.add_argument("--run-id", help="padrão: gru_<data>_ens<n>")
    ap.add_argument("--reference", type=Path, help="pasta do run do ingênuo para o skill (padrão: o mais recente)")
    ap.add_argument("--out", type=Path, default=RUNS, help="pasta onde o run é gravado")
    args = ap.parse_args()
    run_id = args.run_id or f"gru_{date.today():%Y%m%d}_ens{len(args.members)}"
    out = build(args.members, run_id, args.reference or latest_run("naive"), args.out)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    print(f"{run_id}: média de {len(args.members)} membros -> {out}")
    for split in ("val", "test"):
        print(f"  pinball {split}: {m['splits'][split]['overall']['pinball_s']:.4f} s")
    print_summary(m)


if __name__ == "__main__":
    main()
