"""Baseline ingênuo: repete a média do ritmo das últimas 3 voltas limpas em todos os horizontes.

Ao fim da volta t, a previsão das voltas t+1..t+10 é a média do ritmo relativo das 3 últimas voltas
limpas até t (`clean_live`, calculável ao vivo). Sem volta limpa ainda, não há previsão. Não tem
intervalo (p10 e p90 ficam vazios), nem treino: as previsões saem direto da validação e do teste.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.naive
"""
import argparse
import sys
from datetime import date
from pathlib import Path

from modeling.frame import target_frame
from modeling.results import (RUNS, compute_metrics, finish_predictions, git_commit, now_utc, print_summary,
                              write_run)

LAST_LAPS = 3


def naive_pace(laps):
    """Para cada volta, a média do ritmo das últimas LAST_LAPS voltas limpas até ela (inclusive)."""
    clean = laps.loc[laps["clean_live"], ["seq_id", "pace"]]
    mean = clean.groupby("seq_id")["pace"].transform(lambda x: x.rolling(LAST_LAPS, min_periods=1).mean())
    return mean.reindex(laps.index).groupby(laps["seq_id"]).ffill()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="dataset em data/processed (ex.: dataset_split5)")
    ap.add_argument("--out", type=Path, default=RUNS, help="pasta onde o run é gravado")
    ap.add_argument("--run-id", default=f"naive_{date.today():%Y%m%d}")
    args = ap.parse_args()
    run_id = args.run_id
    frame, laps, _, meta = target_frame(("val", "test"), args.dataset)
    frame["pace_p50"] = naive_pace(laps).to_numpy()[frame["anchor"]]
    pred = finish_predictions(frame, run_id, "naive")
    metrics = compute_metrics(pred)
    config = {
        "run_id": run_id, "model": "naive", "seed": None,
        "hyperparameters": {"last_clean_laps": LAST_LAPS},
        "eras_strategy": None, "features": ["pace", "clean_live"],
        "horizon": meta["horizon"], "quantiles": [0.5], "data": meta["splits"],
        "n_params": 0, "train_seconds": 0, "best_epoch": None,
        "device": "cpu", "git_commit": git_commit(), "created_at": now_utc(),
    }
    config["dataset"] = args.dataset
    out = write_run(run_id, config, pred, metrics, root=args.out)
    print(f"{run_id}: {len(pred):,} previsões ({pred['pace_p50'].notna().mean():.1%} com valor) -> {out}")
    print_summary(metrics)


if __name__ == "__main__":
    main()
