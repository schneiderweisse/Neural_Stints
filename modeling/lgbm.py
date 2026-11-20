"""Baseline de árvores: LightGBM com lags do ritmo, um modelo por quantil (p10, p50, p90).

Previsão direta: cada linha é uma volta prevista (janela t, horizonte h) e o horizonte é uma variável.
O alvo é a diferença entre o ritmo da volta t + h e a média das 3 últimas voltas limpas até t (o
ingênuo), em pontos percentuais; a previsão soma essa média de volta. Assim as árvores aprendem só o
quanto o ritmo vai mudar, e não o nível de cada piloto em cada pista.

Variáveis, todas calculáveis ao fim da volta t:
- histórico do ritmo limpo (`clean_live`) até t: última volta, médias de 3 e 5 voltas, desvio de 5
  voltas, média e número de voltas limpas no stint atual, inclinação das últimas 5 voltas do stint
  contra a idade do pneu, melhor volta da corrida e voltas desde a última limpa;
- estado na volta t: ritmo da volta (mesmo não limpa), posição, gaps, tráfego, bandeiras, clima,
  composto, idade do pneu e stint;
- conhecidas da volta t + h, como na GRU: horizonte, composto e número C (C1–C6), idade do pneu, volta
  do stint, pneu novo, progresso e voltas restantes, se é o mesmo stint e quantas paradas até lá;
- circuito e regulamento.

Treino nas janelas das corridas de treino com volta prevista limpa; a validação só escolhe o número
de árvores (early stopping na perda quantílica). Quantis cruzados são reordenados.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.lgbm
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from modeling.frame import target_frame
from modeling.naive import LAST_LAPS
from modeling.results import (QUANTILES, RUNS, compute_metrics, finish_predictions, git_commit, latest_run,
                              now_utc, print_summary, write_run)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
MAX_ROUNDS, PATIENCE = 3000, 100
ANCHOR_RAW = ["pace_in", "clean_live", "neutralised", "yellow", "traffic", "position", "gap_ahead",
              "gap_leader", "track_temp", "air_temp", "humidity", "tyre_life", "stint_lap", "stint"]
TARGET_RAW = ["tyre_life", "stint_lap", "fresh_tyre", "race_progress", "laps_remaining", "compound_c"]
CATEGORICAL = ["compound_a", "compound_t", "circuit", "regulation"]


def text(s: pd.Series) -> np.ndarray:
    """Categoria como texto, com os ausentes como 'NA' (o pyarrow mantém NaN no astype(str))."""
    return s.astype("string").fillna("NA").to_numpy(dtype=str)


def rolling_by(s: pd.Series, by, window: int, fn: str) -> pd.Series:
    return getattr(s.groupby(by, sort=False).rolling(window, min_periods=1), fn)().reset_index(level=0, drop=True)


def history_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Resumo do ritmo limpo até cada volta (inclusive), repetido nas voltas não limpas seguintes."""
    c = laps.loc[laps["clean_live"].to_numpy(bool), ["seq_id", "stint", "lap", "pace", "tyre_life"]].copy()
    c["pace"] = c["pace"] * 100
    st = [c["seq_id"], c["stint"]]
    out = pd.DataFrame(index=c.index)
    out["h_last"] = c["pace"]
    out["h_mean3"] = rolling_by(c["pace"], c["seq_id"], LAST_LAPS, "mean")
    out["h_mean5"] = rolling_by(c["pace"], c["seq_id"], 5, "mean")
    out["h_std5"] = rolling_by(c["pace"], c["seq_id"], 5, "std")
    out["h_best"] = c.groupby("seq_id", sort=False)["pace"].cummin()
    out["h_lap"] = c["lap"]
    # Stint atual: média, número de voltas limpas e inclinação das últimas 5 contra a idade do pneu
    stint = pd.DataFrame(index=c.index)
    stint["s_mean"] = c.groupby(st, sort=False)["pace"].transform(lambda x: x.expanding().mean())
    stint["s_n"] = c.groupby(st, sort=False).cumcount() + 1
    x, y = c["tyre_life"].astype(float), c["pace"]
    key = c["seq_id"] + "|" + c["stint"].astype(str)
    sx, sy = rolling_by(x, key, 5, "sum"), rolling_by(y, key, 5, "sum")
    sxy, sxx = rolling_by(x * y, key, 5, "sum"), rolling_by(x * x, key, 5, "sum")
    n = rolling_by(y, key, 5, "count")
    den = n * sxx - sx ** 2
    stint["s_slope"] = ((n * sxy - sx * sy) / den).where((n >= 3) & (den > 0))
    stint["s_key"] = key

    full = out.reindex(laps.index).groupby(laps["seq_id"], sort=False).ffill()
    lap_key = laps["seq_id"] + "|" + laps["stint"].astype(str)
    sf = stint.drop(columns="s_key").reindex(laps.index).groupby(lap_key, sort=False).ffill()
    full = pd.concat([full, sf], axis=1)
    full["h_since"] = laps["lap"] - full.pop("h_lap")
    return full


def features(frame: pd.DataFrame, laps: pd.DataFrame, hist: pd.DataFrame, cats: dict) -> pd.DataFrame:
    a, r = frame["anchor"].to_numpy(), frame["row"].to_numpy()
    A, T = laps.iloc[a].reset_index(drop=True), laps.iloc[r].reset_index(drop=True)
    X = hist.iloc[a].reset_index(drop=True)
    X["h"] = frame["h"].to_numpy()
    for c in ANCHOR_RAW:
        X[f"a_{c}"] = A[c].astype(float).to_numpy()
    X["a_pace_in"] = X["a_pace_in"] * 100
    for c in TARGET_RAW:
        X[f"t_{c}"] = T[c].astype(float).to_numpy()
    X["same_stint"] = frame["same_stint"].astype(float).to_numpy()
    X["stops_ahead"] = (T["stint"].astype(float) - A["stint"].astype(float)).to_numpy()
    X["d_tyre"] = X["t_tyre_life"] - X["a_tyre_life"]
    X["compound_a"], X["compound_t"] = text(A["compound"]), text(T["compound"])
    X["circuit"], X["regulation"] = text(T["circuit"]), text(T["regulation"])
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def base_pace(X: pd.DataFrame, fallback: float) -> np.ndarray:
    """Nível de partida (pp): a média das 3 últimas voltas limpas, ou a mediana do treino sem histórico."""
    return X["h_mean3"].fillna(fallback).to_numpy()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="dataset em data/processed (ex.: dataset_split5)")
    ap.add_argument("--out", type=Path, default=RUNS, help="pasta onde o run é gravado")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop", nargs="*", default=[], help="variáveis a deixar de fora (ablação)")
    ap.add_argument("--reference", type=Path, help="pasta do run do ingênuo para o skill (padrão: o mais recente)")
    args = ap.parse_args()
    run_id = args.run_id or f"lgbm_{date.today():%Y%m%d}"

    frame_all, laps, d, meta = target_frame(("train", "val", "test"), args.dataset)
    hist = history_features(laps)
    compounds = sorted(set(text(laps["compound"])))
    cats = {"compound_a": compounds, "compound_t": compounds,
            "circuit": sorted(set(text(laps["circuit"]))), "regulation": sorted(set(text(laps["regulation"])))}
    X_all = features(frame_all, laps, hist, cats).drop(columns=args.drop)
    categorical = [c for c in CATEGORICAL if c not in args.drop]
    clean_target = frame_all["clean"].to_numpy(bool) & np.isfinite(frame_all["pace_real"].to_numpy())
    fallback = float(np.median(laps.loc[laps["clean"].to_numpy(bool), "pace"]) * 100)
    base = base_pace(X_all, fallback)
    y_all = frame_all["pace_real"].to_numpy() * 100 - base

    split = frame_all["split"].to_numpy()
    tr, va = (split == "train") & clean_target, (split == "val") & clean_target
    print(f"{run_id}: {tr.sum():,} linhas de treino, {va.sum():,} de validação, {X_all.shape[1]} variáveis")
    dtrain = lgb.Dataset(X_all[tr], y_all[tr], categorical_feature=categorical, free_raw_data=False)
    dval = lgb.Dataset(X_all[va], y_all[va], reference=dtrain, categorical_feature=categorical)

    keep = split != "train"
    frame = frame_all[keep].reset_index(drop=True)
    X_eval, base_eval = X_all[keep].reset_index(drop=True), base[keep]
    boosters, best_rounds, t0 = {}, {}, time.perf_counter()
    for col, q in QUANTILES.items():
        params = PARAMS | {"objective": "quantile", "alpha": q, "metric": "quantile", "seed": args.seed}
        booster = lgb.train(params, dtrain, MAX_ROUNDS, valid_sets=[dval], valid_names=["val"],
                            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        boosters[col], best_rounds[col] = booster, int(booster.best_iteration)
        frame[col] = (base_eval + booster.predict(X_eval, num_iteration=booster.best_iteration)) / 100
        print(f"  {col}: {booster.best_iteration} árvores · perda quantílica na validação "
              f"{booster.best_score['val']['quantile']:.4f} pp · {time.perf_counter() - t0:.0f} s")
    train_seconds = time.perf_counter() - t0
    q = np.sort(frame[list(QUANTILES)].to_numpy(), axis=1)            # sem quantis cruzados
    frame[list(QUANTILES)] = q

    pred = finish_predictions(frame, run_id, "lgbm")
    naive_dir = args.reference or latest_run("naive")
    reference = pd.read_parquet(naive_dir / "predictions.parquet") if naive_dir else None
    metrics = compute_metrics(pred, reference)
    if naive_dir:
        metrics["reference_run"] = naive_dir.name
    gain = boosters["pace_p50"].feature_importance("gain")
    metrics["importance_p50_gain"] = {k: float(v) for k, v in sorted(zip(X_all.columns, gain / gain.sum()),
                                                                      key=lambda kv: -kv[1])}
    config = {
        "run_id": run_id, "model": "lgbm", "seed": args.seed,
        "hyperparameters": PARAMS | {"max_rounds": MAX_ROUNDS, "patience": PATIENCE, "best_rounds": best_rounds,
                                     "target": f"ritmo(t+h) − média das {LAST_LAPS} últimas voltas limpas, em pp",
                                     "fallback_base_pp": fallback},
        "eras_strategy": "pooled", "features": list(X_all.columns), "categorical": categorical,
        "horizon": meta["horizon"], "quantiles": list(QUANTILES.values()), "data": meta["splits"],
        "n_params": None, "n_trees": int(sum(b.num_trees() for b in boosters.values())),
        "train_seconds": round(train_seconds, 1), "best_epoch": None,
        "device": "cpu", "git_commit": git_commit(), "created_at": now_utc(), "dataset": args.dataset,
    }
    out = write_run(run_id, config, pred, metrics, root=args.out)
    print(f"{run_id}: {len(pred):,} previsões -> {out}")
    print_summary(metrics)
    print("  variáveis mais importantes (p50, ganho): " + ", ".join(
        f"{k} {v:.0%}" for k, v in list(metrics["importance_p50_gain"].items())[:8]))


if __name__ == "__main__":
    main()
