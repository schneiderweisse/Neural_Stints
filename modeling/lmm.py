"""Baseline estatístico: modelo linear misto do ritmo, com o nível de cada piloto em cada corrida como
efeito aleatório.

    ritmo (%) = b0 + composto + idade do pneu × composto + progresso da corrida + temperatura da pista
                + u_piloto:corrida + v_piloto:corrida × progresso da corrida + erro

O composto entra como macio/médio/duro (médio na base), o progresso da corrida faz o papel do
combustível (o carro fica mais leve) e a temperatura é a da pista. Ajuste por REML (statsmodels
MixedLM) nas voltas limpas das corridas de treino, com o ritmo em pontos percentuais.

Previsão ao fim da volta t, para t + h:
- efeitos fixos com as covariáveis conhecidas da volta t + h (composto, idade do pneu e progresso da
  corrida, as mesmas que a GRU recebe do futuro) e a temperatura da pista da volta t, a última medida;
- efeito aleatório do par piloto:corrida estimado só com as voltas limpas até t (`clean_live`), pelo
  BLUP, a média condicional da normal:

      b̂ = (G⁻¹ + Z'Z/σ²)⁻¹ Z'r/σ²,   Var(b | voltas até t) = (G⁻¹ + Z'Z/σ²)⁻¹

  com r os resíduos dos efeitos fixos, G a covariância dos efeitos aleatórios e σ² a variância do
  erro. Com poucas voltas, b̂ fica perto de zero (o nível médio dos pilotos); com muitas, perto da
  média dos resíduos;
- quantis da normal com variância z'Var(b | voltas até t)z + σ², z = [1, progresso da corrida].

A inclinação aleatória no progresso da corrida deixa cada piloto ter a sua evolução ao longo da prova
(combustível, pista emborrachando, gestão). Foi a melhor variante na validação (docs/resultados.md):
o intercepto sozinho, o do plano, não acompanha essa evolução, e uma inclinação na idade do pneu,
estimada com as poucas voltas de um stint, extrapola mal nas voltas seguintes.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.lmm                       # intercepto + inclinação no progresso
    .venv/Scripts/python -m modeling.lmm --slopes              # só o intercepto (o do plano)
    .venv/Scripts/python -m modeling.lmm --slopes tyre_life    # outras variantes
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from modeling.frame import lap_splits, target_frame
from modeling.results import (QUANTILES, RUNS, compute_metrics, finish_predictions, git_commit, latest_run,
                              now_utc, print_summary, write_run)

FIXED = "pp ~ (soft + hard) * tyre_life + race_progress + track_temp_c"
FEATURES = ["compound", "tyre_life", "race_progress", "track_temp", "pace", "clean", "clean_live"]


def design(df: pd.DataFrame, temp_mean: float) -> pd.DataFrame:
    """Covariáveis do modelo. Idade do pneu ausente: a volta do stint; composto fora de seco: médio."""
    comp = df["compound"].astype(str)
    return pd.DataFrame({
        "soft": (comp == "SOFT").astype(float).to_numpy(),
        "hard": (comp == "HARD").astype(float).to_numpy(),
        "tyre_life": df["tyre_life"].fillna(df["stint_lap"]).fillna(1).astype(float).to_numpy(),
        "race_progress": df["race_progress"].astype(float).to_numpy(),
        "track_temp_c": (df["track_temp"].astype(float) - temp_mean).to_numpy(),
    })


def fixed_part(x: pd.DataFrame, params: pd.Series) -> np.ndarray:
    """Xβ com os nomes de coeficientes da fórmula (ex.: 'soft:tyre_life')."""
    out = np.full(len(x), params["Intercept"])
    for name, beta in params.drop("Intercept").items():
        out = out + beta * np.prod([x[t].to_numpy() for t in name.split(":")], axis=0)
    return out


def random_design(x: pd.DataFrame, slopes: list[str]) -> np.ndarray:
    """Z: a coluna do intercepto e uma por inclinação aleatória."""
    return np.column_stack([np.ones(len(x))] + [x[c].to_numpy() for c in slopes])


def online_blup(laps: pd.DataFrame, resid: np.ndarray, Z: np.ndarray, G: np.ndarray, sigma2: float):
    """BLUP do efeito aleatório de cada sequência com as voltas limpas até cada volta (inclusive).

    Devolve b̂ (n, q) e Var(b | dados) (n, q, q), q = 1 + número de inclinações aleatórias.
    """
    use = laps["clean_live"].to_numpy(bool) & np.isfinite(resid)
    q = G.shape[0]
    Z = Z * use[:, None]
    r = np.where(use, resid, 0.0)
    seq = laps["seq_id"].to_numpy()
    zz = np.einsum("ni,nj->nij", Z, Z).reshape(len(laps), q * q)
    zr = Z * r[:, None]
    zz = pd.DataFrame(zz).groupby(seq, sort=False).cumsum().to_numpy().reshape(len(laps), q, q)
    zr = pd.DataFrame(zr).groupby(seq, sort=False).cumsum().to_numpy()
    A = np.linalg.inv(G)[None] + zz / sigma2
    V = np.linalg.inv(A)
    b = np.einsum("nij,nj->ni", V, zr / sigma2)
    return b, V


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--slopes", nargs="*", default=["race_progress"], choices=["tyre_life", "race_progress"],
                    help="inclinações aleatórias por piloto:corrida, além do intercepto")
    ap.add_argument("--dataset", default="dataset", help="dataset em data/processed (ex.: dataset_split5)")
    ap.add_argument("--out", type=Path, default=RUNS, help="pasta onde o run é gravado")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--reference", type=Path, help="pasta do run do ingênuo para o skill (padrão: o mais recente)")
    args = ap.parse_args()
    run_id = args.run_id or f"lmm_{date.today():%Y%m%d}"

    frame, laps, d, meta = target_frame(("val", "test"), args.dataset)
    split = lap_splits(d)
    train = laps[(split == "train") & laps["clean"].to_numpy(bool)]
    temp_mean = float(train["track_temp"].mean())
    data = design(train, temp_mean)
    data["pp"] = train["pace"].to_numpy() * 100
    data["seq_id"] = train["seq_id"].to_numpy()
    re_formula = "~" + " + ".join(args.slopes) if args.slopes else "1"
    print(f"{run_id}: {len(data):,} voltas limpas de treino, {data['seq_id'].nunique():,} pares piloto:corrida")

    t0 = time.perf_counter()
    model = smf.mixedlm(FIXED, data, groups=data["seq_id"], re_formula=re_formula)
    fit = model.fit(reml=True, method=["lbfgs"])
    train_seconds = time.perf_counter() - t0
    params = fit.fe_params
    G = np.asarray(fit.cov_re, dtype=float)
    sigma2 = float(fit.scale)
    print(fit.summary().tables[1].to_string() if len(fit.summary().tables) > 1 else fit.summary())
    print(f"  σ_erro {np.sqrt(sigma2):.3f} pp · G {np.round(G, 5).tolist()} · ajuste em {train_seconds:.0f} s")

    # Efeito aleatório ao fim de cada volta, com as voltas limpas até ela
    x_laps = design(laps, temp_mean)
    resid = laps["pace"].to_numpy() * 100 - fixed_part(x_laps, params)
    b, V = online_blup(laps, resid, random_design(x_laps, args.slopes), G, sigma2)

    # Volta t + h: covariáveis conhecidas da volta prevista, temperatura da volta t
    target, anchor = laps.iloc[frame["row"]], laps.iloc[frame["anchor"]]
    x = design(target, temp_mean)
    x["track_temp_c"] = anchor["track_temp"].astype(float).to_numpy() - temp_mean
    a = frame["anchor"].to_numpy()
    z = random_design(x, args.slopes)
    mu = fixed_part(x, params) + np.einsum("ni,ni->n", z, b[a])
    sd = np.sqrt(np.einsum("ni,nij,nj->n", z, V[a], z) + sigma2)
    for col, q in QUANTILES.items():
        frame[col] = (mu + norm.ppf(q) * sd) / 100

    pred = finish_predictions(frame, run_id, "lmm")
    naive_dir = args.reference or latest_run("naive")
    reference = pd.read_parquet(naive_dir / "predictions.parquet") if naive_dir else None
    metrics = compute_metrics(pred, reference)
    if naive_dir:
        metrics["reference_run"] = naive_dir.name
    metrics["fit"] = {"fe_params": {k: float(v) for k, v in params.items()},
                      "cov_re": G.tolist(), "sigma2": sigma2, "loglik_reml": float(fit.llf),
                      "converged": bool(fit.converged), "units": "pontos percentuais de ritmo"}
    config = {
        "run_id": run_id, "model": "lmm", "seed": None,
        "hyperparameters": {"fixed": FIXED, "random": f"{re_formula} | piloto:corrida", "reml": True,
                            "track_temp_mean": temp_mean},
        "eras_strategy": "pooled", "features": FEATURES,
        "horizon": meta["horizon"], "quantiles": list(QUANTILES.values()), "data": meta["splits"],
        "n_params": int(len(params) + G.shape[0] * (G.shape[0] + 1) // 2 + 1),
        "train_seconds": round(train_seconds, 1), "best_epoch": None,
        "device": "cpu", "git_commit": git_commit(), "created_at": now_utc(), "dataset": args.dataset,
    }
    out = write_run(run_id, config, pred, metrics, root=args.out)
    print(f"{run_id}: {len(pred):,} previsões -> {out}")
    print_summary(metrics)


if __name__ == "__main__":
    main()
