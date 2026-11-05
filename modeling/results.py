"""Gravação dos resultados de um modelo e cálculo das métricas (formato de docs/resultados.md)."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
QUANTILES = {"pace_p10": 0.1, "pace_p50": 0.5, "pace_p90": 0.9}
COLUMNS = ["run_id", "model", "split", "race_id", "race_label", "circuit", "year", "regulation", "driver",
           "team", "t", "h", "lap", "compound", "tyre_life", "stint", "same_stint", "clean", "ref_lap",
           "pace_p10", "pace_p50", "pace_p90", "pace_real", "time_p10", "time_p50", "time_p90", "time_real"]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finish_predictions(frame: pd.DataFrame, run_id: str, model: str) -> pd.DataFrame:
    """Completa a tabela de alvos com as previsões em segundos e põe as colunas na ordem padrão."""
    p = frame.copy()
    p["run_id"], p["model"] = run_id, model
    for q in QUANTILES:
        if q not in p:
            p[q] = np.nan
        p[q.replace("pace", "time")] = (1 + p[q]) * p["ref_lap"]
    p["time_real"] = (1 + p["pace_real"]) * p["ref_lap"]
    return p[COLUMNS]


def write_atomic(path: Path, write) -> None:
    """Grava num arquivo temporário e renomeia, para ninguém ler o arquivo pela metade."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_run(run_id: str, config: dict, predictions: pd.DataFrame, metrics: dict,
              history: pd.DataFrame | None = None) -> Path:
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda obj: lambda p: p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    write_atomic(out / "config.json", dump(config))
    write_atomic(out / "predictions.parquet", lambda p: predictions.to_parquet(p, index=False))
    write_atomic(out / "metrics.json", dump(metrics))
    if history is not None:
        write_atomic(out / "history.csv", lambda p: history.to_csv(p, index=False))
    return out


# ---------------------------------------------------------------- métricas

def point_metrics(g: pd.DataFrame) -> dict:
    err = g["time_p50"] - g["time_real"]
    return {"n": int(len(g)), "mae_s": float(err.abs().mean()), "rmse_s": float(np.sqrt((err ** 2).mean())),
            "bias_s": float(err.mean()), "median_ae_s": float(err.abs().median()),
            "mae_pace_pp": float((g["pace_p50"] - g["pace_real"]).abs().mean() * 100)}


def interval_metrics(g: pd.DataFrame) -> dict:
    y = g["time_real"]
    pinball = np.mean([np.maximum(q * (y - g[c.replace("pace", "time")]), (q - 1) * (y - g[c.replace("pace", "time")])).mean()
                       for c, q in QUANTILES.items()])
    return {"pinball_s": float(pinball),
            "coverage_80": float(((y >= g["time_p10"]) & (y <= g["time_p90"])).mean()),
            "below_p10": float((y < g["time_p10"]).mean()), "below_p90": float((y < g["time_p90"]).mean()),
            "width_s": float((g["time_p90"] - g["time_p10"]).mean())}


def summarize(g: pd.DataFrame) -> dict:
    out = point_metrics(g)
    if g["time_p10"].notna().all() and g["time_p90"].notna().all() and len(g):
        out.update(interval_metrics(g))
    return out


def sum5_error(g: pd.DataFrame) -> dict:
    """Erro no tempo acumulado das voltas t+1..t+5, nas janelas em que as cinco são avaliáveis."""
    s = g[g["h"] <= 5].groupby(["race_id", "driver", "t"]).agg(n=("h", "size"), pred=("time_p50", "sum"),
                                                                 real=("time_real", "sum"))
    s = s[s["n"] == 5]
    err = (s["pred"] - s["real"]).abs()
    return {"n_windows": int(len(s)), "mae_s": float(err.mean()), "median_ae_s": float(err.median())}


def skill(g: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """1 − MAE do modelo / MAE da referência, nas mesmas linhas, no total e por horizonte."""
    keys = ["race_id", "driver", "t", "h"]
    m = g.merge(ref[keys + ["time_p50"]].rename(columns={"time_p50": "ref_p50"}), on=keys).dropna(subset=["ref_p50"])
    e, r = (m["time_p50"] - m["time_real"]).abs(), (m["ref_p50"] - m["time_real"]).abs()
    by_h = {int(h): float(1 - e[m["h"] == h].mean() / r[m["h"] == h].mean()) for h in sorted(m["h"].unique())}
    return {"n": int(len(m)), "overall": float(1 - e.mean() / r.mean()), "by_h": by_h}


def compute_metrics(pred: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    """Métricas por divisão, só nas voltas limpas com valor real e previsão."""
    ok = pred[pred["clean"] & pred["time_real"].notna() & pred["time_p50"].notna()]
    out = {"computed_at": now_utc(), "evaluated_rows": int(len(ok)), "total_rows": int(len(pred)), "splits": {}}
    for split, g in ok.groupby("split", sort=False):
        s = {"overall": summarize(g),
             "by_h": {int(h): summarize(gh) for h, gh in g.groupby("h")},
             "sum5": sum5_error(g),
             "by_same_stint": {str(k): summarize(gk) for k, gk in g.groupby("same_stint")},
             "by_compound": {str(k): summarize(gk) for k, gk in g.groupby("compound")},
             "by_race": {str(k): summarize(gk) for k, gk in g.groupby("race_label", sort=False)}}
        if reference is not None:
            s["skill_vs_reference"] = skill(g, reference[reference["split"] == split])
        out["splits"][split] = s
    return out


def print_summary(metrics: dict) -> None:
    for split, s in metrics["splits"].items():
        o = s["overall"]
        line = f"  {split:5}: MAE {o['mae_s']:.3f} s · RMSE {o['rmse_s']:.3f} s · viés {o['bias_s']:+.3f} s · n {o['n']:,}"
        if "coverage_80" in o:
            line += f" · cobertura 80% {o['coverage_80']:.1%}"
        print(line)
        print("         MAE por h (s): " + " ".join(f"{h}:{v['mae_s']:.2f}" for h, v in s["by_h"].items()))
        print(f"         erro em 5 voltas: {s['sum5']['mae_s']:.2f} s ({s['sum5']['n_windows']:,} janelas)")
        if "skill_vs_reference" in s:
            print(f"         skill vs. referência: {s['skill_vs_reference']['overall']:+.1%}")
