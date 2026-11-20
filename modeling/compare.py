"""Comparação de modelos nas mesmas linhas: voltas limpas com valor real e previsão de todos os runs.

Cada run avalia as linhas em que tem previsão; o ingênuo, por exemplo, não prevê antes da primeira
volta limpa. Aqui todos são medidos no mesmo conjunto, e o skill de cada um é contra o primeiro run.

A diferença de MAE de cada modelo para o último run tem intervalo de 95% por bootstrap em blocos: os
blocos são os pares piloto:corrida, porque as voltas de um mesmo piloto numa corrida têm erros
correlacionados. Com 3 ou 4 corridas por divisão, reamostrar corridas inteiras daria pouca informação;
o intervalo por piloto:corrida não cobre a variação entre corridas e é, por isso, otimista.
Grava runs/compare_<data>/comparison.json e imprime as tabelas em Markdown (docs/resultados.md).

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.compare naive_20261106 lmm_20261120 lgbm_20261120 gru_20261106_ens5
"""
import argparse
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from modeling.results import RUNS, now_utc, sum5_error, summarize, write_atomic

KEYS = ["split", "race_id", "driver", "t", "h"]
N_BOOT = 2000
SPLITS = {"val": "validação", "test": "teste"}


def common_rows(runs: list[str]) -> dict[str, pd.DataFrame]:
    preds = {r: pd.read_parquet(RUNS / r / "predictions.parquet") for r in runs}
    ok = None
    for p in preds.values():
        k = p.loc[p["clean"] & p["time_real"].notna() & p["time_p50"].notna(), KEYS]
        ok = k if ok is None else ok.merge(k, on=KEYS)
    return {r: p.merge(ok, on=KEYS) for r, p in preds.items()}


def block_bootstrap(data: dict[str, pd.DataFrame], split: str, target: str, seed: int = 0) -> dict:
    """MAE(modelo) − MAE(target) com intervalo de 95%, reamostrando pares piloto:corrida."""
    rng = np.random.default_rng(seed)
    blocks = None
    sums = {}
    for run, p in data.items():
        g = p[p["split"] == split]
        ae = (g["time_p50"] - g["time_real"]).abs()
        b = ae.groupby([g["race_id"], g["driver"]]).agg(["sum", "count"])
        blocks = b.index if blocks is None else blocks
        sums[run] = b.loc[blocks]
    w = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)
    mae = {r: (w @ b["sum"].to_numpy()) / (w @ b["count"].to_numpy()) for r, b in sums.items()}
    point = {r: b["sum"].sum() / b["count"].sum() for r, b in sums.items()}
    out = {}
    for run in data:
        if run == target:
            continue
        diff = mae[run] - mae[target]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[run] = {"diff_s": float(point[run] - point[target]), "ci95_s": [float(lo), float(hi)],
                    "p_worse": float((diff > 0).mean()), "n_blocks": int(len(blocks))}
    return out


def fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return {"s": f"{v:.3f}", "s2": f"{v:.2f}", "pct": f"{v:.1%}", "signed": f"{v:+.3f}",
            "skill": f"{v:+.1%}", "pct0": f"{v:.0%}"}[kind].replace(".", ",")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="pastas em runs/; o primeiro é a referência do skill")
    ap.add_argument("--run-id", default=f"compare_{date.today():%Y%m%d}")
    args = ap.parse_args()
    data = common_rows(args.runs)
    ref = args.runs[0]
    out = {"computed_at": now_utc(), "runs": args.runs, "reference": ref, "splits": {}}
    for split in SPLITS:
        ref_ae = (data[ref].query("split == @split").eval("time_p50 - time_real")).abs()
        s = {}
        for run, p in data.items():
            g = p[p["split"] == split]
            ae = (g["time_p50"] - g["time_real"]).abs()
            s[run] = {"overall": summarize(g), "sum5": sum5_error(g),
                      "skill": float(1 - ae.mean() / ref_ae.mean()),
                      "by_h": {int(h): float(v) for h, v in ae.groupby(g["h"]).mean().items()},
                      "by_race": {k: float(v) for k, v in ae.groupby(g["race_label"], sort=False).mean().items()},
                      "by_same_stint": {str(k): float(v) for k, v in ae.groupby(g["same_stint"]).mean().items()}}
            if g["time_p10"].notna().all() and len(g):               # calibração dos quantis por horizonte
                inside = (g["time_real"] >= g["time_p10"]) & (g["time_real"] <= g["time_p90"])
                s[run]["coverage_by_h"] = {int(h): float(v) for h, v in inside.groupby(g["h"]).mean().items()}
        out["splits"][split] = s
        out.setdefault("bootstrap", {})[split] = block_bootstrap(data, split, args.runs[-1])
    dest = RUNS / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    write_atomic(dest / "comparison.json",
                 lambda p: p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"))

    for split, name in SPLITS.items():
        s = out["splits"][split]
        n = next(iter(s.values()))["overall"]["n"]
        print(f"\n**{name.capitalize()}** ({n:,} voltas em comum)\n".replace(",", "."))
        print("| Modelo | MAE (s) | RMSE (s) | Viés (s) | Erro em 5 voltas (s) | Cobertura 80% | Pinball (s) | Skill vs. "
              f"{ref.split('_')[0]} |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for run, m in s.items():
            o = m["overall"]
            print(f"| `{run}` | {fmt(o['mae_s'], 's')} | {fmt(o['rmse_s'], 's')} | {fmt(o['bias_s'], 'signed')} | "
                  f"{fmt(m['sum5']['mae_s'], 's2')} | {fmt(o.get('coverage_80'), 'pct')} | "
                  f"{fmt(o.get('pinball_s'), 's')} | {fmt(m['skill'], 'skill') if run != ref else '—'} |")
        calib = {run: m for run, m in s.items() if "coverage_80" in m["overall"]}
        if calib:
            print(f"\nCalibração dos quantis, {name} (ideal: 10% abaixo do p10, 80% dentro, 10% acima do p90):\n")
            print("| Modelo | Abaixo do p10 | Dentro de p10–p90 | Acima do p90 | Largura p10–p90 (s) | Pinball (s) |")
            print("|---|---:|---:|---:|---:|---:|")
            for run, m in calib.items():
                o = m["overall"]
                print(f"| `{run}` | {fmt(o['below_p10'], 'pct')} | {fmt(o['coverage_80'], 'pct')} | "
                      f"{fmt(1 - o['below_p90'], 'pct')} | {fmt(o['width_s'], 's2')} | {fmt(o['pinball_s'], 's')} |")
            print(f"\nCobertura p10–p90 por horizonte, {name}:\n")
            print("| Modelo | " + " | ".join(f"h{h}" for h in range(1, 11)) + " |")
            print("|---|" + "---:|" * 10)
            for run, m in calib.items():
                print(f"| `{run}` | " + " | ".join(fmt(m["coverage_by_h"].get(h), "pct0") for h in range(1, 11)) + " |")
        print(f"\nMAE por horizonte, {name} (s):\n")
        print("| Modelo | " + " | ".join(f"h{h}" for h in range(1, 11)) + " |")
        print("|---|" + "---:|" * 10)
        for run, m in s.items():
            print(f"| `{run}` | " + " | ".join(fmt(m["by_h"].get(h), "s2") for h in range(1, 11)) + " |")
        races = list(next(iter(s.values()))["by_race"])
        print(f"\nMAE por corrida, {name} (s):\n")
        print("| Modelo | " + " | ".join(races) + " |")
        print("|---|" + "---:|" * len(races))
        for run, m in s.items():
            print(f"| `{run}` | " + " | ".join(fmt(m["by_race"][r], "s") for r in races) + " |")
        boot = out["bootstrap"][split]
        target = args.runs[-1]
        print(f"\nDiferença de MAE para `{target}`, {name} (s; IC 95% por bootstrap em "
              f"{next(iter(boot.values()))['n_blocks']} pares piloto:corrida):\n")
        print("| Modelo | Diferença | IC 95% |")
        print("|---|---:|---:|")
        for run, b in boot.items():
            print(f"| `{run}` | {fmt(b['diff_s'], 'signed')} | "
                  f"{fmt(b['ci95_s'][0], 'signed')} a {fmt(b['ci95_s'][1], 'signed')} |")
    print(f"\n-> {dest / 'comparison.json'}")


if __name__ == "__main__":
    main()
