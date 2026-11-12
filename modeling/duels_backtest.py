"""Backtest dos duelos: a GRU acerta quando e em que volta um carro chega a menos de 1 s do da frente?

Situações avaliadas: ao fim de cada volta t, cada carro que está de 1 a 5 s atrás do carro logo à
frente (os dois com previsão da GRU em t e em pneu de pista seca). O que aconteceu de fato vem dos tempos
de volta reais: gap_k = gap_0 + soma dos tempos de B − tempos de A. A situação é encerrada (censurada)
na primeira volta com parada, safety car/VSC, pneu de chuva ou sem tempo de algum dos dois, porque a
projeção supõe pista livre e nenhuma parada. Situações censuradas antes de um desfecho ficam de fora.

Medidas, com o ingênuo (ritmo médio das últimas 3 voltas limpas de cada um) como comparação:
- detecção: a GRU prevê disputa nas próximas 10 voltas (probabilidade >= 50%)? precisão e recall;
- volta: entre as disputas previstas que aconteceram, o erro da volta provável e a parte dentro de ±1 volta;
- janela: a parte das disputas reais que começou dentro da janela de 80% (10% a 90%);
- calibração: escore de Brier da probabilidade de disputa nas 10 voltas e a frequência real por faixa.

O fator de largura `SIGMA_SCALE` é escolhido na validação para a janela cobrir ~80%; o teste e a corrida
de São Paulo (previsões feitas ao vivo, só a parte em pneu liso) medem o resultado.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.duels_backtest
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import duels
from modeling.results import RUNS, write_atomic

ROOT = Path(__file__).resolve().parents[1]
GRU_RUN, NAIVE_RUN = "gru_20261106_ens5", "naive_20261106"
LIVE_RACE = ROOT / "data" / "live" / "2026_interlagos_race"
MAX_GAP = 5.0
H = 10
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)


def forecast_arrays(pred: pd.DataFrame) -> tuple[dict, dict]:
    """{(corrida, piloto, t): matriz (H, 3) p10/p50/p90} e {(corrida, piloto, t): tempo do ingênuo}."""
    arrays, naive = {}, {}
    for (race, drv, t), g in pred.groupby(["race_label", "driver", "t"]):
        m = np.full((H, 3), np.nan)
        hs = g["h"].to_numpy(int) - 1
        m[hs] = g[["time_p10", "time_p50", "time_p90"]].to_numpy(float)
        arrays[(race, drv, int(t))] = m
        if "naive" in g:
            naive[(race, drv, int(t))] = float(g["naive"].iloc[0])
    return arrays, naive


def situations(laps: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    """Pares (B atrás de A) a 1–5 s, com o desfecho real nas H voltas seguintes."""
    rows = []
    for race, r in laps.groupby("race_label"):
        r = r.set_index(["driver", "lap"]).sort_index()
        stop = (r["pit_in"].astype(bool) | r["pit_out"].astype(bool) | r["neutralised"].astype(bool)
                | r["wet"].astype(bool) | r["lap_time"].isna())
        lap_time, position = r["lap_time"], r["position"]
        for (drv, t), row in r.iterrows():
            if not (1.0 < row["gap_ahead"] <= MAX_GAP) or pd.isna(row["position"]) or row["position"] < 2:
                continue
            same_lap = r.xs(t, level="lap")
            ahead = same_lap.index[same_lap["position"] == row["position"] - 1]
            if len(ahead) != 1:
                continue
            a = ahead[0]
            if (race, drv, t) not in arrays or (race, a, t) not in arrays or stop.get((drv, t), True) or stop.get((a, t), True):
                continue
            gap, actual, censored = float(row["gap_ahead"]), None, None
            for k in range(1, H + 1):
                kb, ka = (drv, t + k), (a, t + k)
                if kb not in stop.index or ka not in stop.index or stop[kb] or stop[ka]:
                    censored = k
                    break
                gap += lap_time[kb] - lap_time[ka]
                if gap <= duels.BATTLE_GAP:
                    actual = k
                    break
            if actual is None and censored is not None:
                continue                                   # desfecho desconhecido
            rows.append(dict(race=race, t=int(t), behind=drv, ahead=a, gap0=float(row["gap_ahead"]), actual=actual))
    return pd.DataFrame(rows)


def evaluate(sit: pd.DataFrame, arrays: dict, naive: dict, scale: float) -> dict:
    recs = []
    for s in sit.itertuples():
        B, A = arrays[(s.race, s.behind, s.t)], arrays[(s.race, s.ahead, s.t)]
        proj = duels.project(s.gap0, B, A, scale)
        if len(proj["caught"]) < H:
            continue                                       # perto do fim da corrida: horizonte incompleto
        w = duels.catch_window(proj["caught"])
        nb, na = naive.get((s.race, s.behind, s.t)), naive.get((s.race, s.ahead, s.t))
        nlap = None
        if nb is not None and na is not None and not (np.isnan(nb) or np.isnan(na)):
            mu = s.gap0 + np.arange(1, H + 1) * (nb - na)
            hit = np.flatnonzero(mu <= duels.BATTLE_GAP)
            nlap = int(hit[0]) + 1 if len(hit) else None
        recs.append(dict(actual=s.actual, p=w["p_horizon"], likely=w["likely"], early=w["early"],
                         late=w["late"] if not w["late_open"] else H + 99, naive=nlap))
    e = pd.DataFrame(recs)
    real = e["actual"].notna()
    pred = e["likely"].notna()
    npred = e["naive"].notna()
    tp, ntp = real & pred, real & npred
    covered = real & e["early"].notna() & (e["actual"] >= e["early"]) & (e["actual"] <= e["late"])
    bins = pd.cut(e["p"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    rel = e.groupby(bins, observed=True).agg(n=("p", "size"), previsto=("p", "mean"), real=("actual", lambda x: x.notna().mean()))
    base = real.mean()
    return {
        "situacoes": int(len(e)), "disputas_reais": int(real.sum()), "taxa_real": float(base),
        "precisao_gru": float(tp.sum() / max(pred.sum(), 1)), "recall_gru": float(tp.sum() / max(real.sum(), 1)),
        "precisao_ingenuo": float(ntp.sum() / max(npred.sum(), 1)), "recall_ingenuo": float(ntp.sum() / max(real.sum(), 1)),
        "erro_volta_gru": float((e.loc[tp, "likely"] - e.loc[tp, "actual"]).abs().mean()),
        "dentro_1_volta_gru": float(((e.loc[tp, "likely"] - e.loc[tp, "actual"]).abs() <= 1).mean()),
        "erro_volta_ingenuo": float((e.loc[ntp, "naive"] - e.loc[ntp, "actual"]).abs().mean()),
        "dentro_1_volta_ingenuo": float(((e.loc[ntp, "naive"] - e.loc[ntp, "actual"]).abs() <= 1).mean()),
        "cobertura_janela_80": float(covered.sum() / max(real.sum(), 1)),
        "brier_gru": float(((e["p"] - real.astype(float)) ** 2).mean()),
        "brier_taxa_constante": float(((base - real.astype(float)) ** 2).mean()),
        "calibracao": {str(k): {c: round(float(v), 3) for c, v in row.items()} for k, row in rel.iterrows()},
    }


def run_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Previsões da validação e do teste (com o ingênuo) e as voltas reais dessas corridas."""
    gru = pd.read_parquet(RUNS / GRU_RUN / "predictions.parquet",
                          columns=["split", "race_label", "driver", "t", "h", "time_p10", "time_p50", "time_p90"])
    nai = pd.read_parquet(RUNS / NAIVE_RUN / "predictions.parquet", columns=["race_label", "driver", "t", "h", "time_p50"])
    nai = nai[nai["h"] == 1].rename(columns={"time_p50": "naive"}).drop(columns="h")
    pred = gru.merge(nai, on=["race_label", "driver", "t"], how="left")
    laps = pd.read_parquet(ROOT / "data" / "processed" / "race_laps.parquet",
                           columns=["race_label", "driver", "lap", "lap_time", "position", "gap_ahead", "pit_in", "pit_out",
                                    "neutralised", "wet_tyre"])
    laps = laps[laps["race_label"].isin(pred["race_label"].unique())].rename(columns={"wet_tyre": "wet"})
    return pred, laps


def live_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Previsões feitas ao vivo em São Paulo e as voltas da corrida, com as mesmas flags do treino."""
    from live.predict import LivePredictor
    raw = pd.read_parquet(LIVE_RACE / "laps.parquet")
    f = LivePredictor(LIVE_RACE).features(raw)
    f["race_label"] = "2026 Interlagos (ao vivo)"
    clean = f.loc[f["clean_live"], ["driver", "lap", "lap_time"]].sort_values(["driver", "lap"])
    clean["naive"] = clean.groupby("driver")["lap_time"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    pred = pd.read_parquet(LIVE_RACE / "predictions.parquet", columns=["driver", "t", "h", "time_p10", "time_p50", "time_p90"])
    pred["race_label"] = "2026 Interlagos (ao vivo)"
    naive = f[["driver", "lap"]].merge(clean[["driver", "lap", "naive"]], on=["driver", "lap"], how="left")
    naive["naive"] = naive.groupby("driver")["naive"].ffill()
    pred = pred.merge(naive.rename(columns={"lap": "t"}), on=["driver", "t"], how="left")
    laps = f[["race_label", "driver", "lap", "lap_time", "position", "gap_ahead", "pit_in", "pit_out", "neutralised", "wet_tyre"]]
    return pred, laps.rename(columns={"wet_tyre": "wet"})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred, laps = run_data()
    out = {"gru_run": GRU_RUN, "naive_run": NAIVE_RUN, "battle_gap_s": duels.BATTLE_GAP, "max_gap_s": MAX_GAP, "horizon": H}
    data = {}
    for split in ("val", "test"):
        p = pred[pred["split"] == split]
        arrays, naive = forecast_arrays(p)
        data[split] = (situations(laps[laps["race_label"].isin(p["race_label"].unique())], arrays), arrays, naive)
    lp, ll = live_data()
    arrays, naive = forecast_arrays(lp)
    data["sao_paulo_ao_vivo"] = (situations(ll, arrays), arrays, naive)

    grid = {s: evaluate(*data["val"], scale=s)["cobertura_janela_80"] for s in SCALES}
    scale = min(grid, key=lambda s: abs(grid[s] - 0.80))
    out["escolha_da_largura"] = {"cobertura_janela_80_na_validacao": {str(k): round(v, 3) for k, v in grid.items()}, "sigma_scale": scale}
    print("largura da janela na validação (cobertura de 80%):", {k: round(v, 3) for k, v in grid.items()}, "-> escolhida", scale)
    for name, args in data.items():
        out[name] = evaluate(*args, scale=scale)
        m = out[name]
        print(f"\n{name}: {m['situacoes']} situações, {m['disputas_reais']} disputas ({m['taxa_real']:.0%})")
        print(f"  detecção (10 voltas): GRU precisão {m['precisao_gru']:.0%} recall {m['recall_gru']:.0%} · "
              f"ingênuo {m['precisao_ingenuo']:.0%} / {m['recall_ingenuo']:.0%}")
        print(f"  volta: erro médio GRU {m['erro_volta_gru']:.2f} ({m['dentro_1_volta_gru']:.0%} a ±1) · "
              f"ingênuo {m['erro_volta_ingenuo']:.2f} ({m['dentro_1_volta_ingenuo']:.0%} a ±1)")
        print(f"  janela de 80%: {m['cobertura_janela_80']:.0%} das disputas · Brier {m['brier_gru']:.3f} "
              f"(taxa constante {m['brier_taxa_constante']:.3f})")
        print("  calibração:", m["calibracao"])
    dest = RUNS / f"duels_{date.today():%Y%m%d}"
    dest.mkdir(parents=True, exist_ok=True)
    write_atomic(dest / "metrics.json", lambda p: p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"))
    print(f"\n-> {dest.relative_to(ROOT)}/metrics.json")


if __name__ == "__main__":
    main()
