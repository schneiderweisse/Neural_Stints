"""Backtest de corridas do teste: GRU contra o ingênuo, volta a volta.

Gera runs/<run da GRU>/backtest.html (figuras interativas) e backtest_drivers.csv (erro por piloto).
As métricas usam as mesmas linhas para os dois modelos: voltas limpas com previsão dos dois.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.backtest                      # corridas do teste, últimos runs
    .venv/Scripts/python -m modeling.backtest --races "2026 Austin"
"""
import argparse
import html
import json
import sys

import numpy as np
import pandas as pd

from modeling.figures import fmt_lap, mae_by_horizon, stint_forecasts
from modeling.gru import latest_run
from pipeline.sequences import RACE_LAPS

KEYS = ["race_label", "driver", "t", "h"]


def paired(gru: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    """Linhas avaliáveis pelos dois modelos, com o erro absoluto de cada um."""
    m = gru.merge(naive[KEYS + ["time_p50"]].rename(columns={"time_p50": "naive_p50"}), on=KEYS)
    m = m[m["clean"] & m["time_real"].notna() & m["time_p50"].notna() & m["naive_p50"].notna()].copy()
    m["ae_gru"] = (m["time_p50"] - m["time_real"]).abs()
    m["ae_naive"] = (m["naive_p50"] - m["time_real"]).abs()
    m["covered"] = (m["time_real"] >= m["time_p10"]) & (m["time_real"] <= m["time_p90"])
    return m


def race_summary(m: pd.DataFrame) -> dict:
    after = m[~m["same_stint"]]
    return {"n": int(len(m)), "mae_gru": float(m["ae_gru"].mean()), "mae_naive": float(m["ae_naive"].mean()),
            "skill": float(1 - m["ae_gru"].mean() / m["ae_naive"].mean()), "coverage": float(m["covered"].mean()),
            "after_pit_gru": float(after["ae_gru"].mean()), "after_pit_naive": float(after["ae_naive"].mean()),
            "by_h": {"gru": m.groupby("h")["ae_gru"].mean().to_dict(), "naive": m.groupby("h")["ae_naive"].mean().to_dict()}}


def driver_table(m: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    final = laps.sort_values("lap").groupby("driver").tail(1).set_index("driver")
    t = m.groupby("driver").agg(n=("ae_gru", "size"), mae_gru=("ae_gru", "mean"), mae_naive=("ae_naive", "mean"),
                                coverage=("covered", "mean"))
    t["skill"] = 1 - t["mae_gru"] / t["mae_naive"]
    t["final_position"] = final["position"].reindex(t.index)
    t["laps"] = final["lap"].reindex(t.index)
    t["team"] = final["team"].reindex(t.index)
    return t.sort_values(["final_position", "laps"], ascending=[True, False]).reset_index()


def tiles(race: str, s: dict) -> str:
    items = [("MAE GRU v0", f"{s['mae_gru']:.3f} s"), ("MAE ingênuo", f"{s['mae_naive']:.3f} s"),
             ("Melhora sobre o ingênuo", f"{s['skill']:+.0%}"), ("Cobertura p10–p90", f"{s['coverage']:.0%}"),
             ("Depois de uma parada", f"{s['after_pit_gru']:.2f} s × {s['after_pit_naive']:.2f} s")]
    cells = "".join(f'<div class="tile"><div class="tl">{k}</div><div class="tv">{v}</div></div>' for k, v in items)
    return f'<div class="tiles">{cells}</div>'


def table_html(t: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{'' if pd.isna(r.final_position) else int(r.final_position)}</td><td>{r.driver}</td>"
        f"<td>{html.escape(str(r.team))}</td><td>{int(r.laps)}</td><td>{r.mae_gru:.3f}</td><td>{r.mae_naive:.3f}</td>"
        f"<td>{r.skill:+.0%}</td><td>{r.coverage:.0%}</td><td>{r.n}</td></tr>" for r in t.itertuples())
    head = ("<tr><th>Pos.</th><th>Piloto</th><th>Equipe</th><th>Voltas</th><th>MAE GRU (s)</th><th>MAE ingênuo (s)</th>"
            "<th>Melhora</th><th>Cobertura</th><th>Previsões</th></tr>")
    return f'<table class="drivers"><thead>{head}</thead><tbody>{rows}</tbody></table>'


CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9}
body{background:var(--surface);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px} h3{font-size:15px;margin:20px 0 8px}
p{color:var(--ink2);line-height:1.55;margin:4px 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}
.tile{background:#f3f2ee;border-radius:8px;padding:10px 12px}.tl{color:var(--ink2);font-size:12px}.tv{font-size:20px;margin-top:2px}
table.drivers{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;margin-top:8px}
table.drivers th,table.drivers td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right}
table.drivers th:nth-child(2),table.drivers td:nth-child(2),table.drivers th:nth-child(3),table.drivers td:nth-child(3){text-align:left}
table.drivers th{color:var(--ink2);font-weight:500}
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", nargs="+", help="padrão: as corridas do conjunto de teste do run")
    ap.add_argument("--top", type=int, default=3, help="pilotos com gráfico volta a volta, por corrida")
    args = ap.parse_args()

    gru_dir, naive_dir = latest_run("gru"), latest_run("naive")
    gru = pd.read_parquet(gru_dir / "predictions.parquet")
    if not args.races:
        test = gru[gru["split"] == "test"].sort_values("race_id")
        args.races = list(dict.fromkeys(test["race_label"]))
    naive = pd.read_parquet(naive_dir / "predictions.parquet")
    race_laps = pd.read_parquet(RACE_LAPS)
    gru, naive = gru[gru["race_label"].isin(args.races)], naive[naive["race_label"].isin(args.races)]

    summaries, sections, tables = {}, [], []
    for race in args.races:
        laps = race_laps[race_laps["race_label"] == race]
        m = paired(gru[gru["race_label"] == race], naive[naive["race_label"] == race])
        s = race_summary(m)
        summaries[race] = s
        t = driver_table(m, laps)
        tables.append(t.assign(race=race))
        drivers = t["driver"].head(args.top).tolist()
        titles = [f"P{int(r.final_position)} {r.driver} · MAE {r.mae_gru:.2f} s (ingênuo {r.mae_naive:.2f} s)"
                  for r in t.head(args.top).itertuples()]
        anchors = list(range(10, int(laps["lap"].max()), 10))
        fig = stint_forecasts(laps, {"gru": gru[gru["race_label"] == race], "naive": naive[naive["race_label"] == race]},
                              drivers, anchors, titles)
        sections.append(f"<h2>{html.escape(race)}</h2>{tiles(race, s)}"
                        f"<p>Previsões feitas ao fim das voltas {', '.join(map(str, anchors))}, cada uma para as 10 "
                        f"voltas seguintes. Pontos: tempos reais das voltas limpas. Linhas pontilhadas: entrada nos "
                        f"boxes. Faixas cinza: voltas sob SC/VSC. As previsões não aparecem nas voltas de box e de "
                        f"SC, que não têm tempo comparável.</p>"
                        + fig.to_html(full_html=False, include_plotlyjs=False)
                        + "<h3>Erro por piloto</h3>" + table_html(t))

    hfig = mae_by_horizon({"gru": {"races": {r: {"by_h": s["by_h"]["gru"]} for r, s in summaries.items()}},
                           "naive": {"races": {r: {"by_h": s["by_h"]["naive"]} for r, s in summaries.items()}}},
                          args.races)
    page = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Backtest Neural Stints</title><style>{CSS}</style>"
            f"</head><body><main><h1>Backtest: GRU v0 × ingênuo</h1>"
            f"<p>Runs {gru_dir.name} e {naive_dir.name}. Corridas do conjunto de teste, que a rede não viu no treino "
            f"nem na escolha da época. Métricas nas voltas limpas previstas pelos dois modelos.</p>"
            f"<h2>Erro por horizonte</h2>"
            + hfig.to_html(full_html=False, include_plotlyjs=True)
            + "".join(sections) + "</main></body></html>")
    out = gru_dir / "backtest.html"
    out.write_text(page, encoding="utf-8")
    pd.concat(tables).to_csv(gru_dir / "backtest_drivers.csv", index=False)
    (gru_dir / "backtest_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1, default=float),
                                                   encoding="utf-8")
    for race, s in summaries.items():
        print(f"{race}: MAE GRU {s['mae_gru']:.3f} s · ingênuo {s['mae_naive']:.3f} s · melhora {s['skill']:+.0%} · "
              f"cobertura {s['coverage']:.0%} · depois de parada {s['after_pit_gru']:.2f} × {s['after_pit_naive']:.2f} s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
