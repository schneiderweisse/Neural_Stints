"""Aba GRU v0: resultados do ajuste da rede contra o baseline ingênuo."""
import pandas as pd
import streamlit as st

from dashboard import data
from modeling.figures import coverage_by_horizon, grouped_bars, mae_by_horizon, stint_forecasts, training_curve

SPLIT_LABEL = {"val": "validação", "test": "teste"}


def tile(col, label: str, value: str, note: str = "") -> None:
    col.markdown(f"<div class='tile'><div class='tl'>{label}</div><div class='tv'>{value}</div>"
                 f"<div class='tn'>{note}</div></div>", unsafe_allow_html=True)


def race_table(g: dict, n: dict) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        for race, m in g["splits"][split]["by_race"].items():
            ref = n["splits"][split]["by_race"].get(race, {})
            rows.append({"divisão": SPLIT_LABEL[split], "corrida": race, "MAE GRU (s)": m["mae_s"],
                         "MAE ingênuo (s)": ref.get("mae_s"), "melhora": 1 - m["mae_s"] / ref["mae_s"] if ref else None,
                         "cobertura p10–p90": m.get("coverage_80"), "voltas avaliadas": m["n"]})
    return pd.DataFrame(rows)


def render() -> None:
    grus, naives = data.runs("gru"), data.runs("naive")
    if not grus or not naives:
        st.info("Ainda não há runs em runs/. Rode `python -m modeling.naive` e `python -m modeling.gru`.")
        return
    c1, c2 = st.columns(2)
    gru_id = c1.selectbox("Run da GRU", grus, key="gru_run")
    naive_id = c2.selectbox("Run do ingênuo (referência)", naives, key="naive_run")
    G, N = data.run_files(gru_id), data.run_files(naive_id)
    g, n, cfg = G["metrics"], N["metrics"], G["config"]
    gt, nt = g["splits"]["test"], n["splits"]["test"]

    st.subheader("Teste (Madring, Baku, Singapura, Austin e México 2026)")
    cols = st.columns(5)
    tile(cols[0], "MAE GRU v0", f"{gt['overall']['mae_s']:.3f} s", f"ingênuo {nt['overall']['mae_s']:.3f} s")
    tile(cols[1], "Melhora sobre o ingênuo", f"{gt['skill_vs_reference']['overall']:+.0%}",
         f"de {gt['skill_vs_reference']['by_h']['1']:+.0%} (h=1) a {gt['skill_vs_reference']['by_h']['10']:+.0%} (h=10)")
    tile(cols[2], "Cobertura p10–p90", f"{gt['overall']['coverage_80']:.0%}", "alvo: 80%")
    tile(cols[3], "Depois de uma parada", f"{gt['by_same_stint']['False']['mae_s']:.2f} s",
         f"ingênuo {nt['by_same_stint']['False']['mae_s']:.2f} s")
    tile(cols[4], "Modelo", f"{cfg['n_params']:,} parâmetros".replace(",", "."),
         f"melhor época {cfg['best_epoch']} · {cfg['train_seconds'] / 60:.1f} min de treino")

    st.subheader("Erro por horizonte")
    by_h = {m: {"races": {SPLIT_LABEL[s]: {"by_h": {int(h): v["mae_s"] for h, v in r["splits"][s]["by_h"].items()}}
                          for s in ("val", "test")}} for m, r in (("gru", g), ("naive", n))}
    st.plotly_chart(mae_by_horizon(by_h, ["validação", "teste"], theme="dark"), width="stretch",
                    config={"displaylogo": False})

    left, right = st.columns(2)
    with left:
        st.markdown("**Calibração do intervalo p10–p90 por horizonte**")
        cov = {s: {int(h): v["coverage_80"] for h, v in g["splits"][s]["by_h"].items()} for s in ("val", "test")}
        st.plotly_chart(coverage_by_horizon(cov), width="stretch", config={"displaylogo": False})
        o = gt["overall"]
        st.caption(f"No teste, {o['below_p10']:.0%} das voltas ficaram abaixo do p10 (ideal 10%) e {o['below_p90']:.0%} "
                   f"abaixo do p90 (ideal 90%). Largura média do intervalo: {o['width_s']:.2f} s.")
    with right:
        st.markdown("**Curva de treino**")
        if "history" in G:
            st.plotly_chart(training_curve(G["history"]), width="stretch", config={"displaylogo": False})
            st.caption("Perda pinball média nos três quantis, com o alvo padronizado. O treino para quando a "
                       "validação não melhora por 8 épocas e volta aos pesos da melhor época.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Erro por composto (teste)**")
        comps = [c for c in ("SOFT", "MEDIUM", "HARD") if c in gt["by_compound"]]
        st.plotly_chart(grouped_bars(comps, {"gru": [gt["by_compound"][c]["mae_s"] for c in comps],
                                             "naive": [nt["by_compound"][c]["mae_s"] for c in comps]}, "MAE (s)"),
                        width="stretch", config={"displaylogo": False})
    with right:
        st.markdown("**Mesmo stint × depois de uma parada (teste)**")
        cats, keys = ["mesmo stint (tarefa A)", "depois da parada (tarefa B)"], ["True", "False"]
        st.plotly_chart(grouped_bars(cats, {"gru": [gt["by_same_stint"][k]["mae_s"] for k in keys],
                                            "naive": [nt["by_same_stint"][k]["mae_s"] for k in keys]}, "MAE (s)"),
                        width="stretch", config={"displaylogo": False})

    st.subheader("Por corrida")
    st.dataframe(race_table(g, n), hide_index=True, width="stretch",
                 column_config={"MAE GRU (s)": st.column_config.NumberColumn(format="%.3f"),
                                "MAE ingênuo (s)": st.column_config.NumberColumn(format="%.3f"),
                                "melhora": st.column_config.NumberColumn(format="percent"),
                                "cobertura p10–p90": st.column_config.NumberColumn(format="percent")})

    st.subheader("Exemplos: previsto × real, volta a volta")
    gp, npred = data.run_predictions(gru_id), data.run_predictions(naive_id)
    races = list(dict.fromkeys(gp.sort_values(["split", "race_id"])["race_label"]))
    c1, c2 = st.columns(2)
    race = c1.selectbox("Corrida", races, index=races.index("2026 Austin") if "2026 Austin" in races else 0, key="ex_race")
    laps = data.race_laps(race)
    final = laps.sort_values("lap").groupby("driver").tail(1).sort_values(["position", "lap"], ascending=[True, False])
    drivers = [d for d in final["driver"] if d in set(gp.loc[gp["race_label"] == race, "driver"])]
    drv = c2.selectbox("Piloto (pela posição final)", drivers, key="ex_driver")
    anchors = list(range(10, int(laps["lap"].max()), 10))
    preds = {"gru": gp[gp["race_label"] == race], "naive": npred[npred["race_label"] == race]}
    st.plotly_chart(stint_forecasts(laps, preds, [drv], anchors, [f"{drv} · {race}"], theme="dark"),
                    width="stretch", config={"displaylogo": False})
    st.caption(f"Previsões feitas ao fim das voltas {', '.join(map(str, anchors))}, cada uma para as 10 voltas seguintes. "
               "Pontos: tempos reais das voltas limpas. Linhas pontilhadas: entrada nos boxes. Faixas cinza: SC/VSC.")
