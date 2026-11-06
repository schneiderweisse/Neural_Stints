"""Aba Tarefas A e B: o que a GRU previu e o que aconteceu, para até dois pilotos de uma corrida.

- Tarefa A: a partir de um ponto de partida (começo da corrida ou uma troca de pneus), o ritmo nas
  k voltas seguintes. "Parou na volta 30, k = 5" mostra a previsão feita ao fim da volta 30 para as
  voltas 31 a 35.
- Tarefa B: o stint inteiro de um jogo de pneus. A rede prevê 10 voltas por vez, então o stint é
  coberto por previsões feitas no começo do stint e a cada 10 voltas dentro dele.

Só corridas da validação e do teste, que a rede não viu no treino.
"""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data
from modeling.figures import driver_forecasts, fmt_lap

HIDDEN_RACES = {"2026 Madring"}         # fora desta aba, a pedido (as métricas continuam com ela)
MIN_CONTEXT = 2                          # a primeira previsão é feita ao fim da volta 2


def stints(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    d = laps[laps["driver"] == driver]
    s = d.groupby("stint").agg(start=("lap", "min"), end=("lap", "max"), compound=("compound", "first"),
                               life0=("tyre_life", "first"), fresh=("fresh_tyre", "first")).reset_index()
    return s[s["compound"].isin(["SOFT", "MEDIUM", "HARD"])]


def window(pred: pd.DataFrame, driver: str, t: int, max_lap: int | None = None) -> pd.DataFrame:
    w = pred[(pred["driver"] == driver) & (pred["t"] == t)]
    return w[w["lap"] <= max_lap] if max_lap is not None else w


def hide_unclean(w: pd.DataFrame) -> pd.DataFrame:
    """Voltas de box e de SC não têm tempo comparável: real e previsão somem nelas."""
    w = w.copy()
    w.loc[~w["clean"], ["time_real"]] = np.nan
    w.loc[~w["clean"], ["time_p10", "time_p50", "time_p90"]] = np.nan
    return w


def series(label: str, w: pd.DataFrame, x: pd.Series) -> dict:
    hover = [f"{label} · volta {int(lap)} · {c} ({int(life)} voltas)" for lap, c, life in
             zip(w["lap"], w["compound"], w["tyre_life"].fillna(0))]
    return {"label": label, "x": x.to_numpy(), "real": w["time_real"].to_numpy(), "p10": w["time_p10"].to_numpy(),
            "p50": w["time_p50"].to_numpy(), "p90": w["time_p90"].to_numpy(), "hover": hover}


def summary(label: str, w: pd.DataFrame, x: pd.Series) -> dict:
    ok = w["time_real"].notna() & w["time_p50"].notna()
    err = (w.loc[ok, "time_p50"] - w.loc[ok, "time_real"])
    out = {"piloto": label, "voltas avaliadas": int(ok.sum()),
           "MAE (s)": err.abs().mean() if ok.any() else np.nan, "viés (s)": err.mean() if ok.any() else np.nan,
           "cobertura p10–p90": ((w.loc[ok, "time_real"] >= w.loc[ok, "time_p10"]) &
                                 (w.loc[ok, "time_real"] <= w.loc[ok, "time_p90"])).mean() if ok.any() else np.nan}
    if ok.sum() >= 4:                    # degradação: inclinação do tempo de volta ao longo do stint
        out["degradação real (s/volta)"] = np.polyfit(x[ok], w.loc[ok, "time_real"], 1)[0]
        out["degradação prevista (s/volta)"] = np.polyfit(x[ok], w.loc[ok, "time_p50"], 1)[0]
    return out


def task_a(pred, laps, drivers) -> None:
    st.markdown("**Tarefa A: o ritmo nas próximas k voltas**")
    k = st.slider("k (voltas à frente)", 1, 10, 5, key="task_a_k")
    cols = st.columns(len(drivers))
    sers, sums = [], []
    for col, drv in zip(cols, drivers):
        options = [(f"Começo da corrida (previsão ao fim da volta {MIN_CONTEXT})", MIN_CONTEXT)]
        for r in stints(laps, drv).itertuples():
            if r.stint > 1:
                options.append((f"Parada na volta {r.start - 1} → {r.compound.lower()} (stint {int(r.stint)})", r.start - 1))
        label, t = col.selectbox(f"Ponto de partida de {drv}", options, format_func=lambda o: o[0], key=f"a_{drv}")
        w = hide_unclean(window(pred, drv, t)).sort_values("h")
        w = w[w["h"] <= k]
        if w.empty:
            col.caption("Sem previsão para esse ponto de partida.")
            continue
        sers.append(series(drv, w, w["h"]))
        sums.append(summary(drv, w, w["h"]))
        rows = pd.DataFrame({"volta": w["lap"].astype(int), "pneu": w["compound"].str.lower() + " " + w["tyre_life"].fillna(0).astype(int).astype(str),
                             "previsto": w["time_p50"].map(fmt_lap), "faixa p10–p90": w["time_p10"].map(fmt_lap) + " – " + w["time_p90"].map(fmt_lap),
                             "real": w["time_real"].map(fmt_lap)})
        col.dataframe(rows, hide_index=True, width="stretch")
    if sers:
        st.plotly_chart(driver_forecasts(sers, "voltas depois do ponto de partida"), width="stretch",
                        config={"displaylogo": False})
        st.dataframe(pd.DataFrame(sums), hide_index=True, width="stretch", column_config=fmt_cols())
        st.caption("Voltas de saída dos boxes e sob SC/VSC não aparecem: o tempo delas não é comparável e elas ficam "
                   "fora da avaliação. Depois de uma parada, a primeira volta (a de saída) costuma estar oculta.")


def task_b(pred, laps, drivers) -> None:
    st.markdown("**Tarefa B: o stint inteiro de um jogo de pneus**")
    cols = st.columns(len(drivers))
    sers, sums = [], []
    for col, drv in zip(cols, drivers):
        st_df = stints(laps, drv)
        if st_df.empty:
            col.caption("Sem stints em pneu de pista seca.")
            continue
        opts = list(st_df.itertuples())
        default = 1 if len(opts) > 1 else 0
        r = col.selectbox(f"Stint de {drv}", opts, index=default, key=f"b_{drv}",
                          format_func=lambda r: (f"Stint {int(r.stint)}: {r.compound.lower()}"
                                                 f"{'' if r.fresh else ' usado'}, voltas {int(r.start)}–{int(r.end)} "
                                                 f"({int(r.end - r.start + 1)} voltas)"))
        first = max(int(r.start) - 1, MIN_CONTEXT)
        anchors = list(range(first, int(r.end), 10))
        w = pd.concat([window(pred, drv, t, int(r.end)).assign(anchor=t) for t in anchors])
        w = w[w["lap"] >= r.start]
        if w.empty:
            col.caption("Sem previsão para esse stint.")
            continue
        w = hide_unclean(w).sort_values("lap")
        x = w["lap"] - int(r.start) + 1
        sers.append(series(f"{drv} (stint {int(r.stint)})", w, x))
        sums.append(summary(f"{drv} (stint {int(r.stint)})", w, x))
        col.caption(f"Previsões feitas ao fim das voltas {', '.join(map(str, anchors))}, cada uma para até 10 voltas.")
    if sers:
        st.plotly_chart(driver_forecasts(sers, "volta no stint"), width="stretch", config={"displaylogo": False})
        st.dataframe(pd.DataFrame(sums), hide_index=True, width="stretch", column_config=fmt_cols())
        st.caption("Degradação: inclinação de uma reta ajustada ao tempo de volta ao longo do stint (real × mediana "
                   "prevista), nas voltas limpas. Como a rede prevê 10 voltas por vez, um stint longo é coberto por "
                   "previsões feitas a cada 10 voltas; prever o stint inteiro de uma vez é um objetivo da v1.")


def fmt_cols() -> dict:
    num = lambda f: st.column_config.NumberColumn(format=f)
    return {"MAE (s)": num("%.3f"), "viés (s)": num("%+.3f"), "cobertura p10–p90": num("percent"),
            "degradação real (s/volta)": num("%+.3f"), "degradação prevista (s/volta)": num("%+.3f")}


def render() -> None:
    grus = data.runs("gru")
    if not grus:
        st.info("Ainda não há runs da GRU em runs/.")
        return
    gp = data.run_predictions(grus[0])
    order = gp.sort_values("race_id")["race_label"].drop_duplicates()
    races = [r for r in order if r not in HIDDEN_RACES]
    c1, c2 = st.columns([1, 2])
    race = c1.selectbox("Corrida (validação e teste)", races, index=len(races) - 1, key="tasks_race")
    laps = data.race_laps(race)
    final = laps.sort_values("lap").groupby("driver").tail(1).sort_values(["position", "lap"], ascending=[True, False])
    pred = gp[gp["race_label"] == race]
    options = [d for d in final["driver"] if d in set(pred["driver"])]
    drivers = c2.multiselect("Pilotos (até dois, pela posição final)", options, default=options[:2], max_selections=2,
                             key="tasks_drivers")
    st.caption(f"Modelo: {grus[0]}. Corrida fora do treino, então as previsões são as mesmas do backtest.")
    if not drivers:
        st.info("Escolha um ou dois pilotos.")
        return
    a, b = st.tabs(["Tarefa A · próximas k voltas", "Tarefa B · stint inteiro"])
    with a:
        task_a(pred, laps, drivers)
    with b:
        task_b(pred, laps, drivers)
