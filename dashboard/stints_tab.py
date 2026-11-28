"""Aba Stints: um stint de um piloto numa corrida, o ritmo real e o que a rede projetou para ele.

Exemplo: Verstappen em Las Vegas, stint 2 de pneu médio (voltas 20 a 50). O gráfico mostra os tempos de
volta reais e a projeção da rede, com a faixa de 80% e a tendência (degradação) de cada um. Há três
jeitos de ver a projeção:

- do começo do stint: a previsão feita no começo do stint e, como a rede prevê 10 voltas por vez, as
  feitas a cada 10 voltas dentro dele;
- volta a volta: para cada volta, a previsão feita k voltas antes (k = 1 é a próxima volta);
- a partir de uma volta escolhida: as 10 voltas que a rede projetou naquele momento.

Corridas: as ao vivo (data/live, com as previsões feitas durante a corrida e as refeitas onde se perderam)
e as de validação e teste da LSTM, que a rede não viu no treino. Nessas, as variáveis conhecidas do
futuro são as reais (inclusive as paradas); ao vivo, a previsão supõe o mesmo pneu e nenhuma parada.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.tasks_tab import HIDDEN_RACES
from modeling.figures import stint_projection

WET = {"INTERMEDIATE", "WET"}
COMPOUND_PT = {"SOFT": "macio", "MEDIUM": "médio", "HARD": "duro"}
SPLIT_PT = {"val": "validação", "test": "teste"}


@st.cache_data(show_spinner=False)
def race_options() -> list[dict]:
    """Corridas ao vivo com previsões e as de validação e teste do run da LSTM mais recente."""
    out = []
    for p in sorted(data.LIVE.glob("*_race/predictions.parquet"), reverse=True):
        info = json.loads((p.parent / "session.json").read_text(encoding="utf-8")) if (p.parent / "session.json").exists() else {}
        year = str(info.get("StartDate", p.parent.name))[:4]
        circuit = (info.get("Meeting") or {}).get("Circuit", {}).get("ShortName") or p.parent.name
        run = str(pd.read_parquet(p, columns=["gru_run"])["gru_run"].iloc[-1])
        out.append({"kind": "live", "key": p.parent.name, "run": run,
                    "label": f"{year} {circuit} · ao vivo, {'LSTM' if run.startswith('lstm') else 'GRU'}"})
    lstm = data.runs("lstm")
    if lstm:
        pred = data.run_predictions(lstm[0])
        races = pred[["split", "race_id", "race_label"]].drop_duplicates().sort_values(["split", "race_id"], ascending=[True, False])
        for r in races.itertuples():
            if r.race_label not in HIDDEN_RACES:
                out.append({"kind": "backtest", "key": r.race_label, "run": lstm[0],
                            "label": f"{r.race_label} · {SPLIT_PT.get(r.split, r.split)}, LSTM"})
    return out


def live_laps(session: str, preds: pd.DataFrame) -> pd.DataFrame:
    """Voltas gravadas ao vivo no formato das do pipeline. A volta é limpa se a previsão diz que é (o critério do
    treino); onde não há previsão, sem box, sem SC/VSC, em pneu liso e até 107% da mediana do piloto."""
    laps = pd.read_parquet(data.LIVE / session / "laps.parquet")
    laps = laps.sort_values(["driver", "lap"]).drop_duplicates(["driver", "lap"], keep="last")
    status = laps["track_status"].fillna("").astype(str)
    laps["neutralised"] = status.str.contains("[4567]", regex=True)
    pit = laps["pit_in"].fillna(False).astype(bool) | laps["pit_out"].fillna(False).astype(bool)
    wet = laps["compound"].astype(str).str.upper().isin(WET)
    deleted = laps["deleted"].fillna(False).astype(bool) if "deleted" in laps else False
    green = laps["lap_time"].notna() & ~pit & ~laps["neutralised"] & ~wet & ~deleted
    med = laps[green].groupby("driver")["lap_time"].median()
    laps["clean"] = green & (laps["lap_time"] <= 1.07 * laps["driver"].map(med))
    flag = preds.drop_duplicates(["driver", "lap"]).set_index(["driver", "lap"])["clean"]
    known = pd.MultiIndex.from_frame(laps[["driver", "lap"]]).isin(flag.index)
    laps.loc[known, "clean"] = flag.reindex(pd.MultiIndex.from_frame(laps.loc[known, ["driver", "lap"]])).to_numpy().astype(bool)
    return laps


@st.cache_resource(max_entries=8, show_spinner=False)
def race_data(kind: str, key: str, run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if kind == "live":
        preds = data.session_predictions(key)
        return live_laps(key, preds), preds
    pred = data.run_predictions(run)
    return data.race_laps(key), pred[pred["race_label"] == key].assign(feita="backtest")


def stints(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    d = laps[laps["driver"] == driver].sort_values("lap")
    s = d.groupby("stint").agg(start=("lap", "min"), end=("lap", "max"), compound=("compound", "first"),
                               life0=("tyre_life", "first"), life1=("tyre_life", "last"), fresh=("fresh_tyre", "first")).reset_index()
    return s[~s["compound"].astype(str).str.upper().isin(WET) & s["compound"].notna()]


def stint_label(r, first_lap: int) -> str:
    fresh = "" if pd.isna(r.fresh) or bool(r.fresh) else " usado"
    note = f" (gravado a partir da {int(r.start)})" if r.stint == 1 and r.start > first_lap + 1 else ""
    comp = COMPOUND_PT.get(str(r.compound).upper(), str(r.compound).lower())
    return (f"Stint {int(r.stint)}: {comp}{fresh}, voltas {int(r.start)}–{int(r.end)} "
            f"({int(r.end - r.start + 1)} voltas){note}")


def projection(p: pd.DataFrame, mode: str, start: int, end: int, k: int, origin: int | None) -> tuple[pd.DataFrame, list[int]]:
    """As previsões mostradas e as voltas em que foram feitas. Só valem origens a partir da volta de entrada no
    box que abre o stint (antes dela, a previsão é para o pneu anterior)."""
    p = p[(p["t"] >= start - 1) & (p["lap"] >= start) & (p["lap"] <= end)]
    if mode == "começo":
        avail = sorted(p["t"].unique())
        if not avail:
            return p.iloc[:0], []
        anchors = list(range(int(avail[0]), end, 10))
        w = p[p["t"].isin(anchors)].assign(segment=lambda x: x["t"])
        return w, anchors
    if mode == "volta":
        w = p[(p["h"] == k) & (p["t"] == p["lap"] - k)].assign(segment=0)
        return w, []
    w = p[p["t"] == origin].assign(segment=0)
    return w, [origin]


def summary(real: pd.DataFrame, w: pd.DataFrame) -> dict:
    ev = real[real["clean"]].merge(w.drop_duplicates("lap", keep="last"), on="lap")
    if ev.empty:
        return {}
    err = ev["time_p50"] - ev["lap_time"]
    out = {"voltas avaliadas": int(len(ev)), "MAE (s)": err.abs().mean(), "viés (s)": err.mean(),
           "cobertura p10–p90": ((ev["lap_time"] >= ev["time_p10"]) & (ev["lap_time"] <= ev["time_p90"])).mean()}
    if len(ev) >= 4:
        out["degradação real (s/volta)"] = np.polyfit(ev["lap"], ev["lap_time"], 1)[0]
        out["degradação prevista (s/volta)"] = np.polyfit(ev["lap"], ev["time_p50"], 1)[0]
    return out


def render() -> None:
    options = race_options()
    if not options:
        st.info("Não há corridas com previsões: nem gravadas ao vivo em data/live nem runs da LSTM em runs/.")
        return
    labels = [o["label"] for o in options]
    default = next((i for i, o in enumerate(options) if o["key"] == "2026_las_vegas_race"), 0)
    c1, c2, c3 = st.columns([2, 1, 2])
    race = options[labels.index(c1.selectbox("Corrida", labels, index=default, key="stint_race"))]
    laps, preds = race_data(race["kind"], race["key"], race["run"])
    final = laps.sort_values("lap").groupby("driver").tail(1).sort_values(["lap", "position"], ascending=[False, True])
    drivers = [d for d in final["driver"] if d in set(preds["driver"])]
    if not drivers:
        st.info("Sem previsões nesta corrida.")
        return
    drv = c2.selectbox("Piloto (pela posição final)", drivers, key="stint_driver")
    s = stints(laps, drv)
    if s.empty:
        st.info("Sem stints em pneu de pista seca.")
        return
    opts = list(s.itertuples())
    first_lap = int(laps["lap"].min())
    r = c3.selectbox("Stint", opts, index=1 if len(opts) > 1 else 0, key=f"stint_pick_{race['key']}_{drv}",
                     format_func=lambda r: stint_label(r, first_lap))
    start, end = int(r.start), int(r.end)
    p = preds[preds["driver"] == drv]
    model = "lstm" if race["run"].startswith("lstm") else "gru"

    c1, c2 = st.columns([2, 3])
    mode = c1.radio("Projeção", ["começo", "volta", "escolhida"], horizontal=True, key="stint_mode",
                    format_func={"começo": "do começo do stint", "volta": "volta a volta",
                                 "escolhida": "a partir de uma volta"}.get)
    k, origin = 1, None
    if mode == "volta":
        k = c2.slider("Feita quantas voltas antes", 1, 10, 1, key="stint_k")
    elif mode == "escolhida":
        avail = sorted(int(t) for t in p.loc[(p["t"] >= start - 1) & (p["t"] < end), "t"].unique())
        if not avail:
            st.info("Sem previsões feitas dentro deste stint.")
            return
        origin = c2.select_slider("Previsão feita ao fim da volta", avail, value=avail[0], key=f"stint_origin_{start}")
    w, origins = projection(p, mode, start, end, k, origin)
    real = laps[(laps["driver"] == drv) & laps["lap"].between(start, end)].copy()
    real["clean"] = real["clean"].fillna(False).astype(bool)
    if w.empty:
        st.info("Sem previsão para este stint (a gravação ao vivo pode ter começado depois, ou o previsor estava parado).")
        return
    w = w.copy()
    w["hover"] = [f"volta {int(l)} · prevista ao fim da volta {int(t)} ({int(h)} {'volta' if h == 1 else 'voltas'} antes)"
                  + ("" if f in ("ao vivo", "backtest") else ", refeita") for l, t, h, f in zip(w["lap"], w["t"], w["h"], w["feita"])]
    st.plotly_chart(stint_projection(real, w, model, origins), width="stretch", config={"displaylogo": False})

    sm = summary(real, w)
    life = f"o pneu foi de {int(r.life0)} a {int(r.life1)} voltas de uso" if pd.notna(r.life0) and pd.notna(r.life1) else ""
    st.markdown(f"**{drv} · {stint_label(r, first_lap)}**" + (f" · {life}" if life else ""))
    if sm:
        st.dataframe(pd.DataFrame([sm]), hide_index=True, width="stretch",
                     column_config={"MAE (s)": st.column_config.NumberColumn(format="%.3f"),
                                    "viés (s)": st.column_config.NumberColumn(format="%+.3f"),
                                    "cobertura p10–p90": st.column_config.NumberColumn(format="percent"),
                                    "degradação real (s/volta)": st.column_config.NumberColumn(format="%+.3f"),
                                    "degradação prevista (s/volta)": st.column_config.NumberColumn(format="%+.3f")})
    how = {"começo": f"Previsões feitas ao fim das voltas {', '.join(map(str, origins))}, cada uma para as 10 voltas seguintes.",
           "volta": f"Para cada volta, a previsão feita {k} {'volta' if k == 1 else 'voltas'} antes.",
           "escolhida": f"As voltas que a rede projetou ao fim da volta {origin}."}[mode]
    scenario = ("Ao vivo, a previsão supõe o mesmo pneu e nenhuma parada (cenário A); origens antes da parada que abre o "
                "stint não entram. " if race["kind"] == "live" else
                "Corrida fora do treino: as variáveis conhecidas do futuro (pneu, paradas) são as reais. ")
    refeitas = (w["feita"] == "refeita").mean() if race["kind"] == "live" else 0
    st.caption(f"{how} {scenario}Pontos cheios: voltas limpas, as que entram na avaliação e na tendência; vazios: box, SC/VSC "
               "ou fora do padrão (as voltas lentas demais ficam fora do gráfico). Faixas cinza: SC/VSC. A tendência é a reta "
               "ajustada ao tempo de volta ao longo do stint: a degradação, em segundos por volta, real e projetada."
               + (f" {refeitas:.0%} destas previsões foram refeitas depois da corrida (as feitas ao vivo se perderam na queda "
                  "do previsor); o modelo é o mesmo." if refeitas else ""))
    if race["kind"] == "live" and mode == "começo" and origins and origins[0] >= start:
        st.caption(f"A primeira previsão do stint é do fim da volta {origins[0]}: ao vivo, o previsor não prevê na volta de "
                   "entrada no box (o pneu ainda vai mudar), então a primeira previsão com o pneu novo sai no fim da volta "
                   "de saída.")
