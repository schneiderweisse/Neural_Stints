"""Aba LSTM: o modelo ao vivo da v1 (ensemble de 5 LSTMs com o embedding de regulamento) por dentro.

- Arquitetura: o encoder–decoder com as dimensões reais, a tabela de camadas e parâmetros
  (runs/<run>/model_card.json, de modeling/model_card.py) e a célula LSTM com os quatro portões.
- Portões: quanto cada entrada pesa em cada portão do encoder e do decoder, o peso recorrente e o
  viés de cada portão.
- Embeddings: o que a rede aprendeu de circuitos, pilotos e equipes, em 2 dimensões (PCA de uma semente).
- Treino: curvas por semente com a parada antecipada, os hiperparâmetros e a busca que escolheu o
  tamanho do estado e o dropout (bateria modeling.cells).
- Avaliação: erro por horizonte contra a GRU e o ingênuo, calibração, recortes por composto e por
  stint, por corrida e o erro das previsões feitas ao vivo.

O painel não abre o model.pt (a demo roda no navegador, sem PyTorch): sem a ficha, as seções de
arquitetura, portões e embeddings pedem para rodar modeling/model_card.py.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data
from dashboard.model_tab import SPLIT_LABEL, tile
from modeling.figures import THEMES, base_layout, coverage_by_horizon, grouped_bars, mae_by_horizon, training_curve

C = THEMES["dark"]
FEATURE_PT = {
    "pace_in": "ritmo da volta", "s1_pace": "ritmo no setor 1", "s2_pace": "ritmo no setor 2", "s3_pace": "ritmo no setor 3",
    "speed_i1_rel": "velocidade no ponto 1", "speed_i2_rel": "velocidade no ponto 2", "speed_fl_rel": "velocidade na linha",
    "speed_st_rel": "velocidade no speed trap", "gap_ahead": "gap para o da frente", "gap_leader": "gap para o líder",
    "position": "posição", "track_temp": "temperatura da pista", "air_temp": "temperatura do ar", "humidity": "umidade",
    "clean_live": "volta limpa", "neutralised": "SC/VSC", "yellow": "bandeira amarela", "traffic": "tráfego",
    "rain": "chuva", "race_progress": "progresso da corrida", "laps_remaining": "voltas restantes",
    "tyre_life": "idade do pneu", "stint_lap": "volta no stint", "stint": "número do stint", "fresh_tyre": "pneu novo",
    "pit_in": "entrada no box", "pit_out": "saída do box", "is_soft": "pneu macio", "is_medium": "pneu médio",
    "is_hard": "pneu duro", "laps_total": "voltas da corrida", "ref_from_quali": "referência do quali",
}
CAT_PT = {"circuit": "circuito", "driver": "piloto", "team_year": "equipe-ano", "regulation": "regulamento"}
HYPER = [  # (chave, nome, para que serve)
    ("hidden", "Estado (h e c)", "unidades de cada LSTM: a memória da corrida"),
    ("context", "Contexto estático", "resumo de circuito, piloto, equipe e regulamento"),
    ("dropout", "Dropout", "zera ao acaso partes do estado inicial e da saída do decoder no treino"),
    ("cat_dropout", "Dropout das categorias", "troca ao acaso a categoria por <unk>, para a rede saber lidar com pilotos e circuitos novos"),
    ("lr", "Taxa de aprendizado (Adam)", "tamanho do passo do otimizador"),
    ("weight_decay", "Decaimento dos pesos", "puxa os pesos para zero (regularização L2)"),
    ("batch", "Lote", "corridas de um piloto por passo do otimizador"),
    ("max_epochs", "Máximo de épocas", "limite do treino"),
    ("patience", "Paciência (parada antecipada)", "épocas sem melhora na validação antes de parar"),
    ("clip", "Corte do gradiente", "norma máxima do gradiente, contra explosões na retropropagação no tempo"),
]


def card(run_id: str) -> dict | None:
    path = data.RUNS / run_id / "model_card.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def search_table(cfg: dict) -> pd.DataFrame | None:
    """Bateria que escolheu o tamanho do estado e o dropout (summary.csv da pasta dos membros)."""
    paths = (cfg.get("ensemble") or {}).get("member_paths") or []
    if not paths:
        return None
    path = data.RUNS / paths[0].split("/")[0] / "summary.csv"
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ figuras
def architecture_svg(k: dict) -> str:
    """O encoder–decoder com as dimensões do modelo."""
    f, h, ctx = k["features"], k["hidden"], k["context"]
    cell = k["cell"].upper()
    emb = " · ".join(f"{CAT_PT.get(c, c)} ({d})" for c, d in zip(f["static_cat"], [e["dims"] for e in k["embeddings"].values()]))
    box = lambda x, y, w, hh, title, lines, color: (
        f"<rect x='{x}' y='{y}' width='{w}' height='{hh}' rx='8' fill='#161b22' stroke='{color}' stroke-width='1.6'/>"
        f"<text x='{x + 12}' y='{y + 22}' fill='#e6edf3' font-size='14' font-weight='600'>{title}</text>" +
        "".join(f"<text x='{x + 12}' y='{y + 42 + 17 * i}' fill='#9aa4af' font-size='12'>{t}</text>" for i, t in enumerate(lines)))
    arrow = lambda x1, y1, x2, y2: (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#7d8590' stroke-width='1.6' "
                                   f"marker-end='url(#arr)'/>")
    g = C["lstm"]
    parts = ["<svg viewBox='0 0 1180 390' width='100%' role='img' aria-label='arquitetura da rede' "
             "style='font-family:system-ui,-apple-system,Segoe UI,sans-serif'>",
             "<defs><marker id='arr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
             "<path d='M0,0 L10,5 L0,10 z' fill='#7d8590'/></marker></defs>",
             box(10, 30, 250, 118, "Voltas 1 … t (o passado)", [f"{len(f['observed'])} observadas: ritmo, setores,",
                 "velocidades, gaps, clima, bandeiras", f"{len(f['known'])} conhecidas: pneu, idade, stint,", "paradas, progresso"], "#30363d"),
             box(300, 30, 230, 118, f"Encoder {cell}", [f"{h} unidades · 4 portões", "lê a corrida volta a volta",
                 f"entrada: {len(f['observed']) + len(f['known']) + ctx} por volta"], g),
             box(570, 46, 150, 86, "Estado na volta t", ["memória (c) e", f"saída (h), {h} cada"], g),
             box(760, 30, 230, 118, f"Decoder {cell}", [f"{h} unidades · 4 portões", f"{k['horizon']} passos: voltas t+1 … t+{k['horizon']}",
                 f"entrada: {len(f['known']) + ctx} por volta"], g),
             box(1020, 30, 150, 118, "Saída", ["dropout 0,1", f"linear {h} → 3", "p10 ≤ p50 ≤ p90", "do ritmo relativo"], "#3987e5"),
             box(760, 196, 230, 86, "Voltas t+1 … t+10", [f"só as {len(f['known'])} conhecidas: pneu,", "paradas, progresso (nunca o ritmo)"], "#30363d"),
             box(10, 250, 520, 118, "Contexto estático", [f"embeddings: {emb}", f"+ {len(f['static_x'])} numéricas: voltas da corrida, referência do quali",
                 f"→ linear {ctx} + tanh: inicializa h e c do encoder", "e entra em cada volta do encoder e do decoder"], "#d29922"),
             arrow(260, 89, 296, 89), arrow(530, 89, 566, 89), arrow(720, 89, 756, 89), arrow(990, 89, 1016, 89),
             arrow(875, 196, 875, 152),
             arrow(330, 250, 360, 152), arrow(530, 300, 800, 152),
             "<text x='1095' y='175' fill='#9aa4af' font-size='12' text-anchor='middle'>× referência</text>",
             "<text x='1095' y='192' fill='#9aa4af' font-size='12' text-anchor='middle'>= tempo de volta</text>",
             "</svg>"]
    return "".join(parts)


def gate_heatmap(gs: dict, title: str) -> go.Figure:
    """Peso médio absoluto de cada entrada em cada portão; as 16 entradas do contexto viram uma linha (média)."""
    w = np.array(gs["input_weight"])
    names = gs["inputs"]
    ctx = [i for i, n in enumerate(names) if n.startswith("contexto")]
    rows = [i for i in range(len(names)) if i not in ctx]
    z = np.vstack([w[rows], w[ctx].mean(axis=0, keepdims=True)]) if ctx else w[rows]
    labels = [FEATURE_PT.get(names[i], names[i]) for i in rows] + ([f"contexto (média das {len(ctx)})"] if ctx else [])
    fig = go.Figure(go.Heatmap(z=z, x=gs["gates"], y=labels, zmin=float(z.min()), zmax=float(z.max()),
                               colorscale=[[0, "#0d1117"], [0.35, "#14533f"], [0.7, C["lstm"]], [1, "#7ee8c2"]],
                               colorbar=dict(title=dict(text="|peso|", font=dict(color=C["muted"])), tickfont=dict(color=C["muted"])),
                               hovertemplate="%{y} → portão de %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_layout(title=dict(text=title, font=dict(color=C["ink"], size=13), x=0, xanchor="left"))
    fig = base_layout(fig, 28 * len(labels) + 120, "dark", top=48)
    fig.update_layout(hovermode="closest", margin=dict(l=190))
    return fig


def gate_bars(k: dict) -> go.Figure:
    fig = go.Figure()
    for name, gs, color in (("encoder", k["encoder"], C["lstm"]), ("decoder", k["decoder"], "#3987e5")):
        fig.add_trace(go.Bar(x=gs["gates"], y=gs["bias_mean"], name=f"viés · {name}", marker=dict(color=color, cornerradius=4),
                             error_y=dict(type="data", array=gs["bias_std"], color=C["muted"], thickness=1),
                             hovertemplate=f"{name} · %{{x}}: viés médio %{{y:.3f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title_text="viés médio (± desvio entre unidades)")
    fig = base_layout(fig, 300, "dark")
    fig.update_layout(hovermode="closest")
    return fig


def embedding_scatter(e: dict, cat: str) -> go.Figure:
    names = e["names"]
    keep = [i for i, n in enumerate(names) if n != "<unk>"]
    fig = go.Figure(go.Scatter(x=[e["x"][i] for i in keep], y=[e["y"][i] for i in keep], mode="markers+text",
                               text=[names[i] for i in keep], textposition="top center",
                               textfont=dict(color=C["ink2"], size=11), marker=dict(size=9, color=C["lstm"]),
                               hovertemplate="%{text}<extra></extra>"))
    if "<unk>" in names:
        i = names.index("<unk>")
        fig.add_trace(go.Scatter(x=[e["x"][i]], y=[e["y"][i]], mode="markers+text", text=["<unk>"], textposition="bottom center",
                                 textfont=dict(color=C["muted"], size=11), marker=dict(size=9, color=C["muted"], symbol="x"),
                                 hovertemplate="categoria desconhecida<extra></extra>"))
    var = e["explained"]
    ax = (lambda i: f"componente {i + 1} ({var[i]:.0%} da variância)") if e["pca"] else (lambda i: f"dimensão {i + 1}")
    fig.update_xaxes(title_text=ax(0))
    fig.update_yaxes(title_text=ax(1))
    fig = base_layout(fig, 460, "dark", top=30)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def mae_h1_curve(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (seed, h) in enumerate(history.groupby("seed")):
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_mae_h1"], mode="lines", name=f"semente {seed}", opacity=0.85,
                                 line=dict(width=1.6), hovertemplate=f"semente {seed} · época %{{x}}: %{{y:.3f}} s<extra></extra>"))
    best = history.loc[history.groupby("seed")["val_loss"].idxmin()]
    fig.add_trace(go.Scatter(x=best["epoch"], y=best["val_mae_h1"], mode="markers", name="melhor época",
                             marker=dict(symbol="diamond", size=9, color=C["lstm"], line=dict(color=C["surface"], width=1)),
                             hovertemplate="melhor época %{x}: %{y:.3f} s<extra></extra>"))
    fig.update_xaxes(title_text="época")
    fig.update_yaxes(title_text="MAE na validação, 1 volta à frente (s)")
    return base_layout(fig, 320, "dark")


def search_figure(s: pd.DataFrame, chosen: tuple) -> go.Figure:
    g = (s.groupby(["cell", "hidden", "dropout"])
         .agg(pin=("val_pinball", "mean"), sd=("val_pinball", "std"), mae=("val_mae", "mean"), ep=("best_epoch", "mean"),
              params=("n_params", "first"), n=("seed", "size")).reset_index().sort_values("pin", ascending=False))
    labels = [f"{c.upper()} · estado {h} · dropout {str(d).replace('.', ',')}" for c, h, d in zip(g["cell"], g["hidden"], g["dropout"])]
    colors = [C["lstm"] if (c, h, d) == chosen else (C["gru"] if c == "gru" else C["muted"])
              for c, h, d in zip(g["cell"], g["hidden"], g["dropout"])]
    fig = go.Figure(go.Scatter(x=g["pin"], y=labels, mode="markers", marker=dict(size=11, color=colors),
                               error_x=dict(type="data", array=g["sd"], color=C["muted"], thickness=1.2),
                               customdata=np.c_[g["mae"], g["ep"], g["params"], g["n"]],
                               hovertemplate="%{y}<br>pinball %{x:.4f} s · MAE %{customdata[0]:.3f} s<br>"
                                             "melhor época média %{customdata[1]:.1f} · %{customdata[2]:,} parâmetros · "
                                             "%{customdata[3]} sementes<extra></extra>"))
    fig.update_xaxes(title_text="perda pinball na validação (s), média ± desvio entre sementes")
    fig = base_layout(fig, 60 + 38 * len(g), "dark", top=24)
    fig.update_layout(hovermode="closest", margin=dict(l=230), showlegend=False)
    return fig


def live_metrics(run_id: str) -> list[dict]:
    """Erro das previsões feitas ao vivo por este modelo (data/live/<corrida>/predictions.parquet)."""
    out = []
    for p in sorted(data.LIVE.glob("*/predictions.parquet")):
        pr = pd.read_parquet(p)
        if "gru_run" not in pr or not (pr["gru_run"] == run_id).any():
            continue
        ok = pr["time_real"].notna() & pr["clean"].astype(bool) & (pr["gru_run"] == run_id)
        d = pr[ok]
        if d.empty:
            continue
        err = (d["time_p50"] - d["time_real"]).abs()
        cov = ((d["time_real"] >= d["time_p10"]) & (d["time_real"] <= d["time_p90"])).mean()
        out.append({"session": p.parent.name, "n": int(len(d)), "mae": float(err.mean()), "coverage": float(cov),
                    "by_h": err.groupby(d["h"]).mean().to_dict()})
    return out


@st.cache_resource(max_entries=4, show_spinner=False)
def eval_figures(run_id: str, gru_id: str | None, naive_id: str) -> dict:
    R, N = data.run_files(run_id), data.run_files(naive_id)
    G = data.run_files(gru_id) if gru_id else None
    models = {"lstm": R["metrics"], "naive": N["metrics"]} | ({"gru": G["metrics"]} if G else {})
    by_h = {m: {"races": {SPLIT_LABEL[s]: {"by_h": {int(h): v["mae_s"] for h, v in r["splits"][s]["by_h"].items()}}
                          for s in ("val", "test")}} for m, r in models.items()}
    order = {m: by_h[m] for m in ("lstm", "gru", "naive") if m in by_h}
    lt = R["metrics"]["splits"]["test"]
    comps = [c for c in ("SOFT", "MEDIUM", "HARD") if c in lt["by_compound"]]
    cats, keys = ["mesmo stint", "depois da parada"], ["True", "False"]
    comp_vals = {m: [models[m]["splits"]["test"]["by_compound"][c]["mae_s"] for c in comps] for m in order}
    stint_vals = {m: [models[m]["splits"]["test"]["by_same_stint"][k]["mae_s"] for k in keys] for m in order}
    cov = {s: {int(h): v["coverage_80"] for h, v in R["metrics"]["splits"][s]["by_h"].items()} for s in ("val", "test")}
    return {"by_h": mae_by_horizon(order, ["validação", "teste"], theme="dark"),
            "coverage": coverage_by_horizon(cov, model="lstm"),
            "compound": grouped_bars(comps, comp_vals, "MAE (s)"), "stint": grouped_bars(cats, stint_vals, "MAE (s)")}


def race_table(run_id: str, gru_id: str | None, naive_id: str) -> pd.DataFrame:
    R, N = data.run_files(run_id)["metrics"], data.run_files(naive_id)["metrics"]
    G = data.run_files(gru_id)["metrics"] if gru_id else None
    rows = []
    for split in ("val", "test"):
        for race, m in R["splits"][split]["by_race"].items():
            g = G["splits"][split]["by_race"].get(race, {}) if G else {}
            n = N["splits"][split]["by_race"].get(race, {})
            rows.append({"divisão": SPLIT_LABEL[split], "corrida": race, "MAE LSTM (s)": m["mae_s"],
                         "MAE GRU (s)": g.get("mae_s"), "MAE ingênuo (s)": n.get("mae_s"),
                         "cobertura p10–p90": m.get("coverage_80"), "voltas avaliadas": m["n"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ aba
def render() -> None:
    runs = data.runs("lstm")
    naives = data.runs("naive")
    if not runs or not naives:
        st.info("Ainda não há runs da LSTM em runs/. Rode `python -m modeling.cells` e `python -m modeling.ensemble`.")
        return
    grus = data.runs("gru")
    run_id = st.selectbox("Run da LSTM", runs, key="lstm_run") if len(runs) > 1 else runs[0]
    R = data.run_files(run_id)
    cfg, m = R["config"], R["metrics"]
    hp, ens = cfg["hyperparameters"], cfg.get("ensemble") or {}
    k = card(run_id)
    gru_id, naive_id = (grus[0] if grus else None), naives[0]
    lt = m["splits"]["test"]
    nt = data.run_files(naive_id)["metrics"]["splits"]["test"]

    st.markdown(f"**Modelo ao vivo da v1: `{run_id}`**, em uso desde o GP de Las Vegas. É a mesma rede encoder–decoder da GRU, "
                "com células LSTM e o embedding de regulamento, treinada com 5 sementes; a previsão é a média dos quantis "
                "das 5 redes.")
    cols = st.columns(5)
    tile(cols[0], "MAE no teste", f"{lt['overall']['mae_s']:.3f} s", f"ingênuo {nt['overall']['mae_s']:.3f} s")
    tile(cols[1], "Melhora sobre o ingênuo", f"{lt['skill_vs_reference']['overall']:+.0%}",
         f"de {lt['skill_vs_reference']['by_h']['1']:+.0%} (h=1) a {lt['skill_vs_reference']['by_h']['10']:+.0%} (h=10)")
    tile(cols[2], "Cobertura p10–p90", f"{lt['overall']['coverage_80']:.0%}", "alvo: 80%")
    n_member = ens.get("n_params_member") or (k or {}).get("params_member")
    tile(cols[3], "Parâmetros", f"{len(ens.get('members', [1]))} × {n_member:,}".replace(",", "."),
         f"estado {hp['hidden']} · contexto {hp['context']}")
    tile(cols[4], "Treino", f"{cfg['train_seconds'] / 60:.0f} min",
         f"melhores épocas {min(ens['best_epochs'])}–{max(ens['best_epochs'])} · CPU" if ens else "")

    arch, gates, emb, train, ev = st.tabs(["Arquitetura", "Portões", "Embeddings", "Treino", "Avaliação"])

    with arch:
        if k is None:
            st.info(f"Falta a ficha do modelo: rode `python -m modeling.model_card runs/{run_id}`.")
        else:
            st.markdown(architecture_svg(k), unsafe_allow_html=True)
            st.caption("A previsão feita ao fim da volta t parte do estado do encoder naquela volta. O decoder recebe só o que "
                       "se sabe das voltas futuras e nunca o ritmo previsto, então os erros não se acumulam de uma volta para "
                       "a outra. No treino e no teste, essas variáveis são as reais (inclusive as paradas); ao vivo, é o "
                       "cenário A: mesmo pneu, sem parada. A saída é o ritmo relativo à melhor volta do piloto no quali; "
                       "vezes essa referência, vira tempo de volta.")
            layers = pd.DataFrame(k["layers"]).rename(columns={"camada": "Camada", "tipo": "Tipo", "entrada": "Entrada",
                                                               "saida": "Saída", "parametros": "Parâmetros"})
            st.dataframe(layers, hide_index=True, width="stretch",
                         column_config={"Parâmetros": st.column_config.NumberColumn(format="%d")})
            thousands = lambda v: f"{v:,}".replace(",", ".")
            st.caption(f"Total por rede: {thousands(k['params_member'])} parâmetros; o ensemble tem {k['members']} redes, "
                       f"{thousands(k['params_total'])} no total. " +
                       f"Os dois LSTMs têm {k['cell'].upper() == 'LSTM' and 4 or 3} blocos de pesos de {k['hidden']} unidades "
                       "cada (um por portão), por isso os 10 mil parâmetros do encoder.")
            st.markdown("**A célula LSTM**")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.latex(r"""\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{portão de entrada}\\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{portão de esquecimento}\\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) && \text{candidato}\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{portão de saída}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{memória}\\
h_t &= o_t \odot \tanh(c_t) && \text{saída}
\end{aligned}""")
            with c2:
                st.markdown(f"A cada volta, a célula decide o que esquecer da memória ($f$), o que escrever nela ($i$ e $g$) "
                            f"e o que mostrar adiante ($o$). Com $x_t$ de {len(k['encoder']['inputs'])} entradas e "
                            f"{k['hidden']} unidades, cada $W$ do encoder é uma matriz {k['hidden']} × "
                            f"{len(k['encoder']['inputs'])} e cada $U$, {k['hidden']} × {k['hidden']}; o PyTorch empilha os "
                            "quatro portões numa matriz só. A memória $c$ passa de volta em volta só por soma e produto, o que "
                            "deixa o gradiente atravessar a corrida inteira sem sumir, o problema da RNN simples.")

    with gates:
        if k is None:
            st.info(f"Falta a ficha do modelo: rode `python -m modeling.model_card runs/{run_id}`.")
        else:
            which = st.radio("Camada", ["Encoder (o passado)", "Decoder (as voltas futuras)"], horizontal=True, key="lstm_gate_layer")
            gs = k["encoder"] if which.startswith("Encoder") else k["decoder"]
            st.plotly_chart(gate_heatmap(gs, "Peso médio absoluto de cada entrada em cada portão"), width="stretch",
                            config={"displaylogo": False})
            st.caption(f"Média de |peso| sobre as {k['hidden']} unidades e as {k['members']} sementes. As entradas numéricas "
                       "chegam padronizadas (média 0, desvio 1) e as booleanas como 0 ou 1, então a cor compara a força "
                       "com que cada entrada mexe em cada portão, sem dizer o sinal. É uma leitura dos pesos da primeira "
                       "camada, não uma medida de importância das variáveis.")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(gate_bars(k), width="stretch", config={"displaylogo": False})
            with c2:
                enc = k["encoder"]
                rec = ", ".join(f"{g.split(' ')[0]} {w:.3f}" for g, w in zip(enc["gates"], enc["recurrent_weight"]))
                lo, hi = min(enc["bias_mean"]), max(enc["bias_mean"])
                bound = 1 / np.sqrt(k["hidden"])
                st.markdown("**Viés e peso recorrente**")
                st.caption("O viés diz como cada portão se comporta sem estímulo: σ(0) = 0,5, então um viés perto de zero no "
                           "esquecimento guarda metade da memória por padrão, e o resto depende das entradas e do estado. O "
                           f"viés de cada portão é a soma de dois vetores (b_ih e b_hh), que o PyTorch sorteia entre ±{bound:.2f} "
                           f"cada; depois do treino, as médias por portão do encoder ficaram entre {lo:.2f} e {hi:.2f}, com "
                           "bastante variação entre as unidades (as barras). "
                           f"Peso recorrente médio |U| do encoder, por portão: {rec}.")

    with emb:
        if k is None:
            st.info(f"Falta a ficha do modelo: rode `python -m modeling.model_card runs/{run_id}`.")
        else:
            cats = [c for c in k["embeddings"] if k["embeddings"][c]["pca"] or len(k["embeddings"][c]["names"]) > 3]
            cat = st.radio("Embedding", cats, format_func=lambda c: CAT_PT.get(c, c), horizontal=True, key="lstm_emb")
            e = k["embeddings"][cat]
            st.plotly_chart(embedding_scatter(e, cat), width="stretch", config={"displaylogo": False})
            st.caption(f"Cada {CAT_PT.get(cat, cat)} vira um vetor de {e['dims']} números aprendido no treino, que entra no "
                       f"contexto. Aqui, os vetores da semente {k['embedding_member']} projetados em 2 dimensões por PCA: "
                       "pontos próximos são tratados de forma parecida pela rede. Cada semente aprende os seus próprios "
                       "eixos, por isso não há média entre elas. <unk> é a categoria de quem não estava no treino.")

    with train:
        hist = R.get("history")
        if hist is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Perda por época (pinball, alvo padronizado)**")
                st.plotly_chart(training_curve(hist), width="stretch", config={"displaylogo": False})
            with c2:
                st.markdown("**Erro na validação, 1 volta à frente**")
                st.plotly_chart(mae_h1_curve(hist), width="stretch", config={"displaylogo": False})
            stop = hist.groupby("seed").agg(épocas=("epoch", "max"), melhor=("val_loss", "idxmin"))
            stop["melhor época"] = hist.loc[stop["melhor"], "epoch"].to_numpy()
            stop["perda na validação"] = hist.loc[stop["melhor"], "val_loss"].to_numpy()
            stop["MAE h=1 (s)"] = hist.loc[stop["melhor"], "val_mae_h1"].to_numpy()
            stop["segundos por época"] = hist.groupby("seed")["seconds"].mean().to_numpy()
            stop = stop.drop(columns="melhor").reset_index().rename(columns={"seed": "semente", "épocas": "épocas rodadas"})
            st.dataframe(stop, hide_index=True, width="stretch",
                         column_config={"perda na validação": st.column_config.NumberColumn(format="%.4f"),
                                        "MAE h=1 (s)": st.column_config.NumberColumn(format="%.3f"),
                                        "segundos por época": st.column_config.NumberColumn(format="%.1f")})
            st.caption(f"Parada antecipada: cada semente treina até {hp['patience']} épocas seguidas sem melhorar a perda na "
                       f"validação e volta aos pesos da melhor época. Por isso cada uma roda a melhor época + {hp['patience']}. "
                       "A perda de treino segue caindo depois da melhor época, enquanto a de validação para ou sobe: o começo "
                       "do sobreajuste, que a parada corta.")
        st.markdown("**Hiperparâmetros**")
        rows = [{"hiperparâmetro": name, "valor": str(hp[key]).replace(".", ","), "para que serve": why}
                for key, name, why in HYPER if key in hp]
        rows.append({"hiperparâmetro": "Perda", "valor": "pinball nos quantis 0,1, 0,5 e 0,9",
                     "para que serve": "ensina a mediana e a faixa de 80% ao mesmo tempo; só nas voltas limpas"})
        rows.append({"hiperparâmetro": "Embeddings", "valor": ", ".join(f"{CAT_PT.get(c, c)} {d}" for c, d in hp["emb_dims"].items()),
                     "para que serve": "dimensões dos vetores das categorias"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        s = search_table(cfg)
        if s is not None:
            st.markdown("**A escolha do tamanho e do dropout**")
            st.plotly_chart(search_figure(s, ("lstm", hp["hidden"], hp["dropout"])), width="stretch",
                            config={"displaylogo": False})
            st.caption("Cada configuração treinada com 5 sementes, na mesma divisão; o critério é a perda pinball média na "
                       "validação (o teste não entra na escolha). Em verde, a escolhida; em azul, as GRUs. O estado de 64 "
                       "aprende rápido e sobreajusta cedo (melhor época ~9) e o de 26 fica curto; o dropout de 0,2 empata "
                       "com o de 0,1, que foi o escolhido.")

    with ev:
        figs = eval_figures(run_id, gru_id, naive_id)
        st.markdown("**Erro por horizonte: LSTM × GRU × ingênuo**")
        st.plotly_chart(figs["by_h"], width="stretch", config={"displaylogo": False})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Calibração do intervalo p10–p90**")
            st.plotly_chart(figs["coverage"], width="stretch", config={"displaylogo": False})
            o = lt["overall"]
            st.caption(f"No teste, {o['below_p10']:.0%} das voltas ficaram abaixo do p10 (ideal 10%) e {o['below_p90']:.0%} "
                       f"abaixo do p90 (ideal 90%). Largura média do intervalo: {o['width_s']:.2f} s.")
        with c2:
            st.markdown("**Erro por composto (teste)**")
            st.plotly_chart(figs["compound"], width="stretch", config={"displaylogo": False})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mesmo stint × depois de uma parada (teste)**")
            st.plotly_chart(figs["stint"], width="stretch", config={"displaylogo": False})
        with c2:
            st.markdown("**Ao vivo**")
            lives = live_metrics(run_id)
            if not lives:
                st.caption("Ainda não há previsões feitas ao vivo por este modelo em data/live.")
            for lv in lives:
                h1, h10 = lv["by_h"].get(1), lv["by_h"].get(10)
                n_laps = f"{lv['n']:,}".replace(",", ".")
                st.markdown(f"`{lv['session']}`: MAE de **{lv['mae']:.3f} s** em {n_laps} voltas previstas e limpas, "
                            f"cobertura de {lv['coverage']:.0%}" +
                            (f"; de {h1:.2f} s (1 volta à frente) a {h10:.2f} s (10 voltas)" if h1 and h10 else "") + ".")
            st.caption("Só as previsões feitas durante a corrida (as refeitas depois de uma queda do previsor ficam de fora). "
                       "No teste, as voltas são de corridas inteiras; ao vivo, o tempo real vem do feed.")
        st.markdown("**Por corrida**")
        st.dataframe(race_table(run_id, gru_id, naive_id), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("MAE LSTM (s)", "MAE GRU (s)", "MAE ingênuo (s)")} |
                     {"cobertura p10–p90": st.column_config.NumberColumn(format="percent")})
