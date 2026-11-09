"""Figuras Plotly dos resultados, reaproveitadas no relatório de backtest (tema claro) e no painel (escuro).

Cores com papel fixo (validadas para daltonismo nos dois temas): real em tinta neutra, GRU em azul,
ingênuo em laranja. Compostos nas cores convencionais da F1.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  gru="#2a78d6", naive="#eb6834", band="rgba(42,120,214,0.16)", shade="rgba(137,135,129,0.14)",
                  hover="#ffffff", hard="#52514e"),
    "dark": dict(surface="#0d1117", ink="#e6edf3", ink2="#c9d1d9", muted="#7d8590", grid="#21262d", axis="#30363d",
                 gru="#3987e5", naive="#d95926", band="rgba(57,135,229,0.22)", shade="rgba(125,133,144,0.18)",
                 hover="#161b22", hard="#e6edf3"),
}
COMPOUND = {"SOFT": "#f85149", "MEDIUM": "#d29922", "INTERMEDIATE": "#3fb950", "WET": "#2f81f7"}
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def colors(theme: str) -> dict:
    return THEMES[theme]


def compound_color(compound, theme: str) -> str:
    return COMPOUND.get(str(compound).upper(), THEMES[theme]["hard"])


def model_style(model: str, theme: str) -> tuple[str, str, str]:
    c = THEMES[theme]
    return {"gru": ("GRU v0", c["gru"], "solid"), "naive": ("Ingênuo", c["naive"], "dash")}[model]


def fmt_lap(seconds: float) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:06.3f}"


def base_layout(fig: go.Figure, height: int, theme: str = "light", top: int = 84) -> go.Figure:
    c = THEMES[theme]
    fig.update_layout(height=height, paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
                      margin=dict(l=64, r=24, t=top, b=48), font=dict(family=FONT, color=c["ink2"], size=12),
                      hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1, yref="container", yanchor="top", font=dict(color=c["ink2"])),
                      hoverlabel=dict(bgcolor=c["hover"], bordercolor=c["axis"], font=dict(family=FONT, color=c["ink"])))
    fig.update_xaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, tickfont=dict(color=c["muted"]))
    fig.update_yaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, tickfont=dict(color=c["muted"]))
    for a in fig.layout.annotations:                       # títulos dos painéis
        if a.font is None or a.font.color is None:
            a.font = dict(color=c["ink"], size=13)
    return fig


def mae_by_horizon(metrics: dict[str, dict], races: list[str], theme: str = "light") -> go.Figure:
    """MAE (s) por horizonte, um painel por corrida (ou divisão), uma linha por modelo."""
    c = THEMES[theme]
    fig = make_subplots(rows=1, cols=len(races), shared_yaxes=True, subplot_titles=races, horizontal_spacing=0.06)
    for a in fig.layout.annotations:
        a.font = dict(color=c["ink"], size=13)
    for model, m in metrics.items():
        name, color, dash = model_style(model, theme)
        for j, race in enumerate(races, start=1):
            by_h = m["races"][race]["by_h"]
            hs = sorted(by_h)
            ys = [by_h[h] for h in hs]
            fig.add_trace(go.Scatter(x=hs, y=ys, name=name, legendgroup=model, showlegend=j == 1, mode="lines+markers",
                                     line=dict(color=color, width=2, dash=dash), marker=dict(size=8, color=color),
                                     hovertemplate=f"{name}: %{{y:.2f}} s<extra></extra>"), row=1, col=j)
            fig.add_annotation(x=hs[-1], y=ys[-1], text=name, xanchor="left", xshift=8, showarrow=False,
                               font=dict(color=c["ink2"], size=11), row=1, col=j)
    fig.update_xaxes(title_text="voltas à frente (h)", dtick=1, range=[0.5, 12.5])
    fig.update_yaxes(title_text="MAE (s)", row=1, col=1, rangemode="tozero")
    return base_layout(fig, 360, theme)


def coverage_by_horizon(by_h: dict[str, dict], theme: str = "dark") -> go.Figure:
    """Cobertura do intervalo p10–p90 por horizonte, contra o alvo de 80%."""
    c = THEMES[theme]
    fig = go.Figure()
    fig.add_hline(y=80, line=dict(color=c["muted"], width=1, dash="dot"))
    fig.add_annotation(x=10.4, y=80, text="alvo 80%", xanchor="left", showarrow=False, font=dict(color=c["muted"], size=11))
    styles = {"val": ("validação", "dash"), "test": ("teste", "solid")}
    for split, values in by_h.items():
        label, dash = styles[split]
        hs = sorted(values)
        fig.add_trace(go.Scatter(x=hs, y=[100 * values[h] for h in hs], name=label, mode="lines+markers",
                                 line=dict(color=c["gru"], width=2, dash=dash), marker=dict(size=8, color=c["gru"]),
                                 hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>"))
    fig.update_xaxes(title_text="voltas à frente (h)", dtick=1, range=[0.5, 11.8])
    fig.update_yaxes(title_text="cobertura (%)", range=[50, 100])
    return base_layout(fig, 320, theme)


def training_curve(history: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Perda pinball de treino e validação por época, com a melhor época marcada.

    Com a coluna `seed` (histórico de um ensemble), desenha uma curva por semente e marca a melhor
    época de cada uma.
    """
    c = THEMES[theme]
    fig = go.Figure()
    if "seed" in history:
        for i, (seed, h) in enumerate(history.groupby("seed")):
            for col, name, color in (("train_loss", "treino", c["ink2"]), ("val_loss", "validação", c["gru"])):
                fig.add_trace(go.Scatter(x=h["epoch"], y=h[col], name=name, legendgroup=name, showlegend=i == 0,
                                         mode="lines", line=dict(color=color, width=1.5), opacity=0.75,
                                         hovertemplate=f"semente {seed} · {name}: %{{y:.4f}}<extra></extra>"))
        best = history.loc[history.groupby("seed")["val_loss"].idxmin()]
        fig.add_trace(go.Scatter(x=best["epoch"], y=best["val_loss"], name="melhor época", mode="markers",
                                 marker=dict(symbol="diamond", size=9, color=c["gru"], line=dict(color=c["surface"], width=1)),
                                 customdata=best["seed"],
                                 hovertemplate="semente %{customdata}: melhor época %{x}<extra></extra>"))
        fig.update_xaxes(title_text="época")
        fig.update_yaxes(title_text="perda pinball (padronizada)")
        return base_layout(fig, 320, theme)
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], name="treino", mode="lines+markers",
                             line=dict(color=c["ink2"], width=2), marker=dict(size=6, color=c["ink2"]),
                             hovertemplate="treino: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"], name="validação", mode="lines+markers",
                             line=dict(color=c["gru"], width=2), marker=dict(size=6, color=c["gru"]),
                             hovertemplate="validação: %{y:.4f}<extra></extra>"))
    best = history.loc[history["val_loss"].idxmin()]
    fig.add_vline(x=best["epoch"], line=dict(color=c["muted"], width=1, dash="dot"))
    fig.add_annotation(x=best["epoch"], y=1, yref="paper", text=f"melhor época ({int(best['epoch'])})", showarrow=False,
                       yanchor="bottom", font=dict(color=c["muted"], size=11))
    fig.update_xaxes(title_text="época")
    fig.update_yaxes(title_text="perda pinball (padronizada)")
    return base_layout(fig, 320, theme)


def grouped_bars(categories: list[str], values: dict[str, list[float]], ytitle: str, theme: str = "dark") -> go.Figure:
    """Barras lado a lado por categoria, uma cor por modelo (GRU × ingênuo)."""
    fig = go.Figure()
    for model, ys in values.items():
        name, color, _ = model_style(model, theme)
        fig.add_trace(go.Bar(x=categories, y=ys, name=name, marker=dict(color=color, cornerradius=4),
                             text=[f"{y:.2f}" for y in ys], textposition="outside",
                             hovertemplate=f"{name}: %{{y:.3f}} s<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(title_text=ytitle, rangemode="tozero")
    fig = base_layout(fig, 320, theme)
    fig.update_layout(hovermode="closest")
    return fig


def stint_forecasts(laps: pd.DataFrame, preds: dict[str, pd.DataFrame], drivers: list[str], anchors: list[int],
                    titles: list[str], theme: str = "light") -> go.Figure:
    """Previsões em blocos de 10 voltas para alguns pilotos de uma corrida.

    Para cada âncora t (ex.: 10, 20, 30…), mostra a previsão feita ao fim da volta t para t+1..t+10:
    a mediana e a faixa p10–p90 da GRU e o ingênuo, contra os tempos reais. Paradas marcadas.
    """
    c = THEMES[theme]
    fig = make_subplots(rows=len(drivers), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.07)
    for a in fig.layout.annotations:
        a.font = dict(color=c["ink"], size=13)
    for i, drv in enumerate(drivers, start=1):
        real = laps[laps["driver"] == drv].sort_values("lap")
        clean = real[real["clean"]]
        lo, hi = clean["lap_time"].quantile(0.01), clean["lap_time"].quantile(0.99)
        for t in anchors:
            for model, p in preds.items():
                w = p[(p["driver"] == drv) & (p["t"] == t)].sort_values("lap").copy()
                if w.empty:
                    continue
                # voltas não limpas (box, SC) não têm tempo comparável: a previsão some nelas
                w.loc[~w["clean"], ["time_p10", "time_p50", "time_p90"]] = None
                name, color, dash = model_style(model, theme)
                first = (i == 1 and t == anchors[0])
                if model == "gru":
                    # uma faixa por trecho contínuo de voltas limpas, para não atravessar as voltas de box
                    seg_id = (w["time_p50"].isna() != w["time_p50"].isna().shift()).cumsum()
                    for k, (_, seg) in enumerate(w[w["time_p50"].notna()].groupby(seg_id)):
                        fig.add_trace(go.Scatter(x=list(seg["lap"]) + list(seg["lap"])[::-1],
                                                 y=list(seg["time_p90"]) + list(seg["time_p10"])[::-1],
                                                 mode="lines", fill="toself", fillcolor=c["band"], line=dict(width=0),
                                                 hoverinfo="skip", name="GRU p10–p90", legendgroup="band",
                                                 showlegend=first and k == 0), row=i, col=1)
                    lo, hi = min(lo, w["time_p10"].min()), max(hi, w["time_p90"].max())
                fig.add_trace(go.Scatter(x=w["lap"], y=w["time_p50"], mode="lines", name=name, legendgroup=model,
                                         showlegend=first, line=dict(color=color, width=2, dash=dash),
                                         hovertemplate=f"{name}: %{{y|{LAP_FORMAT}}}<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=clean["lap"], y=clean["lap_time"], mode="markers", name="Real (volta limpa)",
                                 legendgroup="real", showlegend=i == 1,
                                 marker=dict(size=6, color=c["ink"], line=dict(width=1, color=c["surface"])),
                                 hovertemplate=f"Real: %{{y|{LAP_FORMAT}}}<extra></extra>"), row=i, col=1)
        add_pits_and_sc(fig, real, theme, row=i)
        pad = 0.25 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad], title_text="tempo", row=i, col=1)
    fig.update_xaxes(title_text="volta", row=len(drivers), col=1)
    whole_laps(fig)
    lap_time_axis(fig)
    return base_layout(fig, 290 * len(drivers) + 80, theme)


LAP_FORMAT = "%-M:%S.%L"                                  # 1:14.430


def lap_time_axis(fig: go.Figure) -> None:
    """Eixo y em min:seg.mil. As curvas chegam em segundos e viram instantes de um dia qualquer, no eixo de data."""
    def clock(seconds):
        ms = (pd.to_numeric(pd.Series(seconds), errors="coerce") * 1000).round()
        return pd.Timestamp("2000-01-01") + pd.to_timedelta(ms, unit="ms")

    for trace in fig.data:
        trace.y = clock(trace.y)

    def to_clock(axis):
        ends = None if axis.range is None or pd.isna(list(axis.range)).any() else [t.isoformat() for t in clock(axis.range)]
        axis.update(type="date", tickformat=LAP_FORMAT, hoverformat=LAP_FORMAT, automargin=True, range=ends)
    fig.for_each_yaxis(to_clock)


def whole_laps(fig: go.Figure) -> None:
    """Eixo x de voltas só com inteiros: uma marca por volta; os rótulos rareiam para não se sobrepor."""
    laps = [x for trace in fig.data if trace.x is not None for x in trace.x]
    span = max(laps) - min(laps) if laps else 0
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, ticklabelstep=1 if span <= 30 else 2 if span <= 60 else 5)


def add_pits_and_sc(fig: go.Figure, real: pd.DataFrame, theme: str, row: int = 1) -> None:
    """Linhas pontilhadas nas voltas de entrada nos boxes e faixas nas voltas de SC/VSC."""
    c = THEMES[theme]
    for lap in real.loc[real["pit_in"].fillna(False).astype(bool), "lap"]:
        fig.add_vline(x=lap, line=dict(color=c["muted"], width=1, dash="dot"), row=row, col=1)
        fig.add_annotation(x=lap, y=0, yref=f"y{row if row > 1 else ''} domain", text="box", showarrow=False,
                           xanchor="left", xshift=2, yanchor="bottom", font=dict(color=c["muted"], size=10), row=row, col=1)
    if "neutralised" in real:
        for lap in real.loc[real["neutralised"].fillna(False).astype(bool), "lap"]:
            fig.add_vrect(x0=lap - 0.5, x1=lap + 0.5, fillcolor=c["shade"], line_width=0, row=row, col=1)


def lap_times(laps: pd.DataFrame, drivers: list[str], team_colors: dict[str, str], theme: str = "dark") -> go.Figure:
    """Tempos de volta ao vivo: uma linha por piloto (cor da equipe), pontos na cor do composto."""
    c = THEMES[theme]
    fig = go.Figure()
    shown_teams, seen_compounds = set(), set()
    timed = laps[laps["lap_time"].notna() & (laps["driver"].isin(drivers))]
    if timed.empty:
        return base_layout(fig, 380, theme)
    lo, hi = timed["lap_time"].quantile(0.02), timed["lap_time"].median() * 1.07
    for drv in drivers:
        d = timed[timed["driver"] == drv].sort_values("lap")
        if d.empty:
            continue
        team = d["team"].iloc[-1]
        dash = "dot" if team in shown_teams else "solid"            # companheiros de equipe: linha pontilhada
        shown_teams.add(team)
        color = team_colors.get(drv, c["ink2"])
        fig.add_trace(go.Scatter(x=d["lap"], y=d["lap_time"], mode="lines", name=drv, line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=f"{drv}: %{{y|{LAP_FORMAT}}}<extra></extra>"))
        for comp, dc in d.groupby("compound"):
            fig.add_trace(go.Scatter(x=dc["lap"], y=dc["lap_time"], mode="markers", name=COMPOUND_LETTER.get(comp, comp),
                                     legendgroup=f"comp_{comp}", showlegend=comp not in seen_compounds, hoverinfo="skip",
                                     marker=dict(size=8, color=compound_color(comp, theme), line=dict(width=1.5, color=c["surface"]))))
            seen_compounds.add(comp)
    fig.update_xaxes(title_text="volta")
    whole_laps(fig)
    fig.update_yaxes(title_text="tempo de volta", range=[lo - 0.3, hi])
    lap_time_axis(fig)
    return base_layout(fig, 400, theme)


DRIVER_STYLE = [("#3987e5", "solid", "circle"), ("#199e70", "dash", "diamond")]   # validadas no fundo escuro


def rgba(color: str, alpha: float) -> str:
    return f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})"


def driver_forecasts(series: list[dict], xtitle: str, theme: str = "dark") -> go.Figure:
    """Previsto × real de até dois pilotos no mesmo eixo.

    Cada item de `series`: label, x, real (tempo real ou NaN), p10, p50, p90 e hover (texto por ponto).
    O segundo piloto usa linha tracejada e marcador em losango, além da cor.
    """
    c = THEMES[theme]
    fig = go.Figure()
    lo, hi = [], []
    for i, s in enumerate(series):
        color, dash, symbol = DRIVER_STYLE[i]
        df = pd.DataFrame({k: s[k] for k in ("x", "real", "p10", "p50", "p90", "hover")}).sort_values("x")
        seg_id = (df["p50"].isna() != df["p50"].isna().shift()).cumsum()
        for k, (_, seg) in enumerate(df[df["p50"].notna()].groupby(seg_id)):
            fig.add_trace(go.Scatter(x=list(seg["x"]) + list(seg["x"])[::-1], y=list(seg["p90"]) + list(seg["p10"])[::-1],
                                     mode="lines", fill="toself",
                                     fillcolor=rgba(color, 0.16),
                                     line=dict(width=0), hoverinfo="skip", legendgroup=s["label"], showlegend=False))
        fig.add_trace(go.Scatter(x=df["x"], y=df["p50"], mode="lines", name=f"{s['label']} · previsto (p50)",
                                 legendgroup=s["label"], line=dict(color=color, width=2, dash=dash),
                                 customdata=df["hover"], hovertemplate=f"%{{customdata}}<br>previsto %{{y|{LAP_FORMAT}}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df["x"], y=df["real"], mode="markers", name=f"{s['label']} · real",
                                 legendgroup=s["label"], marker=dict(size=9, color=color, symbol=symbol,
                                                                     line=dict(width=1.5, color=c["ink"])),
                                 hovertemplate=f"real %{{y|{LAP_FORMAT}}}<extra></extra>"))
        lo += [df["p10"].min(), df["real"].min()]
        hi += [df["p90"].max(), df["real"].max()]
    lo, hi = pd.Series(lo).min(), pd.Series(hi).max()
    if pd.notna(lo) and pd.notna(hi):
        pad = 0.15 * (hi - lo) + 0.1
        fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_xaxes(title_text=xtitle)
    whole_laps(fig)
    fig.update_yaxes(title_text="tempo de volta")
    lap_time_axis(fig)
    return base_layout(fig, 420, theme)


def live_forecast(real: pd.DataFrame, pred: pd.DataFrame, title: str, theme: str = "dark") -> go.Figure:
    """Voltas reais e a previsão mais recente da GRU (mediana e faixa p10–p90) de um piloto, ou de dois para comparar."""
    c = THEMES[theme]
    fig = go.Figure()
    clean = real[real["lap_time"].notna()].sort_values("lap")
    drivers = list(dict.fromkeys(real["driver"]))
    compare = len(drivers) > 1
    for i, drv in enumerate(drivers):
        color, dash, symbol = DRIVER_STYLE[i] if compare else (c["gru"], "solid", "circle")
        who = f"{drv} · " if compare else ""
        p = pred[pred["driver"] == drv]
        r = clean[clean["driver"] == drv]
        if not p.empty:
            fig.add_trace(go.Scatter(x=list(p["lap"]) + list(p["lap"])[::-1],
                                     y=list(p["time_p90"]) + list(p["time_p10"])[::-1], mode="lines", fill="toself",
                                     fillcolor=rgba(color, 0.16) if compare else c["band"], line=dict(width=0),
                                     hoverinfo="skip", name="GRU p10–p90", legendgroup=who, showlegend=not compare))
            fig.add_trace(go.Scatter(x=p["lap"], y=p["time_p50"], mode="lines+markers",
                                     name=f"{who}GRU v0 (mediana)", legendgroup=who,
                                     line=dict(color=color, width=2, dash=dash),
                                     marker=dict(size=6, color=color, symbol=symbol),
                                     hovertemplate=f"{who}previsto: %{{y|{LAP_FORMAT}}}<extra></extra>"))
            fig.add_vline(x=p["t"].iloc[0] + 0.5, line=dict(color=c["muted"], width=1, dash="dot"))
        marker = (dict(size=8, color=color, symbol=symbol, line=dict(width=1.5, color=c["ink"])) if compare
                  else dict(size=7, color=c["ink"], line=dict(width=1, color=c["surface"])))
        fig.add_trace(go.Scatter(x=r["lap"], y=r["lap_time"], mode="markers", name=f"{who}real", legendgroup=who,
                                 marker=marker, hovertemplate=f"{who}real: %{{y|{LAP_FORMAT}}}<extra></extra>"))
    if len(clean):
        lo = min(clean["lap_time"].quantile(0.02), pred["time_p10"].min() if len(pred) else 1e9)
        hi = max(clean["lap_time"].median() * 1.05, pred["time_p90"].max() if len(pred) else 0)
        fig.update_yaxes(range=[lo - 0.3, hi + 0.3])
    fig.update_layout(title=dict(text=title, font=dict(color=c["ink"], size=14), x=0, xanchor="left"))
    fig.update_xaxes(title_text="volta")
    whole_laps(fig)
    fig.update_yaxes(title_text="tempo de volta")
    lap_time_axis(fig)
    return base_layout(fig, 380, theme, top=96)
