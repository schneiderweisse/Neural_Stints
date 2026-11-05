"""Figuras Plotly dos resultados, reaproveitáveis no relatório de backtest e no painel.

Cores com papel fixo (validadas para daltonismo): real em tinta neutra, GRU em azul, ingênuo em laranja.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GRU, NAIVE = "#2a78d6", "#eb6834"
BAND = "rgba(42,120,214,0.16)"
NEUTRAL_SHADE = "rgba(137,135,129,0.14)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MODEL_STYLE = {"gru": ("GRU v0", GRU, "solid"), "naive": ("Ingênuo", NAIVE, "dash")}


def fmt_lap(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=dict(l=64, r=24, t=84, b=48),
                      font=dict(family=FONT, color=INK2, size=12), hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1, yref="container", yanchor="top", font=dict(color=INK2)),
                      hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def mae_by_horizon(metrics: dict[str, dict], races: list[str], split: str = "test") -> go.Figure:
    """MAE (s) por horizonte, um painel por corrida, uma linha por modelo."""
    fig = make_subplots(rows=1, cols=len(races), shared_yaxes=True, subplot_titles=races, horizontal_spacing=0.06)
    for model, m in metrics.items():
        name, color, dash = MODEL_STYLE[model]
        for j, race in enumerate(races, start=1):
            by_h = m["races"][race]["by_h"]
            hs = sorted(by_h)
            ys = [by_h[h] for h in hs]
            fig.add_trace(go.Scatter(x=hs, y=ys, name=name, legendgroup=model, showlegend=j == 1, mode="lines+markers",
                                     line=dict(color=color, width=2, dash=dash), marker=dict(size=8, color=color),
                                     hovertemplate=f"{name}: %{{y:.2f}} s<extra></extra>"), row=1, col=j)
            fig.add_annotation(x=hs[-1], y=ys[-1], text=name, xanchor="left", xshift=8, showarrow=False,
                               font=dict(color=INK2, size=11), row=1, col=j)
    fig.update_xaxes(title_text="voltas à frente (h)", dtick=1, range=[0.5, 12.5])
    fig.update_yaxes(title_text="MAE (s)", row=1, col=1, rangemode="tozero")
    for a in fig.layout.annotations[: len(races)]:
        a.font = dict(color=INK, size=13)
    return base_layout(fig, 360)


def stint_forecasts(laps: pd.DataFrame, preds: dict[str, pd.DataFrame], drivers: list[str], anchors: list[int],
                    titles: list[str]) -> go.Figure:
    """Previsões em blocos de 10 voltas para alguns pilotos de uma corrida.

    Para cada âncora t (ex.: 10, 20, 30…), mostra a previsão feita ao fim da volta t para t+1..t+10:
    a mediana e a faixa p10–p90 da GRU e o ingênuo, contra os tempos reais. Paradas marcadas.
    """
    fig = make_subplots(rows=len(drivers), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.07)
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
                name, color, dash = MODEL_STYLE[model]
                first = (i == 1 and t == anchors[0])
                if model == "gru":
                    # uma faixa por trecho contínuo de voltas limpas, para não atravessar as voltas de box
                    seg_id = (w["time_p50"].isna() != w["time_p50"].isna().shift()).cumsum()
                    for k, (_, seg) in enumerate(w[w["time_p50"].notna()].groupby(seg_id)):
                        fig.add_trace(go.Scatter(x=list(seg["lap"]) + list(seg["lap"])[::-1],
                                                 y=list(seg["time_p90"]) + list(seg["time_p10"])[::-1],
                                                 mode="lines", fill="toself",
                                                 fillcolor=BAND, line=dict(width=0), hoverinfo="skip",
                                                 name="GRU p10–p90", legendgroup="band",
                                                 showlegend=first and k == 0), row=i, col=1)
                    lo, hi = min(lo, w["time_p10"].min()), max(hi, w["time_p90"].max())
                fig.add_trace(go.Scatter(x=w["lap"], y=w["time_p50"], mode="lines", name=name, legendgroup=model,
                                         showlegend=first, line=dict(color=color, width=2, dash=dash),
                                         hovertemplate=f"{name}: %{{y:.2f}} s<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=clean["lap"], y=clean["lap_time"], mode="markers", name="Real (volta limpa)",
                                 legendgroup="real", showlegend=i == 1,
                                 marker=dict(size=6, color=INK, line=dict(width=1, color=SURFACE)),
                                 hovertemplate="Real: %{y:.3f} s<extra></extra>"), row=i, col=1)
        for lap in real.loc[real["pit_in"], "lap"]:
            fig.add_vline(x=lap, line=dict(color=MUTED, width=1, dash="dot"), row=i, col=1)
            fig.add_annotation(x=lap, y=0, yref=f"y{i if i > 1 else ''} domain", text="box", showarrow=False,
                               xanchor="left", xshift=2, yanchor="bottom", font=dict(color=MUTED, size=10),
                               row=i, col=1)
        for lap in real.loc[real["neutralised"], "lap"]:
            fig.add_vrect(x0=lap - 0.5, x1=lap + 0.5, fillcolor=NEUTRAL_SHADE, line_width=0, row=i, col=1)
        pad = 0.25 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad], title_text="tempo (s)", row=i, col=1)
    fig.update_xaxes(title_text="volta", row=len(drivers), col=1)
    for a in fig.layout.annotations[: len(drivers)]:
        a.font = dict(color=INK, size=13)
    return base_layout(fig, 290 * len(drivers) + 80)
