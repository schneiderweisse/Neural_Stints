"""Duelos, embaixo da cronometragem: pares de pilotos, o gap projetado pela GRU e quando a disputa começa.

Os pares vêm de dois lugares: os duelos fixos escolhidos no painel (sempre um par de pilotos) e os
automáticos, os pares vizinhos a até 5 s com a maior chance de disputa nas próximas 10 voltas. Disputa é
o carro de trás chegar a menos de 1 s do da frente na linha de chegada. A conta está em modeling/duels.py
e foi validada em modeling/duels_backtest.py.

Na gravação, o painel também mostra o que aconteceu depois daquele momento: a volta em que a disputa
começou de fato e se ela caiu dentro da janela prevista.
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

from modeling import duels
from modeling.figures import fmt_lap

H = 10
AUTO_PAIRS = 3
MAX_GAP = 5.0
WET = {"INTERMEDIATE", "WET"}
DEMO_PAIRS = ["NOR × VER"]         # o estudo de caso do relatório de São Paulo
Z80 = 1.2816                      # meia largura da faixa de 80% em desvios-padrão
INK, MUTED, GRID, BAND, LINE, HIT, MISS = "#e6edf3", "#7d8590", "#30363d", "rgba(57,135,229,.22)", "#3987e5", "#3fb950", "#f85149"


def parse_gap(value) -> float | None:
    """'+1.234' -> 1.234; voltas de diferença ('1 L') e textos ('LAP 45') -> None."""
    try:
        return abs(float(str(value).strip().lstrip("+")))
    except ValueError:
        return None


def forecasts(preds: pd.DataFrame, laps_now: pd.DataFrame) -> dict[str, tuple[int, np.ndarray]]:
    """Previsão mais recente de cada piloto até a volta que ele já completou: {piloto: (t, matriz H×3)}."""
    current = laps_now.groupby("driver")["lap"].max()
    out = {}
    for drv, g in preds[preds["driver"].isin(current.index)].groupby("driver"):
        g = g[g["t"] <= current[drv]]
        if g.empty:
            continue
        t = int(g["t"].max())
        g = g[g["t"] == t]
        m = np.full((H, 3), np.nan)
        m[g["h"].to_numpy(int) - 1] = g[["time_p10", "time_p50", "time_p90"]].to_numpy(float)
        out[drv] = (t, m)
    return out


def recent_pace(laps_now: pd.DataFrame, drv: str) -> float | None:
    """Média das 3 últimas voltas com tempo, fora das voltas de box."""
    d = laps_now[(laps_now["driver"] == drv) & laps_now["lap_time"].notna()
                 & ~laps_now["pit_in"].fillna(False).astype(bool) & ~laps_now["pit_out"].fillna(False).astype(bool)]
    d = d.sort_values("lap").tail(3)
    return float(d["lap_time"].mean()) if len(d) else None


def outcome(laps_all: pd.DataFrame, behind: str, ahead: str, t: int, gap0: float) -> dict:
    """O que aconteceu nas H voltas depois de t: a volta da disputa ou o motivo de a conta parar."""
    r = laps_all.set_index(["driver", "lap"])
    status = r["track_status"].fillna("").astype(str) if "track_status" in r else pd.Series("", index=r.index)
    path, gap = [gap0], gap0
    for k in range(1, H + 1):
        kb, ka = (behind, t + k), (ahead, t + k)
        if kb not in r.index or ka not in r.index:
            return {"lap": None, "path": path, "stop": "a gravação ainda não chegou lá" if k == 1 else "fim dos dados"}
        for key in (kb, ka):
            if bool(r.at[key, "pit_in"]) or bool(r.at[key, "pit_out"]):
                return {"lap": None, "path": path, "stop": f"parada de {key[0]} na volta {t + k}"}
            if any(c in status[key] for c in "4567"):
                return {"lap": None, "path": path, "stop": f"safety car ou VSC na volta {t + k}"}
            if str(r.at[key, "compound"]).upper() in WET:
                return {"lap": None, "path": path, "stop": f"pneu de chuva na volta {t + k}"}
        if pd.isna(r.at[kb, "lap_time"]) or pd.isna(r.at[ka, "lap_time"]):
            return {"lap": None, "path": path, "stop": f"volta {t + k} sem tempo"}
        gap += float(r.at[kb, "lap_time"]) - float(r.at[ka, "lap_time"])
        path.append(gap)
        if gap <= duels.BATTLE_GAP:
            return {"lap": t + k, "k": k, "path": path, "stop": None}
    return {"lap": None, "path": path, "stop": None}


def sparkline(gap0: float, proj: dict, window: dict, real: list[float] | None = None, w: int = 300, h: int = 92) -> str:
    """Gap projetado (linha e faixa de 80%), a zona de ataque (1 s), a janela prevista e, na gravação, o real."""
    mu, sd = np.concatenate([[gap0], proj["mu"]]), np.concatenate([[0.0], proj["sd"]])
    hi, lo = mu + Z80 * sd, mu - Z80 * sd
    real = real or []
    top = max(float(np.max(hi)), max(real, default=0), gap0, 1.5) * 1.06
    # abaixo de zero o carro de trás já passou e a projeção perde sentido (os dois passam a se influenciar)
    bottom = max(min(0.0, float(np.min(lo)), min(real, default=0)), -2.0)
    pl, pr, pt, pb = 30, 8, 6, 18
    n = len(mu) - 1
    x = lambda k: pl + (w - pl - pr) * k / max(n, 1)
    y = lambda v: pt + (h - pt - pb) * (top - v) / (top - bottom)
    pts = lambda arr: " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in enumerate(arr))
    band = pts(hi) + " " + " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in reversed(list(enumerate(lo))))
    clip = f"c{abs(hash((gap0, float(mu[-1])))) % 10**8}"
    parts = [f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' role='img' aria-label='gap projetado'>",
             f"<defs><clipPath id='{clip}'><rect x='{pl}' y='{pt}' width='{w - pl - pr}' height='{h - pt - pb}'/></clipPath></defs>"]
    if window.get("early"):
        x0 = x(window["early"] - 0.5)
        x1 = x(min(window["late"] if not window["late_open"] else n, n) + 0.5)
        parts.append(f"<rect x='{x0:.1f}' y='{pt}' width='{max(x1 - x0, 2):.1f}' height='{h - pt - pb}' fill='rgba(210,153,34,.12)'/>")
    parts.append(f"<line x1='{pl}' x2='{w - pr}' y1='{y(duels.BATTLE_GAP):.1f}' y2='{y(duels.BATTLE_GAP):.1f}' "
                 f"stroke='#d29922' stroke-dasharray='4 3' stroke-width='1'/>")
    parts.append(f"<text x='{pl - 4}' y='{y(duels.BATTLE_GAP) + 3:.1f}' fill='#d29922' font-size='9' text-anchor='end'>1 s</text>")
    if abs(y(gap0) - y(duels.BATTLE_GAP)) >= 10:
        parts.append(f"<text x='{pl - 4}' y='{y(gap0) + 3:.1f}' fill='{MUTED}' font-size='9' text-anchor='end'>{gap0:.1f}</text>")
    parts.append(f"<g clip-path='url(#{clip})'><polygon points='{band}' fill='{BAND}' stroke='none'/>")
    parts.append(f"<polyline points='{pts(mu)}' fill='none' stroke='{LINE}' stroke-width='2'/>")
    if len(real) > 1:
        parts.append(f"<polyline points='{pts(real)}' fill='none' stroke='{INK}' stroke-width='1.2' stroke-dasharray='2 2'/>")
        parts += [f"<circle cx='{x(k):.1f}' cy='{y(v):.1f}' r='2.4' fill='{INK}'/>" for k, v in enumerate(real) if k]
    parts.append("</g>")
    for k, label in ((0, "agora"), (5, "+5"), (n, f"+{n}")):
        if k <= n:
            parts.append(f"<text x='{x(k):.1f}' y='{h - 4}' fill='{MUTED}' font-size='9' text-anchor='middle'>{label}</text>")
    parts.append("</svg>")
    return "".join(parts)


def card(pair: dict, show_outcome: bool) -> str:
    b, a = pair["behind"], pair["ahead"]
    head = (f"<div class='duel-head'><span class='dot' style='background:{b['color']}'></span><b>{html.escape(b['tla'])}</b>"
            f"<span class='duel-arrow'>→</span><span class='dot' style='background:{a['color']}'></span><b>{html.escape(a['tla'])}</b>"
            f"<span class='duel-pos'>P{b['pos']} → P{a['pos']}</span>{'<span class=duel-pin>fixo</span>' if pair['pinned'] else ''}</div>")
    pb, pa = pair["pace_b"], pair["pace_a"]
    if pb is not None and pa is not None:
        diff = pa - pb
        faster = b["tla"] if diff > 0 else a["tla"]
        pace = (f"ritmo nas 3 últimas voltas: {html.escape(b['tla'])} {fmt_lap(pb)} · {html.escape(a['tla'])} {fmt_lap(pa)}"
                + (f" → {html.escape(faster)} {abs(diff):.2f} s por volta mais rápido" if abs(diff) >= 0.005 else " → mesmo ritmo"))
    else:
        pace = "ritmo: poucas voltas ainda"
    body = f"<div class='duel-meta'>gap {pair['gap0']:.3f} s · {pace}</div>"
    if pair.get("note"):
        return f"<div class='duel'>{head}{body}<div class='duel-note'>{html.escape(pair['note'])}</div></div>"
    proj, w, t = pair["proj"], pair["window"], pair["t"]
    text = duels.describe(pair["gap0"], proj, t)
    real, result = None, ""
    if show_outcome and pair.get("outcome"):
        o = pair["outcome"]
        real = o["path"]
        if o["lap"] is not None:
            inside = w["early"] is not None and w["early"] <= o["k"] and (w["late_open"] or o["k"] <= w["late"])
            result = (f"<div class='duel-out {'hit' if inside else 'miss'}'>{'✔' if inside else '✘'} A disputa começou na volta {o['lap']}"
                      f"{' dentro da janela' if inside else ' fora da janela'}</div>")
        elif o["stop"]:
            result = f"<div class='duel-out stop'>Conta interrompida: {html.escape(o['stop'])}</div>"
        else:
            hit = w["likely"] is None
            result = (f"<div class='duel-out {'hit' if hit else 'miss'}'>{'✔' if hit else '✘'} Não houve disputa nas {H} voltas seguintes</div>")
    return (f"<div class='duel'>{head}{body}{sparkline(pair['gap0'], proj, w, real)}"
            f"<div class='duel-text'>{html.escape(text)}</div>"
            f"{result}</div>")


def render(rows: list[dict], laps_now: pd.DataFrame, laps_all: pd.DataFrame, preds: pd.DataFrame, key: str,
           recorded: bool, demo: bool) -> None:
    st.subheader("Duelos")
    fc = forecasts(preds, laps_now)
    by_tla = {r["tla"]: r for r in rows}
    order = [r["tla"] for r in rows]
    names = sorted(t for t in order if t in fc)
    options = [f"{x} × {y}" for i, x in enumerate(names) for y in names[i + 1:]]
    default = [p for p in DEMO_PAIRS if p in options] if demo else []
    c1, c2 = st.columns([3, 1])
    pinned = c1.multiselect("Duelos fixos (sempre um par de pilotos)", options, default=default, max_selections=3,
                            key=f"duels_{key}")
    show = c2.toggle("Mostrar o que aconteceu", value=True, key=f"duels_out_{key}") if recorded else False

    def build(behind: str, ahead: str, gap0: float, pinned_pair: bool) -> dict:
        b, a = by_tla[behind], by_tla[ahead]
        pair = {"behind": b, "ahead": a, "gap0": gap0, "pinned": pinned_pair,
                "pace_b": recent_pace(laps_now, behind), "pace_a": recent_pace(laps_now, ahead)}
        if b["compound"] in WET or a["compound"] in WET:
            pair["note"] = "Fora do domínio da rede: pneu de chuva."
        elif b["status"] in ("box", "saída do box") or a["status"] in ("box", "saída do box"):
            pair["note"] = "Projeção suspensa: um dos dois está no box."
        elif behind not in fc or ahead not in fc:
            pair["note"] = "Sem previsão da GRU para um dos dois ainda."
        else:
            t, mb = fc[behind]
            _, ma = fc[ahead]
            pair["t"] = t
            pair["proj"] = duels.project(gap0, mb, ma)
            pair["window"] = duels.catch_window(pair["proj"]["caught"])
            if not len(pair["proj"]["caught"]):
                pair["note"] = "Sem previsão para as próximas voltas (fim da corrida)."
            elif recorded:
                pair["outcome"] = outcome(laps_all, behind, ahead, t, gap0)
        return pair

    pairs, seen = [], set()
    for label in pinned:
        x, y = label.split(" × ")
        if x not in by_tla or y not in by_tla:
            continue
        ahead, behind = (x, y) if order.index(x) < order.index(y) else (y, x)
        ga, gb = parse_gap(by_tla[ahead]["gap"]) if order.index(ahead) else 0.0, parse_gap(by_tla[behind]["gap"])
        if ga is None or gb is None:
            continue                                        # retardatário (voltas de diferença): sem conta de gap
        pairs.append(build(behind, ahead, gb - ga, True))
        seen.add((behind, ahead))
    battles, candidates = [], []
    for i in range(1, len(rows)):
        behind, ahead = rows[i]["tla"], rows[i - 1]["tla"]
        gap0 = parse_gap(rows[i]["interval"])
        if gap0 is None or (behind, ahead) in seen:
            continue
        if gap0 <= duels.BATTLE_GAP:
            battles.append(f"{behind} → {ahead} ({gap0:.1f} s)")
        elif gap0 <= MAX_GAP:
            pair = build(behind, ahead, gap0, False)
            if "proj" in pair and "note" not in pair:
                candidates.append(pair)
    candidates.sort(key=lambda p: -p["window"]["p_horizon"])
    pairs += [p for p in candidates[:AUTO_PAIRS] if p["window"]["p_horizon"] >= 0.05]

    st.caption("Disputa: o carro de trás a menos de 1 s do da frente na linha. A GRU projeta o gap somando as "
               "diferenças de tempo previstas para as próximas 10 voltas; a faixa e a janela (em amarelo) são de 80%. "
               "Supõe que ninguém para. Nas corridas de teste, 82% das disputas previstas aconteceram, 86% delas "
               "a até 1 volta da prevista.")
    if battles:
        st.markdown("<div class='duel-now'>Em disputa agora: " + " · ".join(html.escape(b) for b in battles) + "</div>",
                    unsafe_allow_html=True)
    if not pairs:
        st.caption("Nenhum par a até 5 s com chance de disputa nas próximas voltas.")
        return
    st.markdown("<div class='duels'>" + "".join(card(p, show) for p in pairs) + "</div>", unsafe_allow_html=True)
