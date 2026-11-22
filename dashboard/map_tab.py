"""Aba Mapa: a pista com a posição estimada de cada carro, ao vivo pelo f1-dash.

O feed da F1 que chega pelo f1-dash não traz mais as coordenadas dos carros (Position.z). A posição é
estimada pelos minissetores da cronometragem, como no mapa do próprio f1-dash: o carro está depois do
último minissetor que já fechou nesta volta. Os minissetores de cada setor são tratados como trechos de
mesmo tempo, e a fração da volta vira um ponto do traçado pelo tempo de cada ponto numa volta de
referência (trackPositionTime). Entre uma atualização e outra, o carro avança no ritmo de uma volta
típica, sem passar do fim do próximo minissetor.

O traçado vem da API do MultiViewer (a mesma que o f1-dash usa) e fica guardado em data/live/maps/.
"""
import html
import json
import time
import urllib.request

import numpy as np
import streamlit as st

from dashboard import data
from dashboard.live_tab import TRACK_STATUS, session_header, timing_rows
from live.feed import DEFAULT_URL
from live.laps import as_list, sub

MAPS = data.LIVE / "maps"
MAP_URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
W = 1000                       # largura do desenho, em unidades do SVG
PAD = 60
TRACK, TRACK_EDGE, INK, MUTED, BG = "#6e7681", "#21262d", "#e6edf3", "#7d8590", "#0d1117"
ALERT = {"2": "#d29922", "4": "#d29922", "6": "#d29922", "7": "#d29922", "5": "#f85149"}


def track_map(key: int, year: int) -> dict | None:
    """Traçado do circuito (x, y, tempo de cada ponto, curvas, rotação), baixado uma vez e guardado."""
    path = MAPS / f"{key}_{year}.json"
    if not path.exists():
        try:
            req = urllib.request.Request(MAP_URL.format(key=key, year=year),
                                         headers={"User-Agent": "neural-stints-dashboard"})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read()
            m = json.loads(raw)
            if not (isinstance(m.get("x"), list) and isinstance(m.get("y"), list) and m.get("trackPositionTime")):
                return None
            MAPS.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except (OSError, ValueError):
            return None
    return json.loads(path.read_text(encoding="utf-8"))


def rotate(x, y, deg: float, cx: float, cy: float):
    a = np.radians(deg)
    x, y = np.asarray(x, float) - cx, np.asarray(y, float) - cy
    return x * np.cos(a) - y * np.sin(a), y * np.cos(a) + x * np.sin(a)


def geometry(m: dict) -> dict:
    """Pontos girados como no f1-dash e levados para o quadro do SVG (largura W, y para baixo)."""
    x, y = np.array(m["x"], float), np.array(m["y"], float)
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    deg = float(m.get("rotation", 0)) + 90
    xr, yr = rotate(x, y, deg, cx, cy)
    if np.ptp(yr) > np.ptp(xr):            # deitado, para caber na largura do painel
        deg += 90
        xr, yr = rotate(x, y, deg, cx, cy)
    scale = (W - 2 * PAD) / (xr.max() - xr.min())
    h = (yr.max() - yr.min()) * scale + 2 * PAD
    to_svg = lambda a, b: ((a - xr.min()) * scale + PAD, (b - yr.min()) * scale + PAD)
    px, py = to_svg(xr, yr)
    t = np.array(m["trackPositionTime"], float)
    frac = (t - t[0]) / (t[-1] - t[0])
    corners = []
    for c in m.get("corners") or []:
        pos = c.get("trackPosition") or {}
        ang = np.radians(float(c.get("angle", 0)))
        lx, ly = pos.get("x", 0) + 420 * np.cos(ang), pos.get("y", 0) + 420 * np.sin(ang)
        a, b = to_svg(*rotate([lx], [ly], deg, cx, cy))
        corners.append((int(c.get("number", 0)), float(a[0]), float(b[0])))
    # a mesma transformação como matriz (x_svg = a·x + b·y + e, y_svg = c·x + d·y + f), para o replay
    co, si = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    affine = [scale * co, -scale * si, scale * si, scale * co,
              scale * (-cx * co + cy * si - xr.min()) + PAD, scale * (-cy * co - cx * si - yr.min()) + PAD]
    return {"x": px, "y": py, "frac": frac, "h": float(h), "corners": corners, "affine": [float(v) for v in affine]}


def point_at(g: dict, f: float) -> tuple[float, float]:
    """Ponto do traçado na fração f do tempo de volta."""
    f = min(max(f, 0.0), 1.0)
    return float(np.interp(f, g["frac"], g["x"])), float(np.interp(f, g["frac"], g["y"]))


def seconds(value) -> float | None:
    try:
        v = str(value)
        m, s = v.split(":") if ":" in v else ("0", v)
        return int(m) * 60 + float(s)
    except ValueError:
        return None


def sector_bounds(lines: dict) -> list[float]:
    """Frações da volta em que terminam os setores, pela mediana dos últimos tempos de setor do pelotão."""
    times = [[], [], []]
    for ln in lines.values():
        for k, sec in enumerate(as_list(ln.get("Sectors"))[:3]):
            v = seconds(sec.get("Value") if isinstance(sec, dict) else None)
            if v and v > 5:
                times[k].append(v)
    med = [float(np.median(t)) if t else None for t in times]
    if any(v is None for v in med):
        return [0.0, 1 / 3, 2 / 3, 1.0]
    total = sum(med)
    return [0.0, med[0] / total, (med[0] + med[1]) / total, 1.0]


def lap_fraction(ln: dict, bounds: list[float]) -> tuple[int, float, float]:
    """(último minissetor fechado, fração da volta no fim dele, fração no fim do próximo)."""
    ends = []
    status = []
    for k, sec in enumerate(as_list(ln.get("Sectors"))[:3]):
        segs = as_list(sec.get("Segments")) if isinstance(sec, dict) else []
        n = max(len(segs), 1)
        for j, seg in enumerate(segs):
            ends.append(bounds[k] + (j + 1) / n * (bounds[k + 1] - bounds[k]))
            status.append(int((seg or {}).get("Status") or 0) if isinstance(seg, dict) else 0)
    done = [i for i, s in enumerate(status) if s > 0]
    last = done[-1] if done else -1
    here = ends[last] if last >= 0 else 0.0
    nxt = ends[last + 1] if last + 1 < len(ends) else 1.0
    return last, here, nxt


def car_positions(state: dict, g: dict) -> list[dict]:
    """Posição de cada carro na pista, com o avanço entre atualizações guardado por sessão do navegador."""
    rows, _ = timing_rows(state)
    lines = sub(state, "TimingData", "Lines") or {}
    bounds = sector_bounds(lines)
    laps = [seconds(sub(ln, "LastLapTime", "Value")) for ln in lines.values()]
    laps = [v for v in laps if v and v > 30]
    typical = float(np.median(laps)) if laps else 100.0
    seen = st.session_state.setdefault("map_seen", {})
    now = time.monotonic()
    cars, leader_f = [], None
    for r in rows:
        ln = lines.get(r["num"]) or {}
        if r["status"] in ("abandono", "parado", "box"):
            cars.append({**r, "on_track": False})
            continue
        last, here, nxt = lap_fraction(ln, bounds)
        mark = (ln.get("NumberOfLaps"), last)
        if seen.get(r["num"], (None, 0))[0] != mark:
            seen[r["num"]] = (mark, now)
        elapsed = now - seen[r["num"]][1]
        top = here + 0.97 * (nxt - here)
        f = min(here + elapsed / typical, top)
        # dentro do minissetor, o gap para o líder espalha os carros que fecharam o mesmo trecho
        gap = seconds(str(ln.get("GapToLeader") or "").lstrip("+"))
        if str(r["pos"]) == "1":
            leader_f = f
        elif leader_f is not None and gap is not None:
            est = (leader_f - gap / typical) % 1.0
            if here <= est <= top:
                f = est
            elif abs(est - here) < 0.5 and abs(est - top) < 0.5:
                f = min(max(est, here), top)
        x, y = point_at(g, f)
        cars.append({**r, "on_track": True, "x": x, "y": y})
    return cars


def map_svg(g: dict, cars: list[dict], track_code: str) -> str:
    h = g["h"]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(g["x"], g["y"]))
    color = ALERT.get(track_code, TRACK)
    parts = [f"<svg class='trackmap' viewBox='0 0 {W} {h:.0f}' width='100%' role='img' aria-label='mapa da pista'>",
             f"<polyline points='{pts}' fill='none' stroke='{TRACK_EDGE}' stroke-width='22' stroke-linejoin='round' stroke-linecap='round'/>",
             f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='9' stroke-linejoin='round' stroke-linecap='round'/>"]
    # linha de chegada: um traço perpendicular ao traçado no primeiro ponto
    (x0, y0), (x1, y1) = (g["x"][0], g["y"][0]), (g["x"][3], g["y"][3])
    d = np.hypot(x1 - x0, y1 - y0) or 1
    nx, ny = -(y1 - y0) / d * 16, (x1 - x0) / d * 16
    parts.append(f"<line x1='{x0 - nx:.1f}' y1='{y0 - ny:.1f}' x2='{x0 + nx:.1f}' y2='{y0 + ny:.1f}' stroke='{INK}' stroke-width='4'/>")
    for num, cx, cy in g["corners"]:
        parts.append(f"<text x='{cx:.1f}' y='{cy + 5:.1f}' fill='{MUTED}' font-size='15' text-anchor='middle'>{num}</text>")
    # rótulos: em grupo, cada um vai para o primeiro lugar livre (acima, abaixo, mais acima, mais abaixo)
    on = [c for c in cars if c["on_track"]]
    taken, label_at = [], {}
    for c in on:
        for dy in (-12, 30, -36, 54, -60, 78):
            lx, ly = c["x"] + 14, c["y"] + dy
            if all(abs(lx - tx) > 46 or abs(ly - ty) > 20 for tx, ty in taken):
                break
        taken.append((lx, ly))
        label_at[c["tla"]] = (lx, ly)
    # o líder por último, para ficar por cima
    for c in reversed(on):
        lead = str(c["pos"]) == "1"
        lx, ly = label_at[c["tla"]]
        parts.append(f"<g><circle cx='{c['x']:.1f}' cy='{c['y']:.1f}' r='{13 if lead else 11}' fill='{c['color']}' "
                     f"stroke='{INK if lead else BG}' stroke-width='{3 if lead else 2.5}'/>"
                     f"<text x='{lx:.1f}' y='{ly:.1f}' fill='{INK}' font-size='19' font-weight='700' "
                     f"paint-order='stroke' stroke='{BG}' stroke-width='5'>{html.escape(c['tla'])}</text></g>")
    parts.append("</svg>")
    return "".join(parts)


def order_list(cars: list[dict]) -> str:
    out = ["<div class='map-order'>"]
    for c in cars:
        gap = c["interval"] if str(c["pos"]) != "1" else "líder"
        tag = "" if c["on_track"] else f"<span class='badge'>{html.escape(c['status'])}</span>"
        out.append(f"<div class='map-row'><span class='mono pos'>{html.escape(str(c['pos']))}</span>"
                   f"<span class='dot' style='background:{c['color']}'></span><b>{html.escape(c['tla'])}</b>"
                   f"<span class='mono gap'>{html.escape(str(gap))}</span>{tag}</div>")
    out.append("</div>")
    return "".join(out)


def render() -> None:
    url = st.session_state.get("live_url", DEFAULT_URL)

    @st.fragment(run_every=1)
    def view():
        state = data.fetch_live_state(url)
        if state is None:
            st.info("O f1-dash não tem sessão no momento (ou não está no ar). O mapa aparece durante uma sessão.")
            return
        info = state.get("SessionInfo") or {}
        key = sub(info, "Meeting", "Circuit", "Key")
        year = int(str(info.get("StartDate", "0"))[:4] or 0)
        m = track_map(int(key), year) if key and year else None
        if m is None:
            st.warning("Não foi possível obter o traçado deste circuito (api.multiviewer.app).")
            return
        g = geometry(m)
        cars = car_positions(state, g)
        st.markdown(session_header(state), unsafe_allow_html=True)
        code = str(sub(state, "TrackStatus", "Status") or "")
        c1, c2 = st.columns([4, 1])
        c1.markdown(map_svg(g, cars, code), unsafe_allow_html=True)
        c2.markdown(order_list(cars), unsafe_allow_html=True)
        label = TRACK_STATUS.get(code, ("", ""))[0]
        st.caption("Posições estimadas pelos minissetores da cronometragem (o feed não traz mais as coordenadas dos "
                   "carros): podem adiantar ou atrasar até um minissetor. Traçado: MultiViewer."
                   + (f" Pista: {label.lower()}." if label else ""))

    view()
