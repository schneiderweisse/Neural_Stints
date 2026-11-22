"""Aba Replay: a corrida em time-lapse no mapa, acompanhando um piloto, mais lenta nos duelos.

Os dados vêm do pipeline/race_replay.py (telemetria do TracingInsights: posição real de cada carro).
O player (dashboard/replay_player.html) roda no navegador: escolhe o piloto, acha os duelos dele
(menos de 1 s para o carro da frente ou de trás, com bandeira verde) e anima a corrida. Cada volta leva
2 s; nos duelos, 6 s e com zoom no par, até o gap passar do corte.

No modo "Duelos previstos pela rede" (modeling/duel_replay.py, no arquivo <corrida>_duels.json), o usuário
escolhe um duelo que a rede previu e assiste ao par a partir da volta da previsão, com o gap previsto (e a
janela) contra o gap real num gráfico embaixo do mapa.
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from dashboard.map_tab import W, geometry
from modeling.figures import compound_color

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data" / "processed" / "replay"
PLAYER = Path(__file__).resolve().parent / "replay_player.html"
DEFAULTS = {"sec_per_lap": 2, "sec_per_lap_duel": 6, "duel_gap": 1, "zoom": True,
            "merge": 12.0, "min_duel": 1.5, "zoom_width": 300}   # o duelo acaba com o gap acima do corte por 12 s


def svg_points(affine: list[float], pts) -> np.ndarray:
    a, b, c, d, e, f = affine
    p = np.asarray(pts, float).reshape(-1, 2)
    return np.c_[a * p[:, 0] + b * p[:, 1] + e, c * p[:, 0] + d * p[:, 1] + f]


@st.cache_data(show_spinner="Montando o replay…")
def player_html(path: str, mtime: float, player_mtime: float, duels_mtime: float) -> str:
    text = Path(path).read_text(encoding="utf-8")
    rep = json.loads(text)
    tr = rep["track"]
    m = {"x": tr["x"], "y": tr["y"], "rotation": tr["rotation"], "trackPositionTime": tr["time"],
         "corners": [{"number": n, "angle": a, "trackPosition": {"x": x, "y": y}} for n, x, y, a in tr["corners"]]}
    g = geometry(m)
    aff = g["affine"]
    (x0, y0), (x1, y1) = svg_points(aff, tr["finish"])
    k = 16 / (np.hypot(x1 - x0, y1 - y0) or 1)
    nx, ny = -(y1 - y0) * k, (x1 - x0) * k
    config = {**DEFAULTS, "w": W, "h": round(g["h"], 1), "affine": aff,
              "track": np.round(np.c_[g["x"], g["y"]], 1).tolist(),
              "corners": [[n, round(x, 1), round(y, 1)] for n, x, y in g["corners"]],
              "finish": [[round(x0 - nx, 1), round(y0 - ny, 1)], [round(x0 + nx, 1), round(y0 + ny, 1)]],
              "pit": np.round(svg_points(aff, tr["pit"]), 1).tolist() if tr.get("pit") else [],
              "compounds": {c: compound_color(c, "dark") for c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")}}
    duels_path = duels_file(Path(path))
    duels = duels_path.read_text(encoding="utf-8") if duels_path.exists() else "null"
    html = PLAYER.read_text(encoding="utf-8")
    # "</" dentro de um <script> fecharia a tag
    return (html.replace("/*__DATA__*/null", text.replace("</", "<\\/"))
                .replace("/*__CONFIG__*/null", json.dumps(config).replace("</", "<\\/"))
                .replace("/*__DUELS__*/null", duels.replace("</", "<\\/")))


def duels_file(path: Path) -> Path:
    return path.with_name(path.stem + "_duels.json")


def label(path: Path) -> str:
    meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
    return f"{meta['event']} {meta['year']}"


def render() -> None:
    files = sorted((f for f in REPLAY.glob("*.json") if not f.stem.endswith("_duels")), reverse=True)
    if not files:
        st.info("Nenhum replay montado ainda. Com a telemetria da corrida baixada, rode "
                "`python -m pipeline.race_replay 2026 \"Las Vegas Grand Prix\"`.")
        return
    path = files[0] if len(files) == 1 else st.selectbox("Corrida", files, format_func=label)
    stats = json.loads(path.read_text(encoding="utf-8"))["stats"]
    st.markdown(f"**{label(path)}** · o líder mudou {stats['lead_changes_at_line']} vezes de uma volta para outra, na "
                f"linha · o segundo colocado ficou a menos de 1 s do líder em {stats['p1_fight_share']:.0%} do tempo com "
                f"bandeira verde")
    dp = duels_file(path)
    if dp.exists():
        dj = json.loads(dp.read_text(encoding="utf-8"))
        se, so, b = dj["summary_episodes"], dj["summary_ongoing_origins"], dj["brier_ongoing"]
        st.markdown(f"Duelos previstos pela rede: {se['n']}, dos quais {se['aconteceram']} aconteceram "
                    f"({se['precisao']:.0%} dos que tiveram desfecho; {se['na_janela']:.0%} deles dentro da janela de 80%). "
                    f"Nos duelos já em curso, a rede acertou se o de trás passava em {so['acerto']:.0%} das voltas, "
                    f"contra {b['taxa_de_ultrapassagem']:.0%} de ultrapassagens: aí ela não ajuda. "
                    f"Veja cada um no modo **Duelos previstos pela rede**.")
    components.html(player_html(str(path), path.stat().st_mtime, PLAYER.stat().st_mtime,
                                dp.stat().st_mtime if dp.exists() else 0.0), height=1000, scrolling=True)
    st.caption("Telemetria do TracingInsights. Cada carro fica no traçado na fração da volta que ele já percorreu (no pit "
               "lane, no x/y real). O intervalo é o tempo que o carro de trás leva até onde o da frente está: na linha de "
               "chegada, ele fica a 0,03 s (mediana) do gap oficial; no meio da volta, a ~0,1 s, então carros a menos de "
               "~0,3 s podem aparecer trocados por alguns segundos (a ordem bate com a oficial nos fins de setor em ~93% "
               "dos pares a menos de 0,5 s). Duelo: menos que o corte "
               "para o carro da frente ou de trás, com bandeira verde ou amarela local, fora do box. Sob SC, VSC e bandeira "
               "vermelha, os carros andam juntos por regra, então não conta. Espaço toca e pausa; as setas avançam e "
               "voltam uma volta.")
