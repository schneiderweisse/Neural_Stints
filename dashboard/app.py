"""Painel do Neural Stints.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/streamlit run dashboard/app.py
"""
import base64
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import live_tab, model_tab, tasks_tab  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
AUTHOR = "Eduardo Schneider Arruda"
AUTHOR_ROLE = "engenheiro de computação e pesquisador, graduando de Estatística pela UnB"


def logo(name: str, height: int, alt: str) -> str:
    """<img> embutido de dashboard/assets/<name>.svg|png; vazio se o arquivo não existir (ver assets/README.md)."""
    for ext, mime in (("svg", "image/svg+xml"), ("png", "image/png")):
        path = ASSETS / f"{name}.{ext}"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f"<img src='data:{mime};base64,{data}' alt='{alt}' style='height:{height}px'>"
    return ""


st.set_page_config(page_title="Neural Stints", page_icon="🏁", layout="wide")

# Toques do GitHub (Primer, tema escuro); o resto vem de .streamlit/config.toml
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
h1, h2, h3 {font-weight: 600 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #f78166;}
.mono {font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px;}
.session {border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; background: #161b22;
          position: relative;}
.session .title {font-size: 18px; font-weight: 600; color: #e6edf3;}
.session .meta {font-size: 13px; color: #7d8590; margin-top: 2px;}
.session .flag {position: absolute; right: 14px; top: 12px; border: 1px solid; border-radius: 2em; padding: 1px 10px;
                font-size: 12px; font-weight: 500;}
table.timing {width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid #30363d; border-radius: 6px;}
table.timing th {background: #161b22; color: #7d8590; font-weight: 500; text-align: left; padding: 6px 8px;
                 border-bottom: 1px solid #30363d;}
table.timing td {padding: 5px 8px; border-bottom: 1px solid #21262d; color: #e6edf3; white-space: nowrap;}
table.timing tr:hover td {background: #161b22;}
table.timing .dot {display: inline-block; width: 4px; height: 14px; border-radius: 2px; margin-right: 8px;
                   vertical-align: -2px;}
table.timing .team {color: #7d8590; margin-left: 8px; font-size: 12px;}
table.timing .tyre {display: inline-block; border: 2px solid; border-radius: 50%; width: 20px; height: 20px;
                    line-height: 16px; text-align: center; font-weight: 700; font-size: 11px;}
table.timing .badge {border: 1px solid #30363d; border-radius: 2em; padding: 0 8px; font-size: 11px; color: #d29922;}
.tile {border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; background: #161b22; height: 100%;}
.tile .tl {color: #7d8590; font-size: 12px;}
.tile .tv {font-size: 22px; font-weight: 600; color: #e6edf3; margin-top: 2px;}
.tile .tn {color: #7d8590; font-size: 12px; margin-top: 2px;}
.app-header {display: flex; align-items: center; gap: 18px; padding-bottom: 14px; margin-bottom: 6px;
             border-bottom: 1px solid #21262d;}
.app-header .names {flex: 1; min-width: 0;}
.app-header h2 {margin: 0; padding: 0;}
.app-header p {color: #7d8590; margin: 2px 0 0 0;}
.live-badge {display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
             letter-spacing: .04em; border-radius: 2em; padding: 1px 10px; margin-right: 10px; vertical-align: 3px;}
.live-badge.on {color: #f85149; border: 1px solid #f85149; background: rgba(248,81,73,.10);}
.live-badge.off {color: #7d8590; border: 1px solid #30363d;}
.live-badge .pulse {width: 8px; height: 8px; border-radius: 50%; background: currentColor;}
.live-badge.on .pulse {animation: pulse 1.2s ease-in-out infinite;}
@keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: .25; transform: scale(.7);}}
@media (prefers-reduced-motion: reduce) {.live-badge.on .pulse {animation: none;}}
.app-footer {margin-top: 48px; padding-top: 14px; border-top: 1px solid #21262d; color: #7d8590; font-size: 13px;}
.app-footer b {color: #e6edf3; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='app-header'>{logo('f1_logo', 30, 'Fórmula 1')}"
            "<div class='names'><h2>Neural Stints</h2>"
            "<p>Previsão de ritmo e degradação de pneus na F1 com redes recorrentes</p></div>"
            f"{logo('unb_logo', 44, 'Universidade de Brasília')}</div>", unsafe_allow_html=True)

live, gru, tasks = st.tabs(["Ao Vivo", "GRU v0", "Tarefas A e B"])
with live:
    live_tab.render()
with gru:
    model_tab.render()
with tasks:
    tasks_tab.render()

st.markdown(f"<div class='app-footer'><b>{AUTHOR}</b> · {AUTHOR_ROLE}<br>"
            "Projeto final de Redes Neurais 2 · Bacharelado em Estatística, Universidade de Brasília · "
            "projeto acadêmico e não oficial, sem vínculo com a Fórmula 1 ou a FIA</div>", unsafe_allow_html=True)
