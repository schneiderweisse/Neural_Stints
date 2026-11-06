"""Painel do Neural Stints.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import live_tab, model_tab, tasks_tab  # noqa: E402

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
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin-bottom:0'>Neural Stints</h2>"
            "<p style='color:#7d8590;margin-top:2px'>Previsão de ritmo e degradação de pneus na F1 com redes recorrentes</p>",
            unsafe_allow_html=True)

live, gru, tasks = st.tabs(["Ao Vivo", "GRU v0", "Tarefas A e B"])
with live:
    live_tab.render()
with gru:
    model_tab.render()
with tasks:
    tasks_tab.render()
