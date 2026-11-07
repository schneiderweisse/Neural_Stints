"""Painel do Neural Stints.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/streamlit run dashboard/app.py

A demo pública (sem a parte ao vivo) fica em demo/ e usa as mesmas abas.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import layout, live_tab, model_tab, tasks_tab  # noqa: E402

st.set_page_config(page_title="Neural Stints", page_icon="🏁", layout="wide")
layout.inject_css()
# logos oficiais só no painel local: os arquivos ficam fora do git (ver dashboard/assets/README.md)
layout.header(layout.logo("f1_logo", 30, "Fórmula 1"), layout.logo("unb_logo", 44, "Universidade de Brasília"))

live, gru, tasks = st.tabs(["Ao Vivo", "GRU v0", "Tarefas A e B"])
with live:
    live_tab.render()
with gru:
    model_tab.render()
with tasks:
    tasks_tab.render()

layout.footer()
