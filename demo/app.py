"""Demo do painel do Neural Stints, para o GitHub Pages.

Roda no navegador com stlite (Streamlit em WebAssembly), sem servidor. Mostra a reprodução de uma
corrida gravada e o ajuste do modelo (abas GRU v0 e Tarefas A e B), com as mesmas abas do painel
completo (dashboard/). A parte ao vivo, que lê o f1-dash durante as sessões, fica só no painel completo.

No site, este arquivo vira demo_app.py na raiz de app.zip, ao lado de dashboard/, live/, modeling/ e
dos dados de demo/files (ver demo/build.py). Para testar fora do navegador, a partir de Neural_Stints:
    .venv/Scripts/python demo/build.py site
    e abra demo/_site/index.html por um servidor local (ver demo/README.md)
"""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dashboard import layout, live_tab, model_tab, tasks_tab  # noqa: E402

REPO = "https://github.com/schneiderweisse/Neural_Stints"

st.set_page_config(page_title="Neural Stints · demo", page_icon="🏁", layout="wide")
layout.inject_css()
# marca em texto no lugar do logo da F1, que é marca registrada; o logo da UnB vai junto (assets/)
layout.header(layout.F1_MARK, layout.logo("unb_logo", 44, "Universidade de Brasília", HERE / "assets"))
st.markdown(f"<div class='demo-note'>Demonstração do painel, rodando inteiramente no navegador. "
            "Aqui estão a reprodução do GP de São Paulo de 2026, com as previsões que a GRU fez ao vivo a cada volta, e do "
            "GP da Alemanha de 2019, além do ajuste da rede nas corridas de validação e teste de 2026. O painel completo "
            "também acompanha as sessões ao vivo. Relatório da corrida: "
            f"<a href='{live_tab.RACE_REPORT}'>GP de São Paulo</a>. "
            f"Código e documentação: <a href='{REPO}'>github.com/schneiderweisse/Neural_Stints</a>."
            "</div>", unsafe_allow_html=True)

replay, gru, tasks = st.tabs(["Corridas gravadas", "GRU v0", "Tarefas A e B"])
with replay:
    live_tab.render(demo=True)
with gru:
    model_tab.render()
with tasks:
    tasks_tab.render()

layout.footer()
