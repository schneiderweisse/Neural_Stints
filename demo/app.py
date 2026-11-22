"""Demo do painel do Neural Stints, para o GitHub Pages.

Roda no navegador com stlite (Streamlit em WebAssembly), sem servidor. Tem as mesmas abas do painel
completo (dashboard/), menos a parte ao vivo: as corridas gravadas, o Replay (corrida pela telemetria e
duelos previstos pela rede) e o ajuste do modelo (GRU v0 e Tarefas A e B). A leitura do f1-dash durante
as sessões e a aba Mapa, que só funciona ao vivo, ficam no painel completo.

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

from dashboard import layout, live_tab, model_tab, replay_tab, tasks_tab  # noqa: E402

REPO = "https://github.com/schneiderweisse/Neural_Stints"

st.set_page_config(page_title="Neural Stints · demo", page_icon="🏁", layout="wide")
layout.inject_css()
# marca em texto no lugar do logo da F1, que é marca registrada; o logo da UnB vai junto (assets/)
layout.header(layout.F1_MARK, layout.logo("unb_logo", 44, "Universidade de Brasília", HERE / "assets"))
st.markdown(f"<div class='demo-note'>Demonstração do painel, rodando inteiramente no navegador, com tudo o que o "
            "painel completo tem, menos a parte ao vivo (a leitura do feed durante as sessões e o mapa ao vivo). "
            "Em <b>Corridas gravadas</b>, os GPs de Las Vegas e de São Paulo de 2026, com as previsões que a rede fez ao "
            "vivo a cada volta, e o GP da Alemanha de 2019. Em <b>Replay</b>, a corrida de Las Vegas em time-lapse no mapa, "
            "pela telemetria, acompanhando um piloto, e os duelos que a rede previu, com o que aconteceu de fato. "
            "<b>GRU v0</b> e <b>Tarefas A e B</b> mostram o ajuste da rede nas corridas de validação e teste de 2026. "
            f"Relatório da corrida: <a href='{live_tab.RACE_REPORT}'>GP de São Paulo</a>. "
            f"Código e documentação: <a href='{REPO}'>github.com/schneiderweisse/Neural_Stints</a>."
            "</div>", unsafe_allow_html=True)

# só a seção aberta roda: no navegador, rodar todas a cada clique deixava a demo lenta
layout.keep_widget_state()
SECTIONS = {"Corridas gravadas": lambda: live_tab.render(demo=True), "Replay": replay_tab.render,
            "GRU v0": model_tab.render, "Tarefas A e B": tasks_tab.render}
SECTIONS[layout.section_nav(list(SECTIONS))]()

layout.footer()
