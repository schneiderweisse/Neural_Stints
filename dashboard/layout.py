"""Visual comum ao painel completo (dashboard/app.py) e à demo (demo/app.py): CSS, cabeçalho e rodapé."""
import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
AUTHOR = "Eduardo Schneider Arruda"
AUTHOR_ROLE = "engenheiro de computação e pesquisador, graduando de Estatística pela UnB"
TITLE = "Neural Stints"
SUBTITLE = "Previsão de ritmo e degradação de pneus na F1 com redes recorrentes"
F1_MARK = "<span class='f1-mark' aria-label='F1'>F1</span>"

# Toques do GitHub (Primer, tema escuro); o resto vem do tema do Streamlit (.streamlit/config.toml)
CSS = """
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
.timing-wrap {overflow-x: auto;}
table.timing td.sector {min-width: 78px; padding-top: 4px;}
/* troca de posição: a linha ganha destaque no lugar antigo (--d linhas abaixo ou acima), desliza até o
   lugar novo e o destaque se apaga. Duas fases iguais (a/b) para a animação recomeçar numa nova troca. */
table.timing tr.gain {--hl: rgba(63,185,80,.30); --edge: #3fb950;}
table.timing tr.loss {--hl: rgba(248,81,73,.26); --edge: #f85149;}
table.timing tr.mv-a > td, table.timing tr.mv-b > td {position: relative; z-index: 2;
    animation-duration: 1.5s; animation-timing-function: ease-in-out; animation-fill-mode: both;}
table.timing tr.mv-a > td {animation-name: swap-a;}
table.timing tr.mv-b > td {animation-name: swap-b;}
table.timing tr.loss > td {z-index: 1;}
@keyframes swap-a {
  0% {transform: translateY(calc(var(--d) * 100%)); background: transparent; box-shadow: none;}
  18%, 38% {transform: translateY(calc(var(--d) * 100%)); background: var(--hl); box-shadow: inset 0 1px 0 var(--edge), inset 0 -1px 0 var(--edge);}
  78% {transform: translateY(0); background: var(--hl); box-shadow: inset 0 1px 0 var(--edge), inset 0 -1px 0 var(--edge);}
  100% {transform: translateY(0); background: transparent; box-shadow: none;}
}
@keyframes swap-b {
  0% {transform: translateY(calc(var(--d) * 100%)); background: transparent; box-shadow: none;}
  18%, 38% {transform: translateY(calc(var(--d) * 100%)); background: var(--hl); box-shadow: inset 0 1px 0 var(--edge), inset 0 -1px 0 var(--edge);}
  78% {transform: translateY(0); background: var(--hl); box-shadow: inset 0 1px 0 var(--edge), inset 0 -1px 0 var(--edge);}
  100% {transform: translateY(0); background: transparent; box-shadow: none;}
}
/* duelos: um cartão por par, embaixo da cronometragem */
.duels {display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin: 4px 0 18px 0;}
.duel {border: 1px solid #30363d; border-radius: 6px; background: #161b22; padding: 9px 12px 10px;}
.duel-head {display: flex; align-items: center; gap: 6px; color: #e6edf3; font-size: 14px;}
.duel-head .dot {display: inline-block; width: 4px; height: 14px; border-radius: 2px;}
.duel-arrow {color: #7d8590;}
.duel-pos {margin-left: auto; color: #7d8590; font-size: 12px;}
.duel-pin {font-size: 11px; color: #a371f7; border: 1px solid #a371f7; border-radius: 2em; padding: 0 7px;}
.duel-meta {color: #7d8590; font-size: 12px; margin: 3px 0 4px;}
.duel-text {color: #e6edf3; font-size: 13px; margin-top: 2px;}
.duel-note {color: #d29922; font-size: 13px; margin-top: 6px;}
.duel-out {font-size: 12.5px; margin-top: 5px; font-weight: 500;}
.duel-out.hit {color: #3fb950;} .duel-out.miss {color: #f85149;} .duel-out.stop {color: #7d8590;}
.duel-now {color: #d29922; font-size: 13px; margin: 2px 0 8px;}
table.timing .mv-tag {margin-left: 6px; font-size: 11px; font-weight: 700; color: var(--edge);
    animation: 1.5s ease-in-out both;}
table.timing tr.mv-a .mv-tag {animation-name: tag-a;}
table.timing tr.mv-b .mv-tag {animation-name: tag-b;}
@keyframes tag-a {0% {opacity: 0;} 18%, 78% {opacity: 1;} 100% {opacity: 0;}}
@keyframes tag-b {0% {opacity: 0;} 18%, 78% {opacity: 1;} 100% {opacity: 0;}}
@media (prefers-reduced-motion: reduce) {
  table.timing tr.mv-a > td, table.timing tr.mv-b > td {animation: none; background: var(--hl);}
}
.minis {display: flex; gap: 2px; margin-bottom: 3px;}
.minis i {display: block; width: 8px; height: 4px; border-radius: 2px;}
.minis-legend {display: flex; flex-wrap: wrap; gap: 4px 14px; color: #7d8590; font-size: 12px; margin: 6px 0 2px 0;}
.minis-legend span {display: inline-flex; align-items: center; gap: 5px;}
.minis-legend i {display: inline-block; width: 10px; height: 5px; border-radius: 2px;}
/* atualização a cada poucos segundos sem o painel escurecer: o Streamlit deixa translúcido o que está sendo refeito */
[data-testid="stElementContainer"][data-stale="true"], [data-stale="true"] {opacity: 1 !important; transition: none !important;}
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
.live-badge.stale {color: #d29922; border: 1px solid #d29922; background: rgba(210,153,34,.10);}
.live-badge .pulse {width: 8px; height: 8px; border-radius: 50%; background: currentColor;}
.live-badge.on .pulse {animation: pulse 1.2s ease-in-out infinite;}
@keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: .25; transform: scale(.7);}}
@media (prefers-reduced-motion: reduce) {.live-badge.on .pulse {animation: none;}}
.app-footer {margin-top: 48px; padding-top: 14px; border-top: 1px solid #21262d; color: #7d8590; font-size: 13px;}
.app-footer b {color: #e6edf3; font-weight: 600;}
/* marca em texto, no lugar do logo da F1 (marca registrada), na demo pública */
.f1-mark {font-weight: 800; font-style: italic; font-size: 34px; letter-spacing: -1px; color: #e10600;
          line-height: 1; padding-right: 2px;}
.demo-note {border: 1px solid #30363d; border-left: 3px solid #2f81f7; border-radius: 6px; padding: 8px 12px;
             background: #161b22; color: #c9d1d9; font-size: 13px; margin: 4px 0 12px 0;}
</style>
"""


def logo(name: str, height: int, alt: str, assets: Path = ASSETS) -> str:
    """<img> embutido de <assets>/<name>.svg|png; vazio se o arquivo não existir (ver dashboard/assets/README.md)."""
    for ext, mime in (("svg", "image/svg+xml"), ("png", "image/png")):
        path = assets / f"{name}.{ext}"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f"<img src='data:{mime};base64,{data}' alt='{alt}' style='height:{height}px'>"
    return ""


def keep_widget_state() -> None:
    """Mantém o valor dos controles das seções fechadas.

    O Streamlit apaga o estado de um controle que não aparece numa execução. Reatribuir cada chave no
    começo da execução faz o valor sobreviver, o mesmo recurso que a documentação indica para apps de
    várias páginas. Chame antes de criar qualquer controle.
    """
    for key in list(st.session_state.keys()):
        try:
            st.session_state[key] = st.session_state[key]
        except Exception:                 # chaves que não aceitam valor pela Session State API
            pass


def section_nav(names: list[str], key: str = "section") -> str:
    """Seletor das seções do painel. Ao contrário de st.tabs, que roda o código de todas as abas a cada
    clique, só a seção escolhida roda."""
    return st.segmented_control("Seção", names, default=names[0], required=True, key=key,
                                label_visibility="collapsed") or names[0]


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(left: str, right: str) -> None:
    """Título do projeto entre a marca da esquerda (F1) e a da direita (UnB)."""
    st.markdown(f"<div class='app-header'>{left}<div class='names'><h2>{TITLE}</h2><p>{SUBTITLE}</p></div>{right}</div>",
                unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f"<div class='app-footer'><b>{AUTHOR}</b> · {AUTHOR_ROLE}<br>"
                "Projeto final de Redes Neurais 2 · Bacharelado em Estatística, Universidade de Brasília · "
                "projeto acadêmico e não oficial, sem vínculo com a Fórmula 1 ou a FIA</div>", unsafe_allow_html=True)
