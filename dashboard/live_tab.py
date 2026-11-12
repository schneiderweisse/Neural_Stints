"""Aba Ao Vivo: a sessão atual (f1-dash) ou uma sessão gravada."""
import html
import time

import pandas as pd
import streamlit as st

from dashboard import data
from live.feed import DEFAULT_URL, STALE_AFTER, heartbeat_age, heartbeat_time
from live.laps import as_list, session_key, sub
from modeling.figures import COMPOUND_LETTER, compound_color, lap_times, live_forecast

RACE_REPORT = "https://github.com/schneiderweisse/Neural_Stints/blob/main/docs/race_reports/2026_sao_paulo.md"

TRACK_STATUS = {"1": ("Pista verde", "#3fb950"), "2": ("Bandeira amarela", "#d29922"), "3": ("", "#7d8590"),
                "4": ("Safety car", "#d29922"), "5": ("Bandeira vermelha", "#f85149"), "6": ("VSC", "#d29922"),
                "7": ("Fim do VSC", "#d29922")}
PURPLE, GREEN, MUTED = "#a371f7", "#3fb950", "#7d8590"
# minissetores, com os códigos de status do feed (os mesmos que o f1-dash usa)
YELLOW, PIT, EMPTY = "#d29922", "#2f81f7", "#30363d"
SEGMENT_COLOR = {2048: YELLOW, 2052: YELLOW, 2049: GREEN, 2051: PURPLE, 2064: PIT}
SEGMENT_LEGEND = [(YELLOW, "mais lento que o melhor pessoal"), (GREEN, "melhor pessoal"), (PURPLE, "melhor da sessão"),
                  (PIT, "pit lane"), (EMPTY, "ainda não passou")]
# troca de posição: a linha fica em destaque no lugar antigo e depois desliza para o novo (CSS em layout.py)
SWAP_SECONDS = 1.6     # duração da transição; a linha mantém a marcação até ela terminar
REPLAY_MAX_STEP = 61   # na gravação, só anima entre momentos vizinhos (a 30 s); um salto do controle não anima


def timing_color(item) -> str:
    if not isinstance(item, dict):
        return ""
    if item.get("OverallFastest"):
        return PURPLE
    if item.get("PersonalFastest"):
        return GREEN
    return ""


def cell(value, color: str = "", mono: bool = True) -> str:
    style = f' style="color:{color}"' if color else ""
    cls = ' class="mono"' if mono else ""
    return f"<td{cls}{style}>{html.escape(str(value)) if value not in (None, '') else ''}</td>"


def sector_cell(sector) -> str:
    """Setor com os minissetores em cima, como no f1-dash.

    Com o setor em andamento (sem `Value`), mostra o tempo da volta anterior em cinza.
    """
    sector = sector if isinstance(sector, dict) else {}
    bars = "".join(f"<i style='background:{SEGMENT_COLOR.get(g.get('Status') if isinstance(g, dict) else None, EMPTY)}'></i>"
                   for g in as_list(sector.get("Segments")))
    value, color = sector.get("Value") or "", timing_color(sector)
    if not value:
        value, color = sector.get("PreviousValue") or "", MUTED
    style = f" style='color:{color}'" if color else ""
    return f"<td class='mono sector'><div class='minis'>{bars}</div><span{style}>{html.escape(str(value))}</span></td>"


def segment_legend() -> str:
    return "<div class='minis-legend'>Minissetores:" + "".join(
        f"<span><i style='background:{c}'></i>{html.escape(t)}</span>" for c, t in SEGMENT_LEGEND) + "</div>"


def timing_rows(state: dict) -> tuple[list[dict], bool]:
    """Uma linha por piloto, na ordem da cronometragem."""
    race = str(sub(state, "SessionInfo", "Type") or "").lower() in ("race", "sprint")
    lines = sub(state, "TimingData", "Lines") or {}
    rows = []
    for num, ln in lines.items():
        drv = sub(state, "DriverList", num) or {}
        stints = as_list(sub(state, "TimingAppData", "Lines", num, "Stints"))
        stint = stints[-1] if stints else {}
        sectors = as_list(ln.get("Sectors"))
        if ln.get("Retired"):
            status = "abandono"
        elif ln.get("KnockedOut"):
            status = "eliminado"
        elif ln.get("Stopped"):
            status = "parado"
        elif ln.get("InPit"):
            status = "box"
        elif ln.get("PitOut"):
            status = "saída do box"
        else:
            status = ""
        rows.append({
            "num": str(num), "line": int(ln.get("Line") or drv.get("Line") or 99), "pos": ln.get("Position", ""),
            "tla": drv.get("Tla", num), "team": drv.get("TeamName", ""), "color": f"#{drv.get('TeamColour', '7d8590')}",
            "best": sub(ln, "BestLapTime", "Value") or "", "last": ln.get("LastLapTime") or {},
            "gap": (ln.get("GapToLeader") if race else ln.get("TimeDiffToFastest")) or "",
            "interval": (sub(ln, "IntervalToPositionAhead", "Value") if race else ln.get("TimeDiffToPositionAhead")) or "",
            "sectors": sectors[:3], "compound": str(stint.get("Compound") or "").upper(),
            "tyre_age": stint.get("TotalLaps"), "new": str(stint.get("New", "")).lower() == "true",
            "laps": ln.get("NumberOfLaps", ""), "pits": ln.get("NumberOfPitStops", ""), "status": status,
        })
    return sorted(rows, key=lambda r: r["line"]), race


def position_moves(rows: list[dict], key: str, stamp: float | None) -> dict[str, tuple[int, str]]:
    """Pilotos que trocaram de posição desde a atualização anterior desta tela: {número: (posições, fase)}.

    A ordem anterior fica no session_state, por tela. `posições` é positivo para quem subiu. Uma troca
    continua marcada por SWAP_SECONDS, com a mesma fase, para a transição não ser cortada pela próxima
    atualização (ao vivo, a cada 1 s); uma nova troca do mesmo piloto troca a fase ("a"/"b"), o que
    reinicia a animação no navegador. Na gravação (`stamp` = relógio da gravação), um salto maior que
    REPLAY_MAX_STEP limpa tudo: o usuário arrastou o momento, e não há troca a mostrar.
    """
    order = [r["num"] for r in rows]
    now = time.monotonic()
    prev = st.session_state.get(key) or {}
    active = {n: m for n, m in (prev.get("active") or {}).items() if now - m[2] < SWAP_SECONDS}
    jumped = stamp is not None and prev.get("stamp") is not None and abs(stamp - prev["stamp"]) > REPLAY_MAX_STEP
    if jumped or not prev.get("order"):
        active = {}
    else:
        old = {n: i for i, n in enumerate(prev["order"])}
        for i, n in enumerate(order):
            if n in old and old[n] != i:
                phase = "b" if active.get(n, (0, "b"))[1] == "a" else "a"
                active[n] = (old[n] - i, phase, now)
    st.session_state[key] = {"order": order, "stamp": stamp, "active": active}
    return {n: (m[0], m[1]) for n, m in active.items()}


def timing_table(state: dict, moves: dict[str, tuple[int, str]] | None = None) -> str:
    rows, race = timing_rows(state)
    moves = moves or {}
    head = ["Pos", "Piloto", "Melhor volta", "Gap" if race else "Dif. melhor", "Intervalo" if race else "Dif. à frente",
            "Última volta", "S1", "S2", "S3", "Pneu", "Voltas", "Paradas", ""]
    out = ["<div class='timing-wrap'><table class='timing'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) +
           "</tr></thead><tbody>"]
    for r in rows:
        tyre = ""
        if r["compound"]:
            letter = COMPOUND_LETTER.get(r["compound"], r["compound"][:1])
            age = "" if r["tyre_age"] is None else f" {int(r['tyre_age'])}"
            tyre = (f"<span class='tyre' style='border-color:{compound_color(r['compound'], 'dark')};"
                    f"color:{compound_color(r['compound'], 'dark')}'>{letter}</span>{age}{'' if r['new'] else ' usado'}")
        last = r["last"] if isinstance(r["last"], dict) else {}
        sectors = list(r["sectors"]) + [{}] * (3 - len(r["sectors"]))
        status = f"<span class='badge'>{r['status']}</span>" if r["status"] else ""
        tr, pos = "<tr>", cell(r["pos"])
        if r["num"] in moves:
            delta, phase = moves[r["num"]]
            # --d: de quantas linhas a linha parte (positivo = vem de baixo, porque subiu)
            tr = f"<tr class='mv-{phase} {'gain' if delta > 0 else 'loss'}' style='--d:{delta}'>"
            pos = (f"<td class='mono'>{html.escape(str(r['pos']))}"
                   f"<span class='mv-tag'>{'▲' if delta > 0 else '▼'}{abs(delta)}</span></td>")
        out.append(
            tr + pos +
            f"<td><span class='dot' style='background:{r['color']}'></span><b>{html.escape(r['tla'])}</b>"
            f"<span class='team'>{html.escape(r['team'])}</span></td>" +
            cell(r["best"]) + cell(r["gap"]) + cell(r["interval"]) +
            cell(last.get("Value", ""), timing_color(last)) +
            "".join(sector_cell(s) for s in sectors[:3]) +
            f"<td class='mono'>{tyre}</td>" + cell(r["laps"]) + cell(r["pits"]) + f"<td>{status}</td></tr>")
    out.append("</tbody></table></div>")
    if any(isinstance(s, dict) and s.get("Segments") for r in rows for s in r["sectors"]):
        out.append(segment_legend())
    return "".join(out)


FINISHED = {"Finished", "Finalised", "Ends"}


def badge(state: dict, mode: str) -> str:
    """Selo ao lado do título: ponto vermelho piscando quando os dados chegam ao vivo do f1-dash."""
    if mode == "live":
        if str(sub(state, "SessionStatus", "Status") or "") in FINISHED:
            return "<span class='live-badge off'><span class='pulse'></span>SESSÃO ENCERRADA</span>"
        age = heartbeat_age(state)
        if age is not None and age > STALE_AFTER:            # o f1-dash parou de receber a F1: dados congelados
            return f"<span class='live-badge stale'><span class='pulse'></span>SEM DADOS HÁ {age / 60:.0f} MIN</span>"
        return "<span class='live-badge on'><span class='pulse'></span>AO VIVO</span>"
    label = "REPRODUZINDO GRAVAÇÃO" if mode == "playing" else "GRAVAÇÃO"
    return f"<span class='live-badge off'><span class='pulse'></span>{label}</span>"


def remaining(state: dict, live: bool) -> str:
    """Tempo restante da sessão.

    O feed só manda o relógio quando ele começa, para ou é corrigido: `Remaining` vale no instante `Utc`
    e, com `Extrapolating`, o relógio está correndo desde então. Ao vivo, desconta o tempo passado desde
    `Utc`, como o f1-dash; na gravação, mostra o último valor enviado.
    """
    clock = state.get("ExtrapolatedClock") or {}
    rem = clock.get("Remaining")
    if not rem:
        return "—"
    if not (live and clock.get("Extrapolating") and clock.get("Utc")):
        return rem
    try:
        h, m, s = (float(x) for x in str(rem).split(":"))
        left = h * 3600 + m * 60 + s - (pd.Timestamp.now(tz="UTC") - pd.Timestamp(clock["Utc"])).total_seconds()
    except (ValueError, TypeError):
        return rem
    left = max(0, int(left))
    return f"{left // 3600:02d}:{left % 3600 // 60:02d}:{left % 60:02d}"


def session_header(state: dict, clock_label: str = "", mode: str = "live") -> str:
    info = state.get("SessionInfo") or {}
    title = f"{sub(info, 'Meeting', 'Name') or ''} · {info.get('Name', '')}"
    circuit = sub(info, "Meeting", "Circuit", "ShortName") or ""
    code = str(sub(state, "TrackStatus", "Status") or "")
    label, color = TRACK_STATUS.get(code, (sub(state, "TrackStatus", "Message") or "—", MUTED))
    lap = state.get("LapCount") or {}
    progress = (f"volta {lap.get('CurrentLap')} de {lap.get('TotalLaps')}" if lap.get("TotalLaps")
                else f"restam {remaining(state, mode == 'live')}")
    w = state.get("WeatherData") or {}
    rain = " · chuva" if str(w.get("Rainfall", "0")) not in ("0", "") else ""
    weather = f"ar {w.get('AirTemp', '—')} °C · pista {w.get('TrackTemp', '—')} °C · umidade {w.get('Humidity', '—')}%{rain}"
    status = str(sub(state, "SessionStatus", "Status") or "")
    extra = f" · {clock_label}" if clock_label else ""
    return (f"<div class='session'><div class='title'>{badge(state, mode)}{html.escape(title)}</div>"
            f"<div class='meta'>{html.escape(circuit)} · {html.escape(progress)} · {html.escape(weather)}"
            f"{(' · ' + html.escape(status)) if status else ''}{html.escape(extra)}</div>"
            f"<span class='flag' style='border-color:{color};color:{color}'>{html.escape(label)}</span></div>")


def team_colors(state: dict) -> dict[str, str]:
    return {d.get("Tla"): f"#{d.get('TeamColour', '7d8590')}" for d in (state.get("DriverList") or {}).values()
            if isinstance(d, dict)}


def render_body(state: dict, session_dir, clock: float | None, clock_label: str = "", mode: str = "live",
                demo: bool = False) -> None:
    st.markdown(session_header(state, clock_label, mode), unsafe_allow_html=True)
    rows, _ = timing_rows(state)
    moves = position_moves(rows, f"order_{mode == 'live'}_{session_dir.name if session_dir else 'live'}", clock)
    st.markdown(timing_table(state, moves), unsafe_allow_html=True)

    laps = data.read_parquet_fresh(session_dir / "laps.parquet") if session_dir else None
    if laps is not None and clock is not None:
        # lidas ao vivo, as voltas guardam o horário real; refeitas da gravação, o tempo desde o início dela
        live_clock = laps["clock"].max() > 1e8
        laps = laps[laps["clock"] <= (heartbeat_time(state) or float("inf") if live_clock else clock)]
    st.subheader("Tempos de volta")
    if laps is None or laps.empty:
        st.caption("Ainda não há voltas completadas gravadas pelo live.reader nesta sessão.")
        return
    order = [r["tla"] for r in timing_rows(state)[0]]
    have = set(laps["driver"])
    options = [d for d in order if d in have]
    picked = st.multiselect("Pilotos", options, default=options[:5], key=f"drivers_{session_dir.name}")
    st.plotly_chart(lap_times(laps, picked, team_colors(state)), width="stretch", config={"displaylogo": False})

    preds = data.read_parquet_fresh(session_dir / "predictions.parquet")
    if preds is None or preds.empty:
        if demo:
            st.caption("Nesta corrida não há previsões da GRU: 2019 é anterior aos dados de treino (2022 a 2026). "
                       "O ajuste do modelo está nas abas GRU v0 e Tarefas A e B; ao vivo, no painel completo, a "
                       "previsão das próximas 10 voltas aparece aqui a cada volta.")
        elif str(sub(state, "SessionInfo", "Type") or "").lower() in ("race", "sprint"):
            st.caption("Previsões da GRU: rode `python -m live.predict` para vê-las aqui.")
        return
    st.subheader("Previsão da GRU v0 (cenário A: mesmo pneu, sem parada)")
    if "gru_run" in preds:
        run = str(preds["gru_run"].iloc[-1])
        ens = data.run_files(run)["config"].get("ensemble") if (data.RUNS / run / "config.json").exists() else None
        st.caption(f"Modelo `{run}`" + (f": média dos quantis de {len(ens['members'])} sementes." if ens else "."))
    if demo:
        st.caption("Previsões feitas ao vivo durante a corrida, volta a volta. Com pneu de chuva (começo e fim da corrida) "
                   "a rede está fora do que aprendeu, porque foi treinada sem corridas molhadas, e erra cerca de 10 s. "
                   f"Análise completa no [relatório da corrida]({RACE_REPORT}).")
    have = set(preds["driver"])
    options = [d for d in order if d in have]
    picked = st.multiselect("Pilotos (escolha dois para comparar)", options, default=options[:1], max_selections=2,
                            key=f"pred_{session_dir.name}")
    latest, anchors = [], {}
    for drv in picked:
        current = int(laps.loc[laps["driver"] == drv, "lap"].max()) if (laps["driver"] == drv).any() else 0
        p = preds[(preds["driver"] == drv) & (preds["t"] <= current)]
        if len(p):
            latest.append(p[p["t"] == p["t"].max()].sort_values("h"))
            anchors[drv] = int(p["t"].max())
    if picked:
        if len(picked) == 1:
            title = f"{picked[0]}: previsão feita ao fim da volta {anchors[picked[0]]}" if anchors else picked[0]
        else:
            title = "Previsão feita ao fim da volta: " + " · ".join(f"{d} {t}" for d, t in anchors.items())
        chart = live_forecast(laps[laps["driver"].isin(picked)], pd.concat(latest) if latest else preds.iloc[:0], title)
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
    # só as voltas previstas que já foram disputadas até este momento (na gravação, o resto é "futuro")
    done = preds[preds["time_real"].notna() & preds["clean"]].merge(laps[["driver", "lap"]], on=["driver", "lap"])
    if len(done):
        mae = (done["time_p50"] - done["time_real"]).abs().mean()
        cov = ((done["time_real"] >= done["time_p10"]) & (done["time_real"] <= done["time_p90"])).mean()
        st.caption(f"Erro ao vivo até agora: MAE {mae:.3f} s em {len(done):,} voltas previstas já disputadas · ".replace(",", ".") +
                   f"cobertura do intervalo p10–p90: {cov:.0%}")


def render(demo: bool = False) -> None:
    """Aba Ao Vivo. Na demo (demo=True), só a reprodução de sessões gravadas, sem o f1-dash."""
    c1, c2, c3 = st.columns([2, 3, 2])
    if demo:
        source = "Sessão gravada"
        c1.markdown("<div class='demo-note'>Reprodução de uma corrida gravada: arraste o momento ou aperte "
                    "<b>Reproduzir</b>.</div>", unsafe_allow_html=True)
    else:
        source = c1.radio("Fonte", ["f1-dash ao vivo", "Sessão gravada"], horizontal=True, key="live_source")
    if source == "f1-dash ao vivo":
        url = c2.text_input("Endereço do f1-dash", DEFAULT_URL, key="live_url")
        interval = c3.slider("Atualizar a cada (s)", 1, 15, 1, key="live_every")   # cada atualização leva ~70 ms

        @st.fragment(run_every=interval)
        def live_view():
            state = data.fetch_live_state(url)
            if state is None:
                st.info("O f1-dash não tem sessão no momento (ou não está no ar). Suba o `start-live.sh` e, durante a "
                        "sessão, rode `python -m live.reader` para gravar as voltas.")
                return
            key = session_key(state)
            render_body(state, data.LIVE / key if key else None, None)

        live_view()
        return

    recs = data.recordings()
    if not recs:
        st.info("Nenhuma sessão gravada em data/live ainda.")
        return
    rec = c2.selectbox("Gravação", recs, key="rec_path",
                       format_func=lambda p: p.parent.name if p.name == data.REPLAY_FILE else f"{p.parent.name} · {p.name}")
    snaps = data.recording_snapshots(str(rec))
    if "rec_idx" not in st.session_state or st.session_state.get("rec_for") != str(rec):
        st.session_state.rec_idx, st.session_state.rec_for = len(snaps) // 2, str(rec)
    play = c3.toggle("Reproduzir", key="rec_play")

    @st.fragment(run_every=1.5 if play else None)
    def recorded_view():
        if play:
            st.session_state.rec_idx = min(st.session_state.rec_idx + 1, len(snaps) - 1)
        idx = st.select_slider("Momento da gravação", options=list(range(len(snaps))), key="rec_idx",
                               format_func=lambda i: f"{snaps[i][0] / 60:.1f} min")
        clock, state = snaps[idx]
        render_body(state, rec.parent, clock, f"gravação {clock / 60:.0f} min", "playing" if play else "recorded", demo)

    recorded_view()
