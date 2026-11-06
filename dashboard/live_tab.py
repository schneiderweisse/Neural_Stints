"""Aba Ao Vivo: a sessão atual (f1-dash) ou uma sessão gravada."""
import html

import streamlit as st

from dashboard import data
from live.feed import DEFAULT_URL
from live.laps import as_list, session_key, sub
from modeling.figures import COMPOUND_LETTER, compound_color, lap_times, live_forecast

TRACK_STATUS = {"1": ("Pista verde", "#3fb950"), "2": ("Bandeira amarela", "#d29922"), "3": ("", "#7d8590"),
                "4": ("Safety car", "#d29922"), "5": ("Bandeira vermelha", "#f85149"), "6": ("VSC", "#d29922"),
                "7": ("Fim do VSC", "#d29922")}
PURPLE, GREEN, MUTED = "#a371f7", "#3fb950", "#7d8590"


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
            "line": int(ln.get("Line") or drv.get("Line") or 99), "pos": ln.get("Position", ""),
            "tla": drv.get("Tla", num), "team": drv.get("TeamName", ""), "color": f"#{drv.get('TeamColour', '7d8590')}",
            "best": sub(ln, "BestLapTime", "Value") or "", "last": ln.get("LastLapTime") or {},
            "gap": (ln.get("GapToLeader") if race else ln.get("TimeDiffToFastest")) or "",
            "interval": (sub(ln, "IntervalToPositionAhead", "Value") if race else ln.get("TimeDiffToPositionAhead")) or "",
            "sectors": sectors[:3], "compound": str(stint.get("Compound") or "").upper(),
            "tyre_age": stint.get("TotalLaps"), "new": str(stint.get("New", "")).lower() == "true",
            "laps": ln.get("NumberOfLaps", ""), "pits": ln.get("NumberOfPitStops", ""), "status": status,
        })
    return sorted(rows, key=lambda r: r["line"]), race


def timing_table(state: dict) -> str:
    rows, race = timing_rows(state)
    head = ["Pos", "Piloto", "Melhor volta", "Gap" if race else "Dif. melhor", "Intervalo" if race else "Dif. à frente",
            "Última volta", "S1", "S2", "S3", "Pneu", "Voltas", "Paradas", ""]
    out = ["<table class='timing'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        tyre = ""
        if r["compound"]:
            letter = COMPOUND_LETTER.get(r["compound"], r["compound"][:1])
            age = "" if r["tyre_age"] is None else f" {int(r['tyre_age'])}"
            tyre = (f"<span class='tyre' style='border-color:{compound_color(r['compound'], 'dark')};"
                    f"color:{compound_color(r['compound'], 'dark')}'>{letter}</span>{age}{'' if r['new'] else ' usado'}")
        last = r["last"] if isinstance(r["last"], dict) else {}
        sectors = [s if isinstance(s, dict) else {} for s in r["sectors"]] + [{}] * (3 - len(r["sectors"]))
        status = f"<span class='badge'>{r['status']}</span>" if r["status"] else ""
        out.append(
            "<tr>" + cell(r["pos"]) +
            f"<td><span class='dot' style='background:{r['color']}'></span><b>{html.escape(r['tla'])}</b>"
            f"<span class='team'>{html.escape(r['team'])}</span></td>" +
            cell(r["best"]) + cell(r["gap"]) + cell(r["interval"]) +
            cell(last.get("Value", ""), timing_color(last)) +
            "".join(cell(s.get("Value", ""), timing_color(s)) for s in sectors[:3]) +
            f"<td class='mono'>{tyre}</td>" + cell(r["laps"]) + cell(r["pits"]) + f"<td>{status}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


FINISHED = {"Finished", "Finalised", "Ends"}


def badge(state: dict, mode: str) -> str:
    """Selo ao lado do título: ponto vermelho piscando quando os dados chegam ao vivo do f1-dash."""
    if mode == "live":
        if str(sub(state, "SessionStatus", "Status") or "") in FINISHED:
            return "<span class='live-badge off'><span class='pulse'></span>SESSÃO ENCERRADA</span>"
        return "<span class='live-badge on'><span class='pulse'></span>AO VIVO</span>"
    label = "REPRODUZINDO GRAVAÇÃO" if mode == "playing" else "GRAVAÇÃO"
    return f"<span class='live-badge off'><span class='pulse'></span>{label}</span>"


def session_header(state: dict, clock_label: str = "", mode: str = "live") -> str:
    info = state.get("SessionInfo") or {}
    title = f"{sub(info, 'Meeting', 'Name') or ''} · {info.get('Name', '')}"
    circuit = sub(info, "Meeting", "Circuit", "ShortName") or ""
    code = str(sub(state, "TrackStatus", "Status") or "")
    label, color = TRACK_STATUS.get(code, (sub(state, "TrackStatus", "Message") or "—", MUTED))
    lap = state.get("LapCount") or {}
    progress = (f"volta {lap.get('CurrentLap')} de {lap.get('TotalLaps')}" if lap.get("TotalLaps")
                else f"restam {sub(state, 'ExtrapolatedClock', 'Remaining') or '—'}")
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


def render_body(state: dict, session_dir, clock: float | None, clock_label: str = "", mode: str = "live") -> None:
    st.markdown(session_header(state, clock_label, mode), unsafe_allow_html=True)
    st.markdown(timing_table(state), unsafe_allow_html=True)

    laps = data.read_parquet_fresh(session_dir / "laps.parquet") if session_dir else None
    if laps is not None and clock is not None:
        laps = laps[laps["clock"] <= clock]
    st.subheader("Tempos de volta")
    if laps is None or laps.empty:
        st.caption("Ainda não há voltas completadas gravadas pelo live.reader nesta sessão.")
        return
    order = [r["tla"] for r in timing_rows(state)[0]]
    options = [d for d in order if d in set(laps["driver"])]
    picked = st.multiselect("Pilotos", options, default=options[:5], key=f"drivers_{session_dir.name}")
    st.plotly_chart(lap_times(laps, picked, team_colors(state)), width="stretch", config={"displaylogo": False})

    preds = data.read_parquet_fresh(session_dir / "predictions.parquet")
    if preds is None or preds.empty:
        if str(sub(state, "SessionInfo", "Type") or "").lower() in ("race", "sprint"):
            st.caption("Previsões da GRU: rode `python -m live.predict` para vê-las aqui.")
        return
    st.subheader("Previsão da GRU v0 (cenário A: mesmo pneu, sem parada)")
    drv = st.selectbox("Piloto", [d for d in order if d in set(preds["driver"])], key=f"pred_{session_dir.name}")
    current = int(laps.loc[laps["driver"] == drv, "lap"].max()) if (laps["driver"] == drv).any() else 0
    p = preds[(preds["driver"] == drv) & (preds["t"] <= current)]
    latest = p[p["t"] == p["t"].max()].sort_values("h") if len(p) else p
    title = f"{drv}: previsão feita ao fim da volta {int(latest['t'].iloc[0])}" if len(latest) else drv
    st.plotly_chart(live_forecast(laps[laps["driver"] == drv], latest, title), width="stretch", config={"displaylogo": False})
    # só as voltas previstas que já foram disputadas até este momento (na gravação, o resto é "futuro")
    done = preds[preds["time_real"].notna() & preds["clean"]].merge(laps[["driver", "lap"]], on=["driver", "lap"])
    if len(done):
        mae = (done["time_p50"] - done["time_real"]).abs().mean()
        cov = ((done["time_real"] >= done["time_p10"]) & (done["time_real"] <= done["time_p90"])).mean()
        st.caption(f"Erro ao vivo até agora: MAE {mae:.3f} s em {len(done):,} voltas previstas já disputadas · ".replace(",", ".") +
                   f"cobertura do intervalo p10–p90: {cov:.0%}")


def render() -> None:
    c1, c2, c3 = st.columns([2, 3, 2])
    source = c1.radio("Fonte", ["f1-dash ao vivo", "Sessão gravada"], horizontal=True, key="live_source")
    if source == "f1-dash ao vivo":
        url = c2.text_input("Endereço do f1-dash", DEFAULT_URL, key="live_url")
        interval = c3.slider("Atualizar a cada (s)", 2, 15, 3, key="live_every")

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
    rec = c2.selectbox("Gravação", recs, format_func=lambda p: f"{p.parent.name} · {p.name}", key="rec_path")
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
        render_body(state, rec.parent, clock, f"gravação {clock / 60:.0f} min", "playing" if play else "recorded")

    recorded_view()
