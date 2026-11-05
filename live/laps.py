"""Transforma estados sucessivos do live timing em voltas completadas por piloto.

O feed só traz a última volta de cada carro, então o histórico é montado aqui: a cada estado,
quando o NumberOfLaps de um piloto aumenta, a volta que acabou vira uma linha com as mesmas
colunas usadas no treino (data/processed/race_laps.parquet): tempo, setores, speed traps, pneu,
posição, gaps, status de pista, paradas e clima.

Detalhes que só dá para acompanhar volta a volta, e por isso ficam no rastreador:
- status de pista: todos os códigos vistos durante a volta, como no TracingInsights ("1", "124"…);
- entrada e saída dos boxes: transições de InPit (e o PitOut) durante a volta;
- tempo da volta: às vezes o contador de voltas avança antes do LastLapTime ser atualizado. A volta
  fica pendente (lap_time vazio) até o valor mudar; se a volta seguinte terminar antes, usa a soma
  dos setores;
- idade do pneu: o TotalLaps do feed chega atrasado, então a idade é contada aqui, volta a volta,
  a partir do StartLaps do stint (voltas que o jogo de pneus já tinha). Como no FastF1, a volta de
  entrada é do stint antigo e a de saída já é do novo.
"""
import re
from dataclasses import dataclass, field

WET = {"INTERMEDIATE", "WET"}


def parse_time(value) -> float | None:
    """'1:18.123' -> 78.123; '28.456' -> 28.456; vazio -> None."""
    if value in (None, ""):
        return None
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", str(value).strip())
    return None if m is None else int(m.group(1) or 0) * 60 + float(m.group(2))


def parse_gap(value) -> float | None:
    """'+1.234' -> 1.234; '1 L' ou 'LAP 20' (líder ou volta de diferença) -> None."""
    if value in (None, ""):
        return None
    m = re.fullmatch(r"\+?(\d+(?:\.\d+)?)", str(value).strip())
    return float(m.group(1)) if m else None


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sub(d, *keys):
    """Acesso seguro a dicionários e listas aninhados."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return None
    return d


def as_list(x) -> list:
    """Stints e setores às vezes chegam como {"0": …, "1": …} antes de virarem lista."""
    if isinstance(x, dict):
        return [x[k] for k in sorted(x, key=int)]
    return x or []


def session_key(state: dict) -> str | None:
    info = state.get("SessionInfo") or {}
    year = str(info.get("StartDate", ""))[:4]
    circuit = sub(info, "Meeting", "Circuit", "ShortName")
    name = info.get("Name")
    if not (year and circuit and name):
        return None
    slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return f"{year}_{slug(circuit)}_{slug(name)}"


@dataclass
class DriverTrack:
    laps: int = 0                         # NumberOfLaps já registrado
    statuses: set = field(default_factory=set)
    in_pit_now: bool = False              # InPit no estado anterior
    pit_in: bool = False                  # entrou no pit lane durante a volta corrente (volta de entrada)
    pit_out: bool = False                 # começou ou saiu do pit lane durante a volta corrente (volta de saída)
    last_time: str | None = None          # LastLapTime quando a última volta foi registrada
    pending: dict | None = None           # volta registrada à espera do LastLapTime novo
    stints_seen: int = 0                  # stints que já existiam quando o piloto apareceu
    lap_stint: int | None = None          # stint (índice) em que a volta corrente começou
    stint_laps: dict = field(default_factory=dict)   # índice do stint -> idade do pneu na última volta


class LapTracker:
    """Acumula as voltas completadas de todos os pilotos de uma sessão."""

    def __init__(self):
        self.session = None
        self.drivers: dict[str, DriverTrack] = {}
        self.rows: list[dict] = []

    def reset(self, session: str | None) -> None:
        self.session, self.drivers, self.rows = session, {}, []

    def update(self, state: dict | None, clock: float) -> list[dict]:
        """Processa um estado; devolve as voltas que terminaram desde o estado anterior.

        Uma volta devolvida com lap_time vazio é completada (no mesmo dicionário) num estado seguinte.
        """
        if not state:
            return []
        key = session_key(state)
        if key != self.session:
            self.reset(key)
        status = str(sub(state, "TrackStatus", "Status") or "")
        lines = sub(state, "TimingData", "Lines") or {}
        new = []
        for num, line in lines.items():
            in_pit = bool(line.get("InPit"))
            last_time = sub(line, "LastLapTime", "Value")
            tr = self.drivers.get(num)
            if tr is None:
                n_stints = len(as_list(sub(state, "TimingAppData", "Lines", num, "Stints")))
                tr = self.drivers[num] = DriverTrack(laps=int(line.get("NumberOfLaps") or 0), in_pit_now=in_pit,
                                                     last_time=last_time, stints_seen=n_stints,
                                                     lap_stint=max(n_stints - 1, 0))
            if tr.pending is not None and last_time != tr.last_time:
                tr.pending["lap_time"], tr.pending["lap_time_source"] = parse_time(last_time), "feed"
                tr.last_time, tr.pending = last_time, None
            if status:
                tr.statuses.add(status)
            tr.pit_in |= in_pit and not tr.in_pit_now          # entrou no pit lane
            tr.pit_out |= (tr.in_pit_now and not in_pit) or bool(line.get("PitOut"))
            tr.in_pit_now = in_pit
            n = int(line.get("NumberOfLaps") or 0)
            if n > tr.laps:
                if tr.pending is not None:                       # o tempo nunca chegou: soma dos setores
                    p = tr.pending
                    if None not in (p["s1"], p["s2"], p["s3"]):
                        p["lap_time"], p["lap_time_source"] = round(p["s1"] + p["s2"] + p["s3"], 3), "sectors"
                    tr.pending = None
                row = self.lap_row(state, num, line, n, clock, tr)
                if n > 1 and last_time == tr.last_time:          # LastLapTime ainda é o da volta anterior
                    row["lap_time"], row["lap_time_source"] = None, None
                    tr.pending = row
                tr.last_time = last_time
                new.append(row)
                tr.laps = n
                tr.statuses = {status} if status else set()
                # se o carro cruza a linha dentro do pit lane, a volta seguinte é de saída
                tr.pit_in, tr.pit_out = False, in_pit
        self.rows.extend(new)
        return new

    def stint_of_lap(self, stints: list, tr: DriverTrack) -> tuple[int, dict]:
        """Stint da volta que acabou: a volta de entrada fica no stint antigo, mesmo se o novo já apareceu."""
        current = max(len(stints) - 1, 0)
        idx = tr.lap_stint if (tr.pit_in and tr.lap_stint is not None and tr.lap_stint < current) else current
        tr.lap_stint = current
        return idx, (stints[idx] if idx < len(stints) else {})

    def tyre_life(self, idx: int, stint: dict, tr: DriverTrack) -> int:
        if idx not in tr.stint_laps:
            if idx >= tr.stints_seen:                            # stint começou com o piloto já acompanhado
                tr.stint_laps[idx] = int(to_float(stint.get("StartLaps")) or 0)
            else:                                                # já estava em curso quando começamos a ler
                tr.stint_laps[idx] = max(int(to_float(stint.get("TotalLaps")) or 1) - 1, 0)
        tr.stint_laps[idx] += 1
        return tr.stint_laps[idx]

    def lap_row(self, state, num, line, lap, clock, tr: DriverTrack) -> dict:
        drv = sub(state, "DriverList", num) or {}
        stints = as_list(sub(state, "TimingAppData", "Lines", num, "Stints"))
        idx, stint = self.stint_of_lap(stints, tr)
        weather = state.get("WeatherData") or {}
        sectors = as_list(line.get("Sectors"))
        speeds = line.get("Speeds") or {}
        position = to_float(line.get("Position"))
        compound = str(stint.get("Compound") or "").upper() or None
        lap_time = parse_time(sub(line, "LastLapTime", "Value"))
        return {
            "session": self.session, "clock": clock, "driver_number": num, "driver": drv.get("Tla"),
            "team": drv.get("TeamName"), "lap": lap,
            "lap_time": lap_time, "lap_time_source": "feed" if lap_time is not None else None,
            "s1": parse_time(sub(sectors, 0, "Value")), "s2": parse_time(sub(sectors, 1, "Value")),
            "s3": parse_time(sub(sectors, 2, "Value")),
            "speed_i1": to_float(sub(speeds, "I1", "Value")), "speed_i2": to_float(sub(speeds, "I2", "Value")),
            "speed_fl": to_float(sub(speeds, "FL", "Value")), "speed_st": to_float(sub(speeds, "ST", "Value")),
            "compound": compound, "stint": idx + 1 if stints else None,
            "tyre_life": self.tyre_life(idx, stint, tr) if stints else None,
            "fresh_tyre": str(stint.get("New", "")).lower() == "true",
            "position": position, "gap_leader": 0.0 if position == 1 else parse_gap(line.get("GapToLeader")),
            "gap_ahead": None if position == 1 else parse_gap(sub(line, "IntervalToPositionAhead", "Value")),
            "track_status": "".join(sorted(tr.statuses)), "pit_in": tr.pit_in, "pit_out": tr.pit_out,
            "retired": bool(line.get("Retired")), "stopped": bool(line.get("Stopped")),
            "laps_total": to_float(sub(state, "LapCount", "TotalLaps")),
            "air_temp": to_float(weather.get("AirTemp")), "track_temp": to_float(weather.get("TrackTemp")),
            "humidity": to_float(weather.get("Humidity")), "pressure": to_float(weather.get("Pressure")),
            "rain": str(weather.get("Rainfall", "0")) not in ("0", "", "false", "False"),
            "wind_speed": to_float(weather.get("WindSpeed")),
        }
