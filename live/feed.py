"""Fontes do estado da corrida: o f1-dash ao vivo e o replay de uma gravação.

As duas devolvem o mesmo objeto: o estado consolidado do live timing da F1 (um dicionário com os
tópicos TimingData, TimingAppData, TrackStatus, WeatherData, LapCount, DriverList, SessionInfo…),
igual ao que o serviço `realtime` do f1-dash serve em GET /api/current.

- HttpSource lê o /api/current a cada poucos segundos. Sem sessão ou com a conexão com a F1 caída,
  o f1-dash devolve {} e a fonte devolve None.
- ReplaySource lê um arquivo gravado no formato do simulador do f1-dash (primeira linha: estado
  inicial; depois, uma mensagem SignalR por linha) e aplica as atualizações com a mesma regra de
  merge do f1-dash. Serve para desenvolver e testar sem sessão ao vivo.
"""
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

# 127.0.0.1 e não localhost: o f1-dash só escuta em IPv4, e no Windows cada conexão a "localhost" tenta antes
# o IPv6 (::1) e leva ~2 s para desistir (medido: 2.050 ms contra 10 ms por leitura)
DEFAULT_URL = "http://127.0.0.1:4000/api/current"


def merge(base, update):
    """Mesma regra do f1-dash (realtime/src/services/state_service.rs).

    Objeto sobre objeto: merge recursivo chave a chave. Objeto sobre lista: as chaves numéricas
    atualizam (ou acrescentam) itens da lista, como nos stints. Qualquer outro caso substitui.
    """
    if isinstance(base, dict) and isinstance(update, dict):
        for k, v in update.items():
            base[k] = merge(base.get(k), v)
        return base
    if isinstance(base, list) and isinstance(update, dict):
        for k, v in update.items():
            if k.isdigit():
                i = int(k)
                if i < len(base):
                    base[i] = merge(base[i], v)
                else:
                    base.append(v)
        return base
    return update


STALE_AFTER = 90.0   # s; a F1 manda um Heartbeat a cada ~15 s enquanto o feed está vivo


def heartbeat_time(state: dict | None) -> float | None:
    """Horário do último Heartbeat da F1 no estado, em segundos desde 1970 (None se não houver)."""
    utc = ((state or {}).get("Heartbeat") or {}).get("Utc")
    if not utc:
        return None
    try:
        return datetime.fromisoformat(str(utc)[:26].rstrip("Z") + "+00:00").timestamp()
    except ValueError:
        return None


def heartbeat_age(state: dict | None) -> float | None:
    """Segundos desde o último Heartbeat da F1 no estado (None se não houver).

    Com o f1-dash desconectado da F1 (ou o notebook acordando de uma suspensão), o estado fica parado
    e esta idade cresce; acima de STALE_AFTER, os dados não estão mais chegando.
    """
    then = heartbeat_time(state)
    return None if then is None else datetime.now(timezone.utc).timestamp() - then


class HttpSource:
    """Lê o estado do f1-dash local a cada `interval` segundos."""

    def __init__(self, url: str = DEFAULT_URL, interval: float = 3.0, timeout: float = 5.0):
        self.url, self.interval, self.timeout = url, interval, timeout

    def fetch(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                state = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            return None
        return state or None                       # {} = sem sessão ou sem conexão com a F1

    def __iter__(self) -> Iterator[tuple[float, dict | None]]:
        while True:
            t0 = time.time()
            yield t0, self.fetch()
            time.sleep(max(0.0, self.interval - (time.time() - t0)))


class ReplaySource:
    """Reproduz uma gravação, entregando o estado a cada `interval` segundos do relógio da gravação.

    `speed` controla o ritmo real da reprodução: 0 (padrão) reproduz o mais rápido possível;
    1 é tempo real; 10 é dez vezes mais rápido. Os tópicos comprimidos (.z: telemetria e posição
    no mapa) são ignorados, porque o modelo não usa.
    """

    def __init__(self, path: str | Path, interval: float = 3.0, speed: float = 0.0):
        self.path, self.interval, self.speed = Path(path), interval, speed

    def __iter__(self) -> Iterator[tuple[float, dict]]:
        with self.path.open(encoding="utf-8") as f:
            state = json.loads(f.readline())
            state = {k: v for k, v in state.items() if not k.endswith(".z")}
            next_emit, clock, wall0 = 0.0, 0.0, time.time()
            for line in f:
                msg = json.loads(line)
                if msg.get("type") != 1 or msg.get("target") != "feed":
                    continue
                topic, data, stamp = msg["arguments"]
                clock = float(stamp) / 1000 if str(stamp).replace(".", "", 1).isdigit() else clock
                if not topic.endswith(".z"):
                    state[topic] = merge(state.get(topic), data)
                if clock >= next_emit:
                    if self.speed:
                        time.sleep(max(0.0, wall0 + clock / self.speed - time.time()))
                    yield clock, state
                    next_emit = clock + self.interval
            yield clock, state
