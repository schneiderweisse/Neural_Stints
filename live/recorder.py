"""Grava a sessão ao vivo a partir do fluxo SSE do f1-dash (GET /api/realtime).

O arquivo segue o formato do simulador do f1-dash, então a mesma gravação serve para:
- o ReplaySource do projeto (testes, ensaios e a demonstração na apresentação);
- o próprio simulador do f1-dash (`simulator replay <arquivo>`), para rever a corrida no painel.

Formato: a primeira linha é o estado completo no momento da conexão (o evento `initial` do SSE);
cada linha seguinte é uma mensagem {"type": 1, "target": "feed", "arguments": [tópico, dados,
"<ms desde o início da gravação>"]}. Os tópicos comprimidos (.z: telemetria e posição no mapa) são
gravados como chegam.

Arquivos em data/live/<ano>_<circuito>_<sessão>/recording.jsonl. Se a gravação for interrompida e
retomada na mesma sessão, a nova parte vai para recording_2.jsonl (com o seu próprio estado inicial),
porque as atualizações perdidas no intervalo tornariam uma continuação inconsistente. Quando o
f1-dash troca de sessão (novo SessionInfo), o gravador reconecta e abre o arquivo da sessão nova.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m live.recorder                 # f1-dash em localhost:4000
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from live.laps import session_key
from live.power import keep_awake

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
DEFAULT_URL = "http://127.0.0.1:4000/api/realtime"   # 127.0.0.1: ver live/feed.py


def sse_events(url: str, timeout: float = 60.0):
    """Eventos (nome, dados) de um fluxo SSE; termina quando a conexão fecha."""
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event, data = "message", []
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:                                   # fim do evento
                if data:
                    yield event, "\n".join(data)
                event, data = "message", []
            elif line.startswith(":"):                     # comentário (keep-alive)
                continue
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())


def next_path(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path, k = folder / "recording.jsonl", 2
    while path.exists():
        path, k = folder / f"recording_{k}.jsonl", k + 1
    return path


class Recording:
    """Um arquivo de gravação aberto, com o relógio relativo ao início."""

    def __init__(self, path: Path, initial: dict):
        self.path, self.t0, self.n = path, time.monotonic(), 0
        self.f = path.open("w", encoding="utf-8")
        self.f.write(json.dumps(initial, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.f.flush()

    def write(self, update: dict) -> None:
        ms = str(int((time.monotonic() - self.t0) * 1000))
        for topic, data in update.items():
            msg = {"type": 1, "target": "feed", "arguments": [topic, data, ms]}
            self.f.write(json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.n += 1
        if self.n % 50 == 0:
            self.f.flush()

    def close(self) -> None:
        self.f.flush()
        self.f.close()


def record(url: str = DEFAULT_URL, out_root: Path = LIVE, quiet: bool = False, stop_after: float | None = None) -> None:
    """Grava até ser interrompido (ou até `stop_after` segundos, para testes)."""
    t_start = time.monotonic()
    while stop_after is None or time.monotonic() - t_start < stop_after:
        rec = None
        try:
            for event, data in sse_events(url):
                if stop_after is not None and time.monotonic() - t_start >= stop_after:
                    break
                payload = json.loads(data)
                if event == "initial":
                    key = session_key(payload)
                    if not key:                            # f1-dash sem sessão: estado vazio
                        if not quiet:
                            print(f"{datetime.now():%H:%M:%S} f1-dash sem sessão; aguardando…")
                        break
                    rec = Recording(next_path(out_root / key), payload)
                    if not quiet:
                        print(f"{datetime.now():%H:%M:%S} gravando {key} -> {rec.path.name}")
                elif rec is not None:
                    rec.write(payload)
                    info = payload.get("SessionInfo")
                    if isinstance(info, dict) and "Name" in info:   # o f1-dash vai trocar de sessão
                        if not quiet:
                            print(f"{datetime.now():%H:%M:%S} troca de sessão; reconectando")
                        break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if not quiet:
                print(f"{datetime.now():%H:%M:%S} sem conexão com o f1-dash ({e.__class__.__name__}); tentando de novo")
        finally:
            if rec is not None:
                rec.close()
                if not quiet:
                    print(f"{datetime.now():%H:%M:%S} {rec.path.name}: {rec.n:,} mensagens gravadas")
        if stop_after is None or time.monotonic() - t_start < stop_after:
            time.sleep(5)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", type=Path, default=LIVE)
    ap.add_argument("--stop-after", type=float, help="segundos (para testes)")
    args = ap.parse_args()
    if keep_awake(display=True):
        print("modo acordado: o Windows não suspende nem apaga a tela enquanto o gravador roda (não feche a tampa)")
    try:
        record(args.url, args.out, stop_after=args.stop_after)
    except KeyboardInterrupt:
        print("gravação encerrada")


if __name__ == "__main__":
    main()
