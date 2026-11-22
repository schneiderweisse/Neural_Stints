"""Baixa a telemetria das corridas e sprints do TracingInsights (2022-2026) com um sparse clone parcial.

Mesmo método de fetch_tracinginsights.py (clone raso, sem blobs e com sparse checkout), mas só com a
telemetria volta a volta das sessões de corrida e sprint e os arquivos que ajudam a alinhá-la:

    <GP>/<Race|Sprint>/<piloto>/<volta>_tel.json   telemetria da volta (~7,5 Hz): rpm, velocidade,
                                                   marcha, acelerador, freio, DRS, distância, x/y/z,
                                                   acelerações e carro da frente
    <GP>/<Race|Sprint>/<piloto>/laptimes.json      tempos de volta (para ligar a telemetria às voltas)
    <GP>/<Race|Sprint>/corners.json                curvas por distância e tipo de trecho
    <GP>/<Race|Sprint>/session_info.json           dados da sessão
    <GP>/<Race|Sprint>/drivers.json                pilotos, números e equipes
    <GP>/<Race|Sprint>/rcm.json                    mensagens do race control (para o replay)

Os dados vão para telemetry/tracinginsights/<ano>/, fora do git. Nesta máquina, telemetry/ é uma
junção para C:\\NeuralStintsData\\telemetry, para o OneDrive não sincronizar os ~19 GB.
O significado de cada campo está em telemetry/tracinginsights/<ano>/DATA_REFERENCE.md.

Uso, a partir da pasta do repositório:
    py pipeline/fetch_telemetry.py              # 2022 a 2026
    py pipeline/fetch_telemetry.py 2025 2026    # só alguns anos
    py pipeline/fetch_telemetry.py 2026 --event "Las Vegas Grand Prix"   # só um GP (~150 MB), por
                                                # exemplo para o replay; num clone que já existe, soma
                                                # o GP ao que já está baixado, sem apagar nada

Rodar de novo atualiza os clones existentes (por exemplo, depois de uma nova corrida).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "telemetry" / "tracinginsights"
REPO = "https://github.com/TracingInsights/{year}.git"
YEARS = [2022, 2023, 2024, 2025, 2026]
SESSIONS = ["Race", "Sprint"]


def patterns(event: str | None = None) -> list[str]:
    """Os arquivos do sparse checkout: de todos os GPs ou só de um (event = nome da pasta do GP)."""
    gp = f"/{event}" if event else "/*"
    return ["/LICENSE", "/README.md", "/DATA_REFERENCE.md"] + [
        p for s in SESSIONS for p in (
            f"{gp}/{s}/*/*_tel.json",
            f"{gp}/{s}/*/laptimes.json",
            f"{gp}/{s}/corners.json",
            f"{gp}/{s}/session_info.json",
            f"{gp}/{s}/drivers.json",
            f"{gp}/{s}/rcm.json",
        )
    ]


PATTERNS = patterns()


def git(*args: str, cwd: Path | None = None, tries: int = 5) -> None:
    """Roda um comando git; repete se a conexão cair (o checkout baixa os arquivos sob demanda)."""
    for attempt in range(1, tries + 1):
        if subprocess.run(["git", *args], cwd=cwd).returncode == 0:
            return
        if attempt < tries:
            print(f"   falhou (tentativa {attempt}/{tries}), repetindo: git {' '.join(args)}", flush=True)
    raise RuntimeError(f"git {' '.join(args)} falhou {tries} vezes")


def fetch(year: int, event: str | None = None) -> None:
    dest = DEST / str(year)
    # com --event num clone que já existe, o GP é somado (add); "set" com um GP só apagaria os outros
    mode = "add" if event else "set"
    if (dest / ".git").exists():
        print(f"[{year}] atualizando o clone existente", flush=True)
        git("fetch", "--depth", "1", "--filter=blob:none", "origin", "main", cwd=dest)
        git("sparse-checkout", mode, *(["--no-cone"] if mode == "set" else []), *patterns(event), cwd=dest)
        git("reset", "--hard", "FETCH_HEAD", cwd=dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{year}] clone parcial (só a estrutura)", flush=True)
    git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "--single-branch",
        "--branch", "main", REPO.format(year=year), str(dest), tries=1)
    git("config", "core.longpaths", "true", cwd=dest)
    git("sparse-checkout", "set", "--no-cone", *patterns(event), cwd=dest)
    print(f"[{year}] checkout da telemetria de {event or 'corridas e sprints'}", flush=True)
    git("checkout", "main", cwd=dest)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*", type=int, help="temporadas (padrão: 2022 a 2026)")
    ap.add_argument("--event", help='só um GP, pelo nome da pasta no TracingInsights (ex.: "Las Vegas Grand Prix")')
    args = ap.parse_args()
    for year in args.years or YEARS:
        fetch(year, args.event)
        print(f"[{year}] pronto", flush=True)


if __name__ == "__main__":
    main()
