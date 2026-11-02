"""Baixa os tempos de volta do TracingInsights (2022-2026) com um sparse clone parcial.

Cada temporada é um repositório (github.com/TracingInsights/<ano>). O clone é raso (--depth 1),
sem blobs (--filter=blob:none) e com sparse checkout, então só são baixados os arquivos abaixo,
de todos os GPs e sessões:

    <GP>/<sessão>/<piloto>/laptimes.json   tempos de volta, setores, pneu, status, clima
    <GP>/<sessão>/weather.json             clima da sessão (~1 amostra por minuto)
    <GP>/<sessão>/rcm.json                 mensagens do race control
    <GP>/<sessão>/drivers.json             pilotos, números e equipes

A telemetria (*_tel.json) e os áudios de rádio ficam de fora.

Uso, a partir da pasta Neural_Stints:
    py pipeline/fetch_tracinginsights.py              # 2022 a 2026
    py pipeline/fetch_tracinginsights.py 2025 2026    # só alguns anos

Rodar de novo atualiza os clones existentes (por exemplo, depois de uma nova corrida).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "tracinginsights"
REPO = "https://github.com/TracingInsights/{year}.git"
YEARS = [2022, 2023, 2024, 2025, 2026]
PATTERNS = [
    "/LICENSE",
    "/README.md",
    "/DATA_REFERENCE.md",
    "/*/*/*/laptimes.json",
    "/*/*/weather.json",
    "/*/*/rcm.json",
    "/*/*/drivers.json",
]


def git(*args: str, cwd: Path | None = None, tries: int = 3) -> None:
    """Roda um comando git; repete se a conexão cair (o checkout baixa os arquivos sob demanda)."""
    for attempt in range(1, tries + 1):
        if subprocess.run(["git", *args], cwd=cwd).returncode == 0:
            return
        if attempt < tries:
            print(f"   falhou (tentativa {attempt}/{tries}), repetindo: git {' '.join(args)}")
    raise RuntimeError(f"git {' '.join(args)} falhou {tries} vezes")


def fetch(year: int) -> None:
    dest = DEST / str(year)
    if (dest / ".git").exists():
        print(f"[{year}] atualizando o clone existente")
        git("fetch", "--depth", "1", "--filter=blob:none", "origin", "main", cwd=dest)
        git("reset", "--hard", "FETCH_HEAD", cwd=dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{year}] clone parcial (só a estrutura)")
    git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "--single-branch",
        "--branch", "main", REPO.format(year=year), str(dest), tries=1)
    git("sparse-checkout", "set", "--no-cone", *PATTERNS, cwd=dest)
    print(f"[{year}] checkout dos tempos de volta, clima, race control e pilotos")
    git("checkout", "main", cwd=dest)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    years = [int(y) for y in sys.argv[1:]] or YEARS
    for year in years:
        fetch(year)


if __name__ == "__main__":
    main()
