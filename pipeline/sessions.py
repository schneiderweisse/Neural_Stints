"""Nomes de sessão do TracingInsights e como percorrer as sessões baixadas.

O nome das pastas de sessão mudou ao longo das temporadas:

- 2023: a quali da sprint se chama "Sprint Shootout"; nos outros anos, "Sprint Qualifying".
- 2022: ainda não existia quali da sprint (o grid da sprint vinha da quali de sexta). A pasta
  "Sprint Qualifying" de 2022 é uma cópia da corrida sprint: os laptimes.json são idênticos aos de
  "Sprint". Ela é ignorada para não contar as voltas da sprint duas vezes.

Todo o pipeline usa os nomes canônicos de SESSION_TYPES.
"""
from collections.abc import Iterable, Iterator
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "tracinginsights"

# Nome canônico -> tipo de sessão
SESSION_TYPES = {
    "Practice 1": "FP",
    "Practice 2": "FP",
    "Practice 3": "FP",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint": "S",
    "Race": "R",
}
# Nome da pasta -> nome canônico
SESSION_ALIASES = {"Sprint Shootout": "Sprint Qualifying"}
# (ano, nome da pasta) que não devem ser lidos
SKIP = {(2022, "Sprint Qualifying")}


def canonical_session(year: int, folder: str) -> str | None:
    """Nome canônico da sessão, ou None se a pasta deve ser ignorada."""
    if (year, folder) in SKIP:
        return None
    name = SESSION_ALIASES.get(folder, folder)
    return name if name in SESSION_TYPES else None


def iter_sessions(years: Iterable[int] | None = None, types: Iterable[str] | None = None,
                  include_testing: bool = False) -> Iterator[tuple[int, str, str, Path]]:
    """Percorre (ano, GP, sessão canônica, pasta) de todas as sessões baixadas.

    types filtra pelos tipos de SESSION_TYPES (por exemplo {"R"} só para corridas).
    Os testes de pré-temporada ficam de fora, a menos que include_testing=True.
    """
    years = sorted(years) if years is not None else sorted(int(p.name) for p in RAW.iterdir() if p.name.isdigit())
    types = set(types) if types is not None else None
    for year in years:
        for event in sorted(p for p in (RAW / str(year)).iterdir() if p.is_dir()):
            is_gp = event.name.endswith("Grand Prix")
            is_testing = event.name.startswith("Pre-Season")
            if not (is_gp or (include_testing and is_testing)):
                continue
            for folder in sorted(p for p in event.iterdir() if p.is_dir()):
                session = canonical_session(year, folder.name)
                if session is None or (types is not None and SESSION_TYPES[session] not in types):
                    continue
                yield year, event.name, session, folder
