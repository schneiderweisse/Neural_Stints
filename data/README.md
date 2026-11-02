# Dados

Todos os dados usados pelo projeto ficam nesta pasta. Os dados brutos (`raw/`) e os derivados (`interim/`, `processed/`, `live/`) não vão para o git (ver `.gitignore`); só as tabelas de referência em `reference/` são versionadas.

```
data/
  raw/
    tracinginsights/<ano>/   sparse clone do TracingInsights, 2022 a 2026 (~230 MB)
    openf1/                  cópia parcial do OpenF1, 2023 a 2026 (~140 MB)
  reference/
    compounds.csv            compostos nominados (C1–C6) por GP, com a fonte de cada linha
  interim/, processed/       tabelas intermediárias e finais geradas pelo pipeline
  live/                      gravações das sessões ao vivo (.jsonl)
```

## TracingInsights (`raw/tracinginsights/`)

Um repositório git por temporada (`github.com/TracingInsights/<ano>`), clonado de forma rasa e parcial: só os arquivos abaixo, de todos os GPs e sessões (treinos, quali, sprint e corrida). A telemetria e os áudios de rádio ficam de fora.

| Arquivo | Conteúdo |
|---|---|
| `<GP>/<sessão>/<piloto>/laptimes.json` | Tempo de volta, setores, speed traps, composto, idade do pneu, stint, posição, status de pista e clima por volta |
| `<GP>/<sessão>/weather.json` | Clima da sessão, cerca de uma amostra por minuto |
| `<GP>/<sessão>/rcm.json` | Mensagens do race control (bandeiras, SC/VSC, penalidades) |
| `<GP>/<sessão>/drivers.json` | Pilotos, números e equipes |

O significado de cada campo está em `raw/tracinginsights/<ano>/DATA_REFERENCE.md` (nos anos em que o arquivo existe; o formato é o mesmo em todas as temporadas). Licença: Apache-2.0.

Cobertura em 2 de novembro de 2026:

| Temporada | GPs | Voltas de corrida com tempo |
|---|---|---|
| 2022 | 22 | 23.529 |
| 2023 | 22 | 24.386 |
| 2024 | 24 | 26.574 |
| 2025 | 24 | 26.657 |
| 2026 | 19 (até o México) | 21.997 |

Os nomes das pastas de sessão variam entre temporadas. Em 2023, a quali da sprint se chama `Sprint Shootout`; nos outros anos, `Sprint Qualifying`. Em 2022, a pasta `Sprint Qualifying` é uma cópia da corrida sprint (os `laptimes.json` são idênticos aos de `Sprint`). O módulo `pipeline/sessions.py` padroniza os nomes, descarta a cópia de 2022 e é a forma de percorrer as sessões no pipeline (`iter_sessions`).

Para baixar ou atualizar (por exemplo, depois de uma nova corrida), a partir da pasta `Neural_Stints`:

```bash
py pipeline/fetch_tracinginsights.py
```

## OpenF1 (`raw/openf1/`)

Cópia da pasta `OpenF1` coletada antes do projeto (dados até o GP dos EUA de 2026), só com os endpoints úteis aqui: `meeting`, `sessions`, `drivers`, `laps`, `stints`, `pit` e `race_control`. Serve para checagem cruzada e para estimar o tempo perdido no pit lane (`pit.json`). Licença: CC BY-NC-SA 4.0.

Para atualizar, rode `node raw/openf1/download_openf1.js` (Node 18+). Esse script baixa todos os endpoints do OpenF1 para `raw/openf1/meetings/`.

## Referência (`reference/`)

`compounds.csv` traz os compostos nominados (hard, medium, soft → C1–C6) de cada GP, com a fonte. Hoje cobre só os 7 circuitos do estudo de regulamento 2025 × 2026 (14 eventos) e será completada para 2022–2026.
