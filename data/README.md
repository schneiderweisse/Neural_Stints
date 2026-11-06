# Dados

Todos os dados usados pelo projeto ficam nesta pasta. Os dados brutos (`raw/`) e os derivados (`interim/`, `processed/`, `live/`) não vão para o git (ver `.gitignore`); só as tabelas de referência em `reference/` são versionadas.

```
data/
  raw/
    tracinginsights/<ano>/   sparse clone do TracingInsights, 2022 a 2026 (~230 MB)
    openf1/                  cópia parcial do OpenF1, 2023 a 2026 (~140 MB)
  reference/
    compounds.csv            compostos nominados (C1–C6) por GP, com a fonte de cada linha
    circuits.csv             circuito de cada GP, 2022–2026
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

## Tabelas derivadas (`interim/` e `processed/`)

Geradas pelo pipeline, nesta ordem, a partir da pasta `Neural_Stints`:

```bash
.venv/Scripts/python -m pipeline.build_laps   # interim/laps.parquet
.venv/Scripts/python -m pipeline.race_laps    # processed/race_laps.parquet
.venv/Scripts/python -m pipeline.sequences    # processed/dataset.npz e dataset_meta.json
```

| Arquivo | Conteúdo |
|---|---|
| `interim/laps.parquet` | Uma linha por volta de todas as sessões (~304 mil voltas), com nomes de coluna legíveis, `race_id` (`<ano>_<rodada>`) e a data da sessão |
| `processed/race_laps.parquet` | Voltas de corrida (~123 mil) com circuito, referência do piloto no fim de semana, alvo `pace`, flags de limpeza e variáveis por volta |
| `processed/dataset.npz` | Matrizes normalizadas por volta, variáveis estáticas por sequência (piloto-corrida) e o índice das janelas encoder/decoder |
| `processed/dataset_meta.json` | Nomes das variáveis, vocabulários, médias e desvios do treino e a divisão das corridas |

Decisões principais:

- **Referência e alvo:** o alvo é `pace = tempo de volta / referência − 1`. A referência é a melhor volta do piloto em pneu de pista seca no **quali**. Se o quali foi molhado (mais da metade das voltas com pneu de chuva) ou o piloto não tem volta seca nele, a referência é a melhor volta seca dos **treinos livres**. Sem volta seca nem nos treinos (Spa 2023), usa as voltas com pneu liso do quali molhado e, por último, a mediana do evento. A coluna `ref_source` guarda a fonte (Q, FP, Q_wet ou event_median) e `ref_from_quali` entra na rede: uma referência dos treinos é mais lenta que uma de quali seco, então o ritmo relativo daquela corrida fica menor (Las Vegas 2025, por exemplo).
- **Circuitos, não GPs:** as corridas são identificadas pelo circuito (`reference/circuits.csv`), porque o nome do GP muda entre anos. Em 2026, o GP da Espanha foi em Madri (Madring) e o GP do Bahrein, em Kuala Lumpur; o circuito de Barcelona passou a sediar o "GP de Barcelona". O rótulo de cada corrida é `<ano> <circuito>` (coluna `race_label`) e `reference/compounds.csv` também usa o circuito como chave.
- **Voltas limpas (`clean`):** com tempo, fora da volta 1, sem entrada ou saída dos boxes, sem SC/VSC/bandeira vermelha, não deletada nem gerada pelo FastF1, em pneu de pista seca e até 107% da mediana limpa do piloto na corrida. Só elas entram na perda e nas métricas; as outras continuam na sequência como entrada.
- **Voltas limpas ao vivo (`clean_live`):** como a mediana da corrida inteira não existe durante a corrida, as entradas da rede e os baselines usam uma versão calculável ao vivo: os mesmos critérios, com o corte de 107% sobre a mediana das voltas *anteriores* do piloto e um teto de ritmo de 30% acima da referência (para o começo da corrida, quando ainda não há voltas anteriores; voltas de relargada depois de bandeira vermelha chegam ao dobro do tempo normal). As duas versões concordam em 99,9% das voltas das corridas secas.
- **Corridas molhadas** (mais de 10% das voltas com pneu intermediário ou de chuva) ficam fora das sequências: 13 das 111.
- **Divisão por corrida e em ordem cronológica:** as 3 últimas corridas secas são o teste (Singapura, Austin e México 2026), as 4 anteriores a validação (Zandvoort, Monza, Madring e Baku 2026) e todo o resto, de 2022 até a Hungria 2026, o treino. A validação escolhe a época do treino; o teste só é usado para medir o resultado. Normalização e vocabulários são ajustados só no treino; valores não vistos (um circuito novo, por exemplo) viram `<unk>`.
- **Janela:** ao fim da volta *t*, o encoder lê as voltas 1..*t* e o decoder prevê as voltas *t*+1..*t*+10, recebendo só as variáveis conhecidas no futuro (pneu, composto, paradas, progresso da corrida).
## OpenF1 (`raw/openf1/`)

Cópia da pasta `OpenF1` coletada antes do projeto (dados até o GP dos EUA de 2026), só com os endpoints úteis aqui: `meeting`, `sessions`, `drivers`, `laps`, `stints`, `pit` e `race_control`. Serve para checagem cruzada e para estimar o tempo perdido no pit lane (`pit.json`). Licença: CC BY-NC-SA 4.0.

Para atualizar, rode `node raw/openf1/download_openf1.js` (Node 18+). Esse script baixa todos os endpoints do OpenF1 para `raw/openf1/meetings/`.

## Referência (`reference/`)

`circuits.csv` liga cada GP (ano e nome do evento) ao circuito, de 2022 até o fim de 2026. `compounds.csv` traz os compostos nominados (hard, medium, soft → C1–C6) por ano e circuito, com a fonte. Hoje cobre só os 7 circuitos do estudo de regulamento 2025 × 2026 (14 eventos) e será completada para 2022–2026.
