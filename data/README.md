# Dados

Todos os dados usados pelo projeto ficam nesta pasta. Os dados brutos (`raw/`) e os derivados (`interim/`, `processed/`, `live/`) não vão para o git (ver `.gitignore`); só as tabelas de referência em `reference/` são versionadas.

```
data/
  raw/
    tracinginsights/<ano>/   sparse clone do TracingInsights, 2022 a 2026 (~230 MB)
    openf1/                  cópia parcial do OpenF1, 2023 a 2026 (~140 MB)
  reference/
    compounds.csv            compostos nominados (C0–C6) por GP, 2022–2026, com a fonte de cada linha
    tyre_specs.csv           faixa de compostos de cada temporada e o que mudou nela
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

Cobertura em 10 de novembro de 2026:

| Temporada | GPs | Voltas de corrida com tempo |
|---|---|---|
| 2022 | 22 | 23.529 |
| 2023 | 22 | 24.386 |
| 2024 | 24 | 26.574 |
| 2025 | 24 | 26.657 |
| 2026 | 20 (até São Paulo) | 23.473 |

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
| `processed/race_laps.parquet` | Voltas de corrida (~125 mil) com circuito, referência do piloto no fim de semana, alvo `pace`, flags de limpeza, variáveis por volta e a dureza nominal do composto (`compound_c`, C1–C6) com a nominação do fim de semana (`c_hard`, `c_medium`, `c_soft`) |
| `processed/dataset.npz` | Matrizes normalizadas por volta, variáveis estáticas por sequência (piloto-corrida) e o índice das janelas encoder/decoder |
| `processed/dataset_meta.json` | Nomes das variáveis, vocabulários, médias e desvios do treino e a divisão das corridas |

Decisões principais:

- **Referência e alvo:** o alvo é `pace = tempo de volta / referência − 1`. A referência é a melhor volta do piloto em pneu de pista seca no **quali**. Se o quali foi molhado (mais da metade das voltas com pneu de chuva) ou o piloto não tem volta seca nele, a referência é a melhor volta seca dos **treinos livres**. Sem volta seca nem nos treinos (Spa 2023), usa as voltas com pneu liso do quali molhado e, por último, a mediana do evento. A coluna `ref_source` guarda a fonte (Q, FP, Q_wet ou event_median) e `ref_from_quali` entra na rede: uma referência dos treinos é mais lenta que uma de quali seco, então o ritmo relativo daquela corrida fica menor (Las Vegas 2025, por exemplo).
- **Circuitos, não GPs:** as corridas são identificadas pelo circuito (`reference/circuits.csv`), porque o nome do GP muda entre anos. Em 2026, o GP da Espanha foi em Madri (Madring) e o GP do Bahrein, em Kuala Lumpur; o circuito de Barcelona passou a sediar o "GP de Barcelona". O rótulo de cada corrida é `<ano> <circuito>` (coluna `race_label`) e `reference/compounds.csv` também usa o circuito como chave.
- **Voltas limpas (`clean`):** com tempo, fora da volta 1, sem entrada ou saída dos boxes, sem SC/VSC/bandeira vermelha, não deletada nem gerada pelo FastF1, em pneu de pista seca e até 107% da mediana limpa do piloto na corrida. Só elas entram na perda e nas métricas; as outras continuam na sequência como entrada.
- **Voltas limpas ao vivo (`clean_live`):** como a mediana da corrida inteira não existe durante a corrida, as entradas da rede e os baselines usam uma versão calculável ao vivo: os mesmos critérios, com o corte de 107% sobre a mediana das voltas *anteriores* do piloto e um teto de ritmo de 30% acima da referência (para o começo da corrida, quando ainda não há voltas anteriores; voltas de relargada depois de bandeira vermelha chegam ao dobro do tempo normal). As duas versões concordam em 99,9% das voltas das corridas secas.
- **Corridas molhadas** (mais de 10% das voltas com pneu intermediário ou de chuva) ficam fora das sequências: 14 das 112 (a 14ª é o GP de São Paulo de 2026).
- **Divisão por corrida e em ordem cronológica:** as 3 últimas corridas secas são o teste (Singapura, Austin e México 2026), as 4 anteriores a validação (Zandvoort, Monza, Madring e Baku 2026) e todo o resto, de 2022 até a Hungria 2026, o treino. A validação escolhe a época do treino; o teste só é usado para medir o resultado. Normalização e vocabulários são ajustados só no treino; valores não vistos (um circuito novo, por exemplo) viram `<unk>`.
- **Janela:** ao fim da volta *t*, o encoder lê as voltas 1..*t* e o decoder prevê as voltas *t*+1..*t*+10, recebendo só as variáveis conhecidas no futuro (pneu, composto, paradas, progresso da corrida).
## Telemetria (`telemetry/`, na raiz do repositório)

A telemetria volta a volta das corridas e sprints de 2022 a 2026 fica fora de `data/` e fora do git. Ela não chega no feed ao vivo, então só serve para modelos avaliados fora do ao vivo. Numa máquina com o projeto dentro do OneDrive, `telemetry/` (e também `data/raw/` e `.venv/`) pode ser uma junção para uma pasta fora dele, para os ~19 GB não serem sincronizados.

```bash
py pipeline/fetch_telemetry.py                        # telemetry/tracinginsights/<ano>/ (~19 GB de JSON, 135 mil voltas)
.venv/Scripts/python -m pipeline.telemetry_compact    # telemetry/compact/ (~2,5 GB, ~2 min com 14 processos)
```

| Arquivo | Conteúdo |
|---|---|
| `tracinginsights/<ano>/<GP>/<Race\|Sprint>/<piloto>/<volta>_tel.json` | Telemetria original, ~7,5 amostras por segundo: velocidade, rpm, marcha, acelerador, freio, DRS, distância, posição x/y/z (em décimos de metro, apesar do DATA_REFERENCE dizer metros), acelerações e carro da frente |
| `compact/index.parquet` | Uma linha por volta: `row` (posição nos arrays), `race_id` (o mesmo de `race_laps.parquet`), sessão (R ou S), piloto, volta e medidas de qualidade da volta original |
| `compact/tel_f16.npy` | (voltas, 512, 10) em float16: velocidade, rpm, marcha, acelerador, freio, DRS, acc_x, acc_y, acc_z e distância ao carro da frente |
| `compact/tel_f32.npy` | (voltas, 512, 4) em float32: tempo desde o início da volta, x, y e z |
| `compact/meta.json` | Canais, unidades, grade e o commit do TracingInsights de cada ano |

Cada volta é reamostrada em 512 pontos igualmente espaçados na distância da própria volta (~10 m num circuito de 5 km), então o ponto *k* é o mesmo trecho da pista em todas as voltas de um circuito. Os arrays abrem com `np.load(..., mmap_mode="r")`. Conferência em 11/11/2026: todas as 99.315 voltas limpas de corrida têm telemetria, e a duração da telemetria bate com o tempo oficial da volta (diferença mediana de 0,000 s). A pasta `compact/parts/` é um cache por sessão: depois de atualizar a telemetria, a conversão refaz só as sessões que mudaram.

## OpenF1 (`raw/openf1/`)

Cópia da pasta `OpenF1` coletada antes do projeto (dados até o GP dos EUA de 2026), só com os endpoints úteis aqui: `meeting`, `sessions`, `drivers`, `laps`, `stints`, `pit` e `race_control`. Serve para checagem cruzada e para estimar o tempo perdido no pit lane (`pit.json`). Licença: CC BY-NC-SA 4.0.

Para atualizar, rode `node raw/openf1/download_openf1.js` (Node 18+). Esse script baixa todos os endpoints do OpenF1 para `raw/openf1/meetings/`.

## Referência (`reference/`)

`circuits.csv` liga cada GP (ano e nome do evento) ao circuito, de 2022 até o fim de 2026. `compounds.csv` traz os compostos que a Pirelli nominou como hard, medium e soft em cada GP de 2022 a 2026 (115 corridas), por ano e circuito, com a fonte de cada linha: comunicados da Pirelli e as páginas "What tyres will the teams and drivers have" da F1, com algumas linhas de imprensa especializada que reproduz o comunicado. A coluna `note` marca as nominações com pulo de composto (Austrália 2022; Spa, Austin e Cidade do México 2025). Las Vegas, Catar e Abu Dhabi de 2026 ficam em branco até a Pirelli anunciar; Las Vegas precisa ser preenchida antes do teste ao vivo.

`tyre_specs.csv` registra a faixa de cada temporada: C1–C5 em 2022, C0–C5 em 2023, C1–C5 em 2024, C1–C6 em 2025 e C1–C5 em 2026, com os pneus novos. A numeração não tem o mesmo sentido em todos os anos. Em 2023 entrou um C1 novo, entre o C2 e o antigo C1, que passou a se chamar C0. Por isso a dureza `compound_c` só se compara dentro de uma temporada; entre temporadas, o ano e o regulamento entram junto na rede.
