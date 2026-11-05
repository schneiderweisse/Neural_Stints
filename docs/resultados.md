# Formato dos resultados e métricas

Este documento define como cada modelo grava os seus resultados e quais medidas são calculadas a partir deles. Todos os modelos (redes e baselines) seguem o mesmo formato, então o painel e as análises funcionam para qualquer modelo sem código específico. A previsão ao vivo usa o mesmo formato.

A seção [O que existe na v0 (Interlagos)](#o-que-existe-na-v0-interlagos) lista a parte deste formato que já estará pronta para o GP de São Paulo.

## Pasta de um treino

Cada treino (ou cada ajuste de baseline) grava uma pasta em `runs/`, que fica fora do git:

```
runs/<modelo>_<aaaammdd>_s<semente>/     ex.: runs/gru_20261104_s1/
  config.json          o que foi treinado e como
  history.csv          a perda a cada época (só redes)
  predictions.parquet  a tabela padrão de previsões
  metrics.json         o resumo das métricas, calculado a partir da tabela
  model.pt             pesos e normalização, usados na previsão ao vivo (só redes)
  backtest.html        relatório interativo contra o ingênuo em corridas do teste (modeling.backtest)
  backtest_drivers.csv erro por piloto no backtest
  backtest_summary.json resumo do backtest por corrida
```

### `predictions.parquet`

Formato longo: uma linha é a previsão de uma volta. Uma janela com horizonte 10 gera 10 linhas.

| Coluna | Tipo | Exemplo | Descrição |
|---|---|---|---|
| `run_id` | texto | `gru_20261104_s1` | pasta do treino |
| `model` | texto | `gru` | família do modelo: `naive`, `gru`, `lstm`, `lmm`, `lgbm` |
| `split` | texto | `test` | `train`, `val`, `test` ou `live` |
| `race_id` | texto | `2026_19` | `<ano>_<rodada>` |
| `race_label` | texto | `2026 Mexico City` | `<ano> <circuito>` |
| `circuit` | texto | `Mexico City` | circuito |
| `year` | inteiro | `2026` | temporada |
| `regulation` | texto | `2026` | era de regulamento: `2022-25` ou `2026` |
| `driver` | texto | `VER` | piloto |
| `team` | texto | `Red Bull Racing` | equipe |
| `t` | inteiro | `20` | volta em que a previsão foi feita ("agora") |
| `h` | inteiro | `3` | horizonte: voltas à frente, de 1 a 10 |
| `lap` | inteiro | `23` | volta prevista (`t + h`) |
| `compound` | texto | `MEDIUM` | composto na volta prevista |
| `tyre_life` | inteiro | `16` | idade do pneu na volta prevista |
| `stint` | inteiro | `2` | stint da volta prevista |
| `same_stint` | lógico | `True` | a volta prevista está no mesmo stint da volta `t` (tarefa A) ou depois de uma parada (tarefa B) |
| `clean` | lógico | `True` | a volta real é limpa (critério `clean`, com a mediana da corrida inteira); só essas entram nas métricas |
| `ref_lap` | real (s) | `78.95` | referência do piloto no fim de semana |
| `pace_p10`, `pace_p50`, `pace_p90` | real | `0.031`, `0.034`, `0.038` | previsão do ritmo relativo (`tempo / ref_lap − 1`) |
| `pace_real` | real | `0.033` | ritmo real |
| `time_p10`, `time_p50`, `time_p90` | real (s) | `81.40`, `81.63`, `81.95` | previsão em segundos: `(1 + ritmo) × ref_lap` |
| `time_real` | real (s) | `81.56` | tempo real |

Modelos sem previsão de intervalo, como o ingênuo, deixam `p10` e `p90` vazios.

Colunas a mais nas previsões ao vivo (`split = live`):

| Coluna | Tipo | Exemplo | Descrição |
|---|---|---|---|
| `predicted_at` | data e hora (UTC) | `2026-11-08T17:41:05Z` | quando a previsão foi feita |
| `scenario` | texto | `A` | `A`: o piloto segue na pista; `B`: para nos boxes |
| `scenario_compound` | texto | `HARD` | pneu colocado no cenário B (vazio no A) |
| `scenario_pit_lap` | inteiro | `24` | volta da parada no cenário B (vazio no A) |

Ao vivo, `pace_real` e `time_real` ficam vazios até a volta acontecer e são preenchidos depois.

### `config.json`

| Campo | Exemplo |
|---|---|
| `run_id`, `model`, `seed` | `gru_20261104_s1`, `gru`, `1` |
| `hyperparameters` | `{"hidden": 64, "layers": 1, "dropout": 0.1, "lr": 0.001, "batch_sequences": 32, "max_epochs": 60, "patience": 8}` |
| `eras_strategy` | `pooled` (todas as eras, com embeddings), `2026_only` ou `pretrain_finetune` |
| `features` | as listas `observed`, `known`, `static_cat` e `static_x` do `dataset_meta.json` |
| `horizon`, `quantiles` | `10`, `[0.1, 0.5, 0.9]` |
| `data` | divisão das corridas usada (`splits` do `dataset_meta.json`) |
| `n_params`, `train_seconds`, `best_epoch` | `33411`, `214.5`, `27` |
| `device`, `git_commit`, `created_at` | `cpu`, `28ef721`, `2026-11-04T15:02:11Z` |

### `history.csv`

Uma linha por época:

| Coluna | Descrição |
|---|---|
| `epoch` | número da época |
| `train_loss` | perda pinball média no treino |
| `val_loss` | perda pinball média na validação |
| `val_mae_h1` | MAE (s) da próxima volta na validação |
| `lr` | taxa de aprendizado na época |
| `seconds` | duração da época |
| `best` | se é a melhor época até ali (critério: `val_loss`) |

### `metrics.json`

O resumo das medidas da próxima seção, por split, recalculável a qualquer momento a partir de `predictions.parquet`.

## Medidas

Todas são calculadas só sobre as voltas limpas (`clean = True`), em segundos e em pontos percentuais de ritmo, separadas por split e por horizonte `h`.

### Precisão pontual (sobre o p50)

| Medida | Definição | Pergunta |
|---|---|---|
| MAE | média de \|p50 − real\| | quanto erra, em média |
| RMSE | raiz da média de (p50 − real)² | o mesmo, penalizando erros grandes |
| Viés | média de (p50 − real) | se o modelo é sistematicamente otimista (< 0) ou pessimista (> 0) |
| Skill vs. ingênuo | 1 − MAE do modelo / MAE do ingênuo | quanto melhora sobre repetir o ritmo recente (0 = igual, > 0 = melhor) |
| Erro em 5 voltas | \|Σ p50 − Σ real\| nas voltas `t+1` a `t+5`, quando todas são limpas | o erro no tempo acumulado, que é o que pesa numa decisão de parada |

### Qualidade da incerteza (sobre p10 e p90)

| Medida | Definição | Ideal |
|---|---|---|
| Pinball loss | média da perda quantílica nos três quantis (a perda de treino) | menor |
| Cobertura 80% | fração de voltas com p10 ≤ real ≤ p90 | ~80% |
| Calibração | fração de voltas com real < p10 e com real < p90 | ~10% e ~90% |
| Largura | média de p90 − p10 | estreita, com a cobertura certa |

### Medidas do problema

| Medida | Para quê |
|---|---|
| Todas as anteriores separadas por `same_stint` | tarefa A (mesmo pneu) × tarefa B (curva depois da parada) |
| Inclinação da degradação: em cada stint, a inclinação de p50 × idade do pneu contra a inclinação real | se a rede aprende a forma da degradação; comparável ao modelo linear misto |
| Recortes por composto, circuito, regulamento, fase do stint (idade do pneu) e tráfego | onde o modelo acerta e onde erra |

### Diagnóstico do treino

Curvas de perda de treino e validação por época, melhor época, número de parâmetros e tempo de treino.

### Comparação entre modelos

- Cada configuração é treinada com várias sementes (por exemplo 5), e as métricas são relatadas como média ± desvio-padrão entre sementes.
- A diferença de MAE entre dois modelos tem intervalo de confiança por bootstrap reamostrando corridas, não voltas, porque voltas da mesma corrida são correlacionadas.

## Previsões ao vivo

O serviço de previsão e o painel são processos separados. O serviço lê o f1-dash, roda o modelo a cada volta completada e grava em disco; o painel só lê os arquivos. Cada sessão ao vivo tem uma pasta em `data/live/`, fora do git:

```
data/live/<ano>_<circuito>_<sessão>/       ex.: data/live/2026_interlagos_race/
  recording.jsonl      fluxo bruto do f1-dash (/api/realtime), com horário; permite o replay
  laps.parquet         histórico de voltas montado pelo serviço, com as mesmas colunas de race_laps.parquet
  predictions.parquet  previsões, no formato da tabela padrão com split = live
```

Os arquivos são sempre gravados num arquivo temporário e depois renomeados, para o painel nunca ler um arquivo pela metade. O replay de uma gravação passa pelo mesmo serviço e gera os mesmos arquivos.

## O que existe na v0 (Interlagos)

A v0 é a versão que roda ao vivo no GP de São Paulo, em 8 de novembro de 2026. Ela usa só uma parte do formato acima; o resto entra a partir da v1 (Las Vegas), conforme os modelos ficarem prontos. O painel só mostra o que existe.

### Modelos

| Modelo | `run_id` | Na v0 |
|---|---|---|
| Ingênuo: média do ritmo das últimas 3 voltas limpas até a volta `t` (critério `clean_live`), repetida em todos os horizontes; sem intervalo | `naive_<data>` | ✅ |
| GRU encoder–decoder, saída quantílica, todas as eras com embeddings (`pooled`), 1 semente | `gru_<data>_s1` | ✅ |
| LSTM | | v1 |
| Modelo linear misto e LightGBM | | v1 |
| Estratégias de eras `2026_only` e `pretrain_finetune` | | v1 |
| Várias sementes por configuração | | v1 |

### Arquivos

| Arquivo | Na v0 |
|---|---|
| `config.json` | ✅ completo |
| `history.csv` | ✅ (GRU) |
| `predictions.parquet` | ✅ todas as colunas, para validação e teste |
| `metrics.json` | ✅ com as medidas marcadas abaixo |
| `model.pt` | ✅ (GRU) |
| `data/live/2026_interlagos_race/` | ✅ `recording.jsonl`, `laps.parquet` e `predictions.parquet` |
| Colunas `scenario*` ao vivo | só o cenário `A`; o cenário `B` (parar agora) chega com o simulador de estratégia na v1 |

### Medidas

| Medida | Na v0 |
|---|---|
| MAE, RMSE e viés por horizonte | ✅ |
| Skill vs. ingênuo | ✅ |
| Erro em 5 voltas | ✅ |
| Pinball loss, cobertura 80%, calibração e largura | ✅ (GRU) |
| Separação por `same_stint` | ✅ |
| Recorte por composto | ✅ |
| Recortes por circuito, regulamento, fase do stint e tráfego | v1 |
| Inclinação da degradação contra o modelo misto | v1 |
| Média ± desvio entre sementes e bootstrap por corrida | v1 |
| Erro ao vivo em Interlagos | ✅ calculado depois da corrida, com os tempos oficiais |

### Painel

| Aba | Na v0 |
|---|---|
| Exploração | curvas empíricas de ritmo × idade do pneu por composto, circuito e regulamento |
| Modelos | ingênuo × GRU v0: MAE por horizonte, cobertura, calibração e curva de treino |
| Backtest | previsto × real volta a volta nas corridas de validação e teste (incluindo Austin e México) |
| Ao vivo | Interlagos: por piloto, voltas reais e previsão das próximas 10 voltas no cenário A, com faixa p10–p90; tabela de todos os pilotos |
| Simulador de estratégia (cenário B) | v1 |
