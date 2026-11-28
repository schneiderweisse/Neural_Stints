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
  model_card.json      a ficha do modelo para o painel: camadas e parâmetros, pesos por portão e embeddings
                       (modeling.model_card; o painel e a demo não abrem o model.pt)
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
| `eras_strategy` | `pooled` (todas as eras, com embeddings), `pooled_noreg`, `2026_only`, `pre_2022_25`, `pretrain_finetune` ou `pooled_finetune` ([Eras de regulamento](#eras-de-regulamento)) |
| `hyperparameters` dos runs das eras | também `train_years`, `init` (run de partida do fine-tuning), `zero_cat` e `unk_categories` (categorias tratadas como `<unk>`, também ao vivo) |
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
  recording.jsonl      fluxo bruto do f1-dash (/api/realtime) no formato do simulador do f1-dash;
                       uma parte nova (recording_2.jsonl…) a cada reinício do gravador
  laps.parquet         histórico de voltas montado pelo live.reader, com os nomes de coluna de race_laps.parquet
  predictions.parquet  previsões, no formato da tabela padrão com split = live; um previsor reaberto
                       retoma o arquivo e prevê as voltas que passaram enquanto estava parado
  predictions_replay.parquet  as previsões do live.predict --once, de todas as voltas de uma vez
```

Os arquivos são sempre gravados num arquivo temporário e depois renomeados, para o painel nunca ler um arquivo pela metade. O replay de uma gravação passa pelo mesmo serviço e gera os mesmos arquivos.

## O que existe na v0 (Interlagos)

A v0 é a versão que roda ao vivo no GP de São Paulo, em 8 de novembro de 2026. Ela usa só uma parte do formato acima; o resto entra a partir da v1 (Las Vegas), conforme os modelos ficarem prontos. O painel só mostra o que existe.

### Modelos

| Modelo | `run_id` | Na v0 |
|---|---|---|
| Ingênuo: média do ritmo das últimas 3 voltas limpas até a volta `t` (critério `clean_live`), repetida em todos os horizontes; sem intervalo | `naive_<data>` | ✅ |
| GRU encoder–decoder, saída quantílica, todas as eras com embeddings (`pooled`), 1 semente | `gru_<data>_s1` | ✅ |
| LSTM: a mesma rede com `--cell lstm`, 5 sementes da configuração escolhida na validação (`modeling.cells`, feita em 20/11) | `lstm_<data>_ens5` | ✅ |
| Modelo linear misto: composto × idade do pneu, progresso da corrida e temperatura, com intercepto e inclinação no progresso aleatórios por piloto:corrida (`modeling.lmm`, feito em 20/11) | `lmm_<data>` | ✅ |
| LightGBM com lags do ritmo, um modelo por quantil (`modeling.lgbm`, feito em 20/11) | `lgbm_<data>` | ✅ |
| Estratégias de eras: só 2026, só 2022–25, sem o embedding de regulamento e fine-tunings em 2026, 5 sementes cada (`modeling.eras`, feito em 20/11) | `eras_<data>/<estratégia>_ens5` | ✅ |
| Várias sementes por configuração (bateria `modeling.sweep`) | `sweep_<data>/` | ✅ |
| Ensemble da GRU: média dos quantis de 5 sementes (`modeling.ensemble`), usado ao vivo e no painel | `gru_<data>_ens5` | ✅ |

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
| Média ± desvio entre sementes | ✅ na bateria (`report.md`) |
| Bootstrap da diferença de MAE | ✅ por piloto:corrida (`modeling.compare`); por corrida inteira, v1 |
| Erro ao vivo em Interlagos | ✅ calculado depois da corrida ([relatório](race_reports/2026_sao_paulo.md)), com os tempos do feed gravado |

### Painel

| Aba | Na v0 |
|---|---|
| Exploração | curvas empíricas de ritmo × idade do pneu por composto, circuito e regulamento |
| Modelos | ingênuo × GRU v0: MAE por horizonte, cobertura, calibração e curva de treino |
| Backtest | previsto × real volta a volta nas corridas de validação e teste (incluindo Austin e México) |
| Ao vivo | Interlagos: por piloto, voltas reais e previsão das próximas 10 voltas no cenário A, com faixa p10–p90; tabela de todos os pilotos |
| Simulador de estratégia (cenário B) | v1 |

## Comparação dos modelos na validação e no teste

`modeling.compare` mede todos os runs nas mesmas voltas: limpas, com valor real e com previsão de todos os modelos (o ingênuo não prevê antes da primeira volta limpa). O skill é contra o ingênuo. O intervalo da diferença de MAE para a GRU vem de um bootstrap em blocos por piloto:corrida (2000 reamostras). Ele não cobre a variação entre corridas, porque cada divisão tem só 3 ou 4 corridas, e por isso é otimista. Resultados em `runs/compare_20261120/comparison.json`.

**Validação** (Zandvoort, Monza, Madring, Baku; 33.507 voltas em comum)

| Modelo | MAE (s) | RMSE (s) | Viés (s) | Erro em 5 voltas (s) | Cobertura 80% | Pinball (s) | Skill vs. ingênuo | MAE − GRU (IC 95%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo `naive_20261106` | 0,783 | 1,131 | +0,402 | 2,68 | — | — | — | +0,200 (+0,170 a +0,232) |
| Linear misto `lmm_20261120` | 0,772 | 1,140 | +0,104 | 2,85 | 75,2% | 0,265 | +1,4% | +0,190 (+0,149 a +0,229) |
| LightGBM `lgbm_20261120` | 0,597 | 0,889 | +0,089 | 2,00 | 71,4% | 0,204 | +23,7% | +0,015 (+0,002 a +0,028) |
| LSTM `lstm_20261120_ens5` | **0,569** | **0,853** | +0,063 | **1,84** | 78,6% | **0,195** | **+27,3%** | −0,013 (−0,023 a −0,004) |
| GRU `gru_20261106_ens5` | 0,582 | 0,863 | +0,089 | 1,92 | 79,3% | 0,198 | +25,6% | — |

**Teste** (Singapura, Austin, México; 28.220 voltas em comum)

| Modelo | MAE (s) | RMSE (s) | Viés (s) | Erro em 5 voltas (s) | Cobertura 80% | Pinball (s) | Skill vs. ingênuo | MAE − GRU (IC 95%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo `naive_20261106` | 0,655 | 0,933 | +0,195 | 2,07 | — | — | — | +0,183 (+0,159 a +0,207) |
| Linear misto `lmm_20261120` | 0,540 | 0,783 | −0,062 | 1,88 | 86,9% | 0,184 | +17,6% | +0,068 (+0,048 a +0,088) |
| LightGBM `lgbm_20261120` | 0,498 | 0,729 | +0,030 | 1,63 | 81,2% | 0,165 | +23,9% | +0,026 (+0,015 a +0,038) |
| LSTM `lstm_20261120_ens5` | **0,457** | **0,689** | −0,009 | **1,46** | 87,4% | **0,157** | **+30,2%** | −0,015 (−0,022 a −0,008) |
| GRU `gru_20261106_ens5` | 0,472 | 0,700 | +0,012 | 1,51 | 87,7% | 0,159 | +28,0% | — |

As duas redes são ensembles de 5 sementes; a comparação entre as células, semente a semente, está em [LSTM × GRU](#lstm--gru).

MAE por horizonte (s):

| Modelo | Divisão | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo | validação | 0,53 | 0,62 | 0,69 | 0,75 | 0,80 | 0,84 | 0,87 | 0,90 | 0,94 | 0,98 |
| Linear misto | validação | 0,62 | 0,66 | 0,70 | 0,74 | 0,77 | 0,80 | 0,83 | 0,86 | 0,89 | 0,92 |
| LightGBM | validação | 0,44 | 0,50 | 0,54 | 0,57 | 0,60 | 0,62 | 0,65 | 0,67 | 0,70 | 0,72 |
| LSTM | validação | 0,43 | 0,48 | 0,52 | 0,55 | 0,58 | 0,60 | 0,62 | 0,64 | 0,65 | 0,67 |
| GRU | validação | 0,43 | 0,49 | 0,53 | 0,56 | 0,59 | 0,61 | 0,63 | 0,66 | 0,68 | 0,70 |
| Ingênuo | teste | 0,47 | 0,54 | 0,59 | 0,63 | 0,66 | 0,68 | 0,71 | 0,74 | 0,77 | 0,80 |
| Linear misto | teste | 0,45 | 0,49 | 0,51 | 0,53 | 0,54 | 0,55 | 0,57 | 0,59 | 0,59 | 0,60 |
| LightGBM | teste | 0,40 | 0,44 | 0,47 | 0,49 | 0,51 | 0,52 | 0,54 | 0,55 | 0,55 | 0,55 |
| LSTM | teste | 0,38 | 0,42 | 0,44 | 0,45 | 0,46 | 0,47 | 0,47 | 0,49 | 0,49 | 0,50 |
| GRU | teste | 0,38 | 0,42 | 0,45 | 0,47 | 0,48 | 0,49 | 0,50 | 0,51 | 0,52 | 0,53 |

MAE por corrida e por tarefa (s; tarefa A: mesmo stint da volta `t`; tarefa B: depois de uma parada):

| Modelo | Zandvoort | Monza | Madring | Baku | Singapura | Austin | México | A val. | B val. | A teste | B teste |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo | 0,823 | 0,568 | 0,988 | 0,723 | 0,763 | 0,691 | 0,521 | 0,706 | 1,340 | 0,553 | 1,315 |
| Linear misto | 0,701 | 0,552 | 1,064 | 0,774 | 0,677 | 0,492 | 0,450 | 0,718 | 1,167 | 0,521 | 0,662 |
| LightGBM | 0,541 | **0,456** | 0,825 | 0,560 | 0,657 | 0,468 | 0,372 | 0,569 | 0,802 | 0,487 | 0,572 |
| LSTM | 0,519 | 0,462 | **0,794** | **0,485** | **0,573** | **0,452** | **0,349** | **0,543** | **0,762** | **0,442** | 0,553 |
| GRU | **0,507** | 0,466 | 0,821 | 0,528 | 0,597 | 0,457 | 0,364 | 0,557 | 0,765 | 0,461 | **0,544** |

Leitura:
- **As redes recorrentes são as melhores, e o LightGBM fica perto.**
  - O ensemble da LSTM ganha da GRU por 0,013 s na validação e 0,015 s no teste.
  - A GRU ganha do LightGBM por 0,015 s e 0,026 s.
  - Os intervalos não cruzam o zero. O LightGBM só ganha em Monza; a GRU, em Zandvoort e depois das paradas no teste.
- **O linear misto ganha do ingênuo no teste (+17,6%) e quase empata na validação (+1,4%).**
  - Ele perde na volta seguinte (h1): o seu nível de piloto:corrida é a média de todas as voltas limpas até `t`, e não o ritmo das últimas voltas.
  - Ganha nos horizontes longos e depois da parada (tarefa B), porque sabe o efeito do pneu novo e do composto.

### Calibração dos quantis

A previsão quantílica existe desde a v0: a rede sai com p10, p50 e p90, treinada com a perda pinball, e p10 ≤ p50 ≤ p90 por construção. Se os quantis estão certos, 10% das voltas reais ficam abaixo do p10, 80% entre p10 e p90 e 10% acima do p90.

| Modelo | Divisão | Abaixo do p10 | Dentro de p10–p90 | Acima do p90 | Largura p10–p90 (s) | Pinball (s) |
|---|---|---:|---:|---:|---:|---:|
| Linear misto | validação | 13,8% | 75,2% | 11,0% | 2,03 | 0,265 |
| LightGBM | validação | 18,9% | 71,4% | 9,6% | 1,52 | 0,204 |
| LSTM | validação | 13,8% | 78,6% | 7,6% | 1,76 | **0,195** |
| GRU | validação | 13,3% | 79,3% | 7,3% | 1,84 | 0,198 |
| Linear misto | teste | 4,6% | 86,9% | 8,6% | 2,03 | 0,184 |
| LightGBM | teste | 10,8% | 81,2% | 8,1% | 1,68 | 0,165 |
| LSTM | teste | 5,8% | 87,4% | 6,8% | 1,77 | **0,157** |
| GRU | teste | 5,4% | 87,7% | 6,8% | 1,80 | 0,159 |

Cobertura p10–p90 por horizonte:

| Modelo | Divisão | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear misto | validação | 82% | 80% | 78% | 77% | 76% | 74% | 73% | 71% | 70% | 68% |
| LightGBM | validação | 78% | 75% | 73% | 71% | 71% | 70% | 70% | 69% | 68% | 66% |
| LSTM | validação | 87% | 83% | 80% | 79% | 78% | 77% | 77% | 76% | 75% | 75% |
| GRU | validação | 86% | 83% | 81% | 80% | 79% | 78% | 77% | 77% | 76% | 75% |
| Linear misto | teste | 91% | 89% | 89% | 88% | 87% | 86% | 85% | 85% | 84% | 84% |
| LightGBM | teste | 83% | 82% | 81% | 82% | 81% | 81% | 81% | 81% | 80% | 80% |
| LSTM | teste | 90% | 88% | 87% | 86% | 87% | 87% | 87% | 87% | 87% | 87% |
| GRU | teste | 90% | 88% | 88% | 87% | 88% | 88% | 87% | 87% | 87% | 87% |

Leitura:
- **As redes erram a cobertura por pouco, e em direções opostas nas duas divisões:** 79% na validação e 87–88% no teste. A pinball das redes é a menor nas duas.
- **A validação tem o lado de baixo curto:**
  - 13% das voltas saem mais rápidas que o p10 e só 7% mais lentas que o p90;
  - é o mesmo sinal do viés positivo (+0,06 a +0,09 s): nessas corridas o ritmo melhorou mais do que a rede esperava;
  - a cobertura cai com o horizonte, de 86% em h1 para 75% em h10.
- **No teste, a cobertura fica em ~87% em todos os horizontes:** os intervalos são um pouco largos.
- **Sem recalibração:** uma correção ajustada na validação alargaria os horizontes longos, que no teste já cobrem 87%. Fica a calibração do treino, que erra por até 8 pontos.
- **O LightGBM tem os intervalos mais estreitos.** Ele acerta a cobertura no teste (81%), mas não na validação (71%), em que 19% das voltas ficam abaixo do p10.

### Modelo linear misto (`modeling.lmm`)

Ajuste por REML (statsmodels `MixedLM`) nas 86.118 voltas limpas das 91 corridas de treino. O ritmo está em pontos percentuais: 1 pp ≈ 0,83 s numa volta de referência média de 83 s.

    ritmo (pp) = 8,25 + composto + idade × composto − 3,94 × progresso da corrida
                 + 0,027 × (temperatura da pista − média) + u_piloto:corrida + v_piloto:corrida × progresso + erro

| Termo | Estimativa | Leitura em segundos |
|---|---:|---|
| Idade do pneu, médio | +0,054 pp por volta | +0,045 s por volta de pneu |
| Idade do pneu, macio | +0,073 pp por volta | +0,061 s por volta, 36% mais que o médio |
| Idade do pneu, duro | +0,054 pp por volta | igual ao médio |
| Macio novo × médio novo | −0,24 pp | −0,20 s |
| Progresso da corrida (do início ao fim) | −3,94 pp | −3,3 s: combustível e pista emborrachando |
| Temperatura da pista | +0,027 pp por °C | +0,022 s por °C |
| Desvio do intercepto piloto:corrida | 2,75 pp | 2,3 s |
| Desvio da inclinação no progresso | 1,42 pp | 1,2 s ao longo da corrida |
| Desvio do erro | 0,83 pp | 0,69 s |

Ao vivo, o efeito aleatório de cada piloto:corrida é o BLUP com as voltas limpas (`clean_live`) até `t`, em forma fechada. Nas corridas de treino, com todas as voltas, ele coincide com o do statsmodels (diferença máxima de 4·10⁻¹²). Os quantis são os da normal, com a variância do efeito aleatório (dado o que já se viu) somada à do erro.

Variantes da parte aleatória (MAE em segundos de cada run, nas suas próprias voltas):

| Efeitos aleatórios por piloto:corrida | Validação | Teste |
|---|---:|---:|
| Intercepto (o do plano) | 0,799 | 0,544 |
| Intercepto + inclinação na idade do pneu | 0,858 | 0,639 |
| **Intercepto + inclinação no progresso da corrida** (escolhida) | **0,766** | **0,541** |
| Intercepto + as duas inclinações | 0,972 | 0,669 |

A inclinação no progresso deixa cada piloto ter a sua evolução ao longo da prova. A inclinação na idade do pneu piora: estimada com as poucas voltas de um stint, ela extrapola mal nas voltas seguintes.

### LightGBM (`modeling.lgbm`)

- **Previsão direta:** cada linha é uma volta prevista (`t`, `h`) e o horizonte é uma variável.
- **Alvo:** a diferença entre o ritmo da volta `t + h` e a média das 3 últimas voltas limpas até `t` (o ingênuo). Sem volta limpa ainda, a base é a mediana do treino.
- **37 variáveis**, todas calculáveis ao fim da volta `t`:
  - histórico do ritmo limpo: última volta, médias de 3 e 5, desvio de 5, média e número de voltas limpas no stint, inclinação das 5 últimas do stint contra a idade do pneu, melhor volta e voltas desde a última limpa;
  - estado na volta `t`: ritmo, posição, gaps, tráfego, bandeiras, clima, composto, idade do pneu e stint;
  - conhecidas da volta `t + h`, como na GRU: composto e número C, idade do pneu, volta do stint, pneu novo, progresso e voltas restantes, mesmo stint e paradas até lá;
  - circuito e regulamento.
- **Treino:** 781.815 linhas das corridas de treino com volta prevista limpa.
  - Um modelo por quantil (perda quantílica), 63 folhas, taxa de aprendizado 0,05.
  - Número de árvores escolhido por early stopping na validação: 94 (p10), 379 (p50) e 201 (p90).
  - Treina em ~2 min em CPU.
- **Variáveis mais importantes no p50** (ganho): circuito 15%, inclinação do stint 13%, mesmo stint 10%, média de 3 voltas 7%, desvio de 5 voltas 5%, voltas de pneu até `t + h` 4,5%.
- **Ablação do circuito:** sem ele, o MAE da validação piora de 0,597 para 0,626 s e o do teste melhora de 0,503 para 0,489 s.
  - Pela validação, o circuito fica.
  - Sem ele, o clima (temperatura do ar, umidade) sobe no ranking, como um substituto do circuito.

## LSTM × GRU

`modeling.cells` treina a mesma rede encoder–decoder com a célula trocada (`modeling.gru --cell lstm`). São os mesmos dados (divisão de 3 corridas), a mesma taxa (0,001), o mesmo treino e 5 sementes por configuração. As GRUs vêm da bateria `sweep_20261106`. Relatório completo em `runs/cells_20261120/report.md`.

- **Estado da LSTM:** é o par (h, c). O decoder de cada volta parte dos dois. Como a `nn.LSTM` só devolve o h de cada passo, o encoder da LSTM anda volta a volta para guardar o c.
- **Configurações testadas:**
  - estado 32, a mesma da GRU escolhida. A LSTM tem uma porta a mais e fica com 34% mais parâmetros;
  - estado 26, o mesmo número de parâmetros da GRU (~15 mil);
  - duas vizinhas (estado 64; dropout 0,2), para a LSTM não ficar presa à configuração ótima da GRU.
- **Critério de escolha:** a média da pinball na validação, o mesmo da GRU.

**Validação: a escolha** (média ± desvio entre 5 sementes)

| Célula | Estado | Dropout | Parâmetros | Pinball validação (s) | MAE validação (s) | Cobertura validação | Melhor época |
|---|---:|---:|---:|---|---|---|---:|
| LSTM | 32 | 0,1 | 19.957 | **0,206 ± 0,008** (escolhida) | 0,606 ± 0,026 | 76,6% ± 3,2 | 20,2 |
| GRU | 32 | 0,1 | 14.901 | 0,207 ± 0,007 (escolhida) | 0,610 ± 0,022 | 77,4% ± 2,7 | 16,0 |
| LSTM | 32 | 0,2 | 19.957 | 0,207 ± 0,009 | 0,608 ± 0,024 | 78,2% ± 2,4 | 19,8 |
| GRU | 32 | 0,2 | 14.901 | 0,208 ± 0,007 | 0,612 ± 0,019 | 79,2% ± 2,1 | 14,6 |
| GRU | 64 | 0,1 | 41.365 | 0,211 ± 0,004 | 0,624 ± 0,013 | 75,3% ± 2,5 | 10,0 |
| LSTM | 26 | 0,1 | 15.103 | 0,213 ± 0,008 | 0,630 ± 0,032 | 76,8% ± 2,7 | 17,8 |
| LSTM | 64 | 0,1 | 55.573 | 0,214 ± 0,010 | 0,629 ± 0,027 | 76,2% ± 3,5 | 9,4 |

**Teste, por semente** (Singapura, Austin, México)

| Configuração | MAE teste (s) | Skill vs. ingênuo | Cobertura teste | Pinball teste (s) | Depois de parada (s) |
|---|---|---|---|---|---|
| GRU escolhida (estado 32) | 0,496 ± 0,014 | 25,1% ± 2,1 | 85,5% ± 1,8 | 0,166 ± 0,004 | 0,565 ± 0,014 |
| LSTM escolhida (estado 32) | 0,488 ± 0,020 | 26,1% ± 3,2 | 85,3% ± 1,1 | 0,164 ± 0,007 | 0,571 ± 0,037 |
| LSTM com os parâmetros da GRU (estado 26) | 0,496 ± 0,016 | 24,9% ± 2,6 | 85,7% ± 1,9 | 0,168 ± 0,005 | 0,572 ± 0,035 |

Diferença de MAE, LSTM escolhida − GRU escolhida:
- validação: −0,004 s, erro-padrão entre sementes 0,015 s;
- teste: −0,008 s, erro-padrão entre sementes 0,011 s.

**Ensembles de 5 sementes** (média dos quantis), nas mesmas voltas, com o bootstrap por piloto:corrida de `modeling.compare`:

| Divisão | Modelo | MAE (s) | Pinball (s) | Cobertura 80% | Erro em 5 voltas (s) | MAE − GRU (IC 95%) |
|---|---|---:|---:|---:|---:|---:|
| validação | LSTM `lstm_20261120_ens5` | **0,569** | **0,195** | 78,6% | **1,84** | −0,013 (−0,023 a −0,004) |
| validação | GRU `gru_20261106_ens5` | 0,582 | 0,198 | 79,3% | 1,92 | — |
| teste | LSTM `lstm_20261120_ens5` | **0,457** | **0,157** | 87,4% | **1,46** | −0,015 (−0,022 a −0,008) |
| teste | GRU `gru_20261106_ens5` | 0,472 | 0,159 | 87,7% | 1,51 | — |
| São Paulo (reprodução) | LSTM | **1,001** | | 61,0% | | |
| São Paulo (reprodução) | GRU | 1,037 | | 57,8% | | |

A linha de São Paulo roda o previsor ao vivo (`live.predict`) sobre o histórico final da corrida, com cada ensemble, nas 7.661 voltas limpas previstas pelos dois. O número da GRU difere do relatório da corrida (1,115 s), que usa as previsões feitas durante a prova.

**Tempo por época** (um treino de cada vez, CPU): GRU 32, 4,8 s; LSTM 32, 7,7 s; LSTM 26, 6,6 s. A LSTM também para mais tarde: a melhor época média é 20 (contra 16 da GRU). O treino de uma semente leva ~3,5 min, contra ~2 min da GRU.

Leitura:
- **Uma rede sozinha: empate.** A diferença entre as escolhidas é menor que a variação entre sementes. Com o mesmo número de parâmetros, a LSTM fica um pouco atrás na validação.
- **Ensemble: a LSTM ganha por pouco e de forma consistente.** Fica 0,013 s melhor na validação e 0,015 s no teste, com os intervalos fora do zero, e também melhor na reprodução de São Paulo.
  - A média das sementes ajuda mais a LSTM: ganha 0,037 s sobre a semente média na validação, contra 0,028 s da GRU. As sementes dela variam mais entre si.
  - O bootstrap reamostra pilotos, não sementes. Com outro sorteio de 5 sementes, a diferença pode mudar de tamanho.
- **Custo:** a LSTM treina ~60% mais devagar por época e para algumas épocas depois.

## Eras de regulamento

A validação e o teste são corridas de 2026, o primeiro ano do regulamento novo. No treino, 1.594 sequências piloto:corrida são de 2022–25 e só 241 de 2026, de 11 corridas. `modeling.eras` mede como usar as temporadas antigas. Usa a GRU escolhida (estado 32, dropout 0,1, taxa 0,001), 5 sementes por estratégia e escolha pela média da pinball na validação. Relatório completo em `runs/eras_20261120/report.md`.

- **Estratégias:**
  - `pooled`: todas as eras juntas, com o embedding de regulamento. É o modelo em uso, da bateria `sweep_20261106`;
  - `pooled_noreg`: todas as eras, com o regulamento sempre em `<unk>`. Mede se o embedding ajuda;
  - `2026_only`: só as 11 corridas de 2026 do treino;
  - `pre_2022_25`: só 2022–25, sem nenhuma corrida de 2026. Mede a transferência direta para a era nova;
  - `pretrain_finetune`: parte do `pre_2022_25` da mesma semente e continua só em 2026, com taxa 0,0003;
  - `pooled_finetune`: parte do `pooled` da mesma semente e continua só em 2026.
- **Categorias que o run não viu no treino viram `<unk>`, como ao vivo.**
  - No `pre_2022_25` são as 11 equipes-ano de 2026, o regulamento de 2026 e um piloto estreante. No `2026_only`, 14 circuitos, 10 pilotos e as equipes-ano de 2022–25.
  - Sem isso, essas categorias entrariam com o embedding aleatório do início. O `<unk>` é treinado pelo dropout de categorias, que apaga 10% delas no treino.
  - A lista vai no `config.json` (`unk_categories`), e o previsor ao vivo aplica a de cada membro do ensemble.
- **Com 241 sequências, cada época tem só 8 lotes.** Por isso, no `2026_only` e nos fine-tunings, a paciência sobe para 20 épocas e o limite para 300.

**Validação: a escolha** (média ± desvio entre 5 sementes)

| Estratégia | Sequências de treino | Pinball validação (s) | MAE validação (s) | Cobertura validação | Melhor época |
|---|---:|---|---|---|---:|
| `pooled_noreg`: todas as eras, sem embedding de regulamento | 1.835 | **0,202 ± 0,005** (escolhida) | **0,588 ± 0,009** | 75,9% ± 3,0 | 18,6 |
| `pre_2022_25`: só 2022–25 | 1.594 | 0,205 ± 0,008 | 0,597 ± 0,019 | 74,6% ± 2,3 | 22,4 |
| `pooled`: todas as eras, com embedding de regulamento (em uso) | 1.835 | 0,207 ± 0,007 | 0,610 ± 0,022 | 77,4% ± 2,7 | 16,0 |
| `pretrain_finetune`: 2022–25 + fine-tuning em 2026 | 241 | 0,213 ± 0,010 | 0,630 ± 0,032 | 75,7% ± 2,0 | 1,0 |
| `pooled_finetune`: todas as eras + fine-tuning em 2026 | 241 | 0,224 ± 0,015 | 0,664 ± 0,053 | 72,5% ± 3,4 | 1,0 |
| `2026_only`: só 2026 | 241 | 0,357 ± 0,033 | 1,085 ± 0,153 | 64,2% ± 12,5 | 23,6 |

Nos fine-tunings, as sequências e a melhor época são as da etapa em 2026.

**Teste, por semente** (Singapura, Austin, México)

| Estratégia | MAE teste (s) | Cobertura teste | Pinball teste (s) | Depois de parada (s) |
|---|---|---|---|---|
| `pooled_noreg` | **0,485 ± 0,013** | 84,0% ± 2,0 | **0,163 ± 0,004** | **0,547 ± 0,024** |
| `pre_2022_25` | 0,490 ± 0,030 | 82,6% ± 2,2 | 0,164 ± 0,008 | 0,553 ± 0,026 |
| `pooled` (em uso) | 0,496 ± 0,014 | 85,5% ± 1,8 | 0,166 ± 0,004 | 0,565 ± 0,014 |
| `pretrain_finetune` | 0,513 ± 0,027 | 83,3% ± 2,3 | 0,170 ± 0,007 | 0,574 ± 0,027 |
| `pooled_finetune` | 0,516 ± 0,020 | 83,3% ± 1,1 | 0,170 ± 0,006 | 0,594 ± 0,024 |
| `2026_only` | 0,672 ± 0,143 | 88,2% ± 4,3 | 0,228 ± 0,051 | 0,755 ± 0,202 |

**Ensembles de 5 sementes** (média dos quantis), nas mesmas voltas. O bootstrap por piloto:corrida de `modeling.compare` dá a diferença para o modelo em uso. A última coluna roda o previsor ao vivo sobre o histórico final de São Paulo, nas 7.661 voltas limpas previstas por todos.

| Estratégia | MAE val. (s) | MAE − em uso, val. (IC 95%) | MAE teste (s) | MAE − em uso, teste (IC 95%) | Cobertura val. / teste | São Paulo: MAE (s) e cobertura |
|---|---:|---:|---:|---:|---:|---:|
| `pooled_noreg` | **0,565** | −0,017 (−0,024 a −0,010) | **0,463** | −0,009 (−0,014 a −0,005) | 77,8% / 86,3% | 1,062 · 50,3% |
| `pre_2022_25` | 0,572 | −0,011 (−0,021 a −0,000) | 0,465 | −0,007 (−0,011 a −0,002) | 76,7% / 84,8% | 1,136 · 41,3% |
| `pooled` `gru_20261106_ens5` (em uso) | 0,582 | — | 0,472 | — | 79,3% / 87,7% | 1,037 · 57,8% |
| `pretrain_finetune` | 0,603 | +0,021 (+0,015 a +0,027) | 0,487 | +0,015 (+0,007 a +0,023) | 78,3% / 85,9% | 1,085 · 52,4% |
| `pooled_finetune` | 0,638 | +0,056 (+0,045 a +0,066) | 0,496 | +0,024 (+0,016 a +0,032) | 75,0% / 85,5% | 1,028 · 59,1% |
| `2026_only` | 1,011 | +0,428 (+0,370 a +0,494) | 0,592 | +0,120 (+0,088 a +0,153) | 71,3% / 92,6% | **0,954 · 75,1%** |

MAE por corrida dos ensembles (s):

| Estratégia | Zandvoort | Monza | Madring | Baku | Singapura | Austin | México |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pooled_noreg` | **0,505** | 0,464 | **0,800** | 0,476 | **0,579** | 0,460 | **0,353** |
| `pre_2022_25` | 0,511 | **0,463** | 0,826 | **0,465** | 0,599 | **0,446** | **0,353** |
| `pooled` (em uso) | 0,507 | 0,466 | 0,821 | 0,528 | 0,597 | 0,457 | 0,364 |
| `pretrain_finetune` | 0,514 | 0,478 | 0,861 | 0,557 | 0,637 | 0,459 | 0,366 |
| `pooled_finetune` | 0,510 | 0,518 | 0,899 | 0,637 | 0,647 | 0,455 | 0,385 |
| `2026_only` | 0,753 | 1,173 | 1,056 | 1,136 | 0,685 | 0,519 | 0,564 |

**Fine-tuning com taxas menores.** Os dois fine-tunings rodaram também com taxas de 0,0001 e 0,00003, 5 sementes cada. Cada linha parte dos mesmos 5 runs, e "sem fine-tuning" é o próprio ponto de partida. A melhor época é a do fine-tuning: o treino começa na época 1 e não avalia o ponto de partida.

| Ponto de partida | Taxa do fine-tuning | Pinball validação (s) | MAE validação (s) | MAE teste (s) | Melhor época |
|---|---|---|---|---|---:|
| `pre_2022_25` | sem fine-tuning | **0,205 ± 0,008** | **0,597 ± 0,019** | **0,490 ± 0,030** | — |
| | 0,0003 | 0,213 ± 0,010 | 0,630 ± 0,032 | 0,513 ± 0,027 | 1,0 |
| | 0,0001 | 0,212 ± 0,009 | 0,624 ± 0,033 | 0,521 ± 0,034 | 2,4 |
| | 0,00003 | 0,213 ± 0,010 | 0,626 ± 0,034 | 0,523 ± 0,032 | 7,4 |
| `pooled` (em uso) | sem fine-tuning | **0,207 ± 0,007** | **0,610 ± 0,022** | **0,496 ± 0,014** | — |
| | 0,0003 | 0,224 ± 0,015 | 0,664 ± 0,053 | 0,516 ± 0,020 | 1,0 |
| | 0,0001 | 0,219 ± 0,008 | 0,650 ± 0,031 | 0,514 ± 0,015 | 2,0 |
| | 0,00003 | 0,212 ± 0,006 | 0,626 ± 0,021 | 0,505 ± 0,011 | 1,0 |

Leitura:
- **Treinar só com 2026 não funciona.** Com 11 corridas, a rede erra mais que o ingênuo na validação: 1,09 s de MAE contra 0,78 s. As sementes variam muito (0,95 a 1,31 s), e duas param nas épocas 4 e 5.
- **2022–25 sozinhas já dão uma rede tão boa quanto a em uso.**
  - Sem nenhuma corrida de 2026, a pinball de validação é 0,205, contra 0,207 da em uso. O ensemble ganha por 0,011 s na validação (IC até −0,000) e por 0,007 s no teste.
  - O alvo é o ritmo relativo à volta de referência do quali do mesmo fim de semana. Provavelmente é ela que absorve a mudança de velocidade entre as eras; o desgaste do pneu e o efeito do combustível se parecem nos dois regulamentos.
- **O embedding de regulamento não ajuda, e sem ele a rede fica um pouco melhor.** `pooled_noreg` é a escolhida pela validação.
  - Pinball 0,202 contra 0,207, melhor em 4 das 5 sementes. O ensemble ganha por 0,017 s na validação e 0,009 s no teste, com os intervalos fora do zero.
  - Por semente, no teste, a diferença vai de −0,040 a +0,017 s. O bootstrap reamostra pilotos, não sementes.
  - Uma leitura possível: com o embedding, as 241 sequências de 2026 formam um grupo à parte, e a rede se ajusta a elas. Sem ele, entram como mais dados do mesmo problema.
- **O fine-tuning em 2026 piora a rede, em todas as taxas testadas.**
  - Em geral, a perda de validação sobe desde a primeira época: a rede se ajusta às 11 corridas de treino e perde nas 4 de validação. Na média das sementes, nenhum fine-tuning volta ao nível do ponto de partida.
  - O ponto de partida carrega certo: antes do primeiro passo, a perda de validação do `pooled` é a mesma do run de origem.
  - No `pretrain_finetune` há um custo a mais. As equipes-ano e o regulamento de 2026, que eram `<unk>` no ponto de partida, passam a ter embeddings próprios, que começam aleatórios. Só essa troca, sem nenhum passo, já piora a perda de validação em 3 das 5 sementes, até 25% numa delas.
- **São Paulo inverte a ordem: lá, quanto mais peso em 2026, melhor.**
  - Na reprodução com o previsor ao vivo, o só 2026 é o melhor (0,954 s, cobertura de 75%). Depois vêm o fine-tuning do `pooled` (1,028 s) e o modelo em uso (1,037 s). O sem regulamento fica em 1,062 s e o só 2022–25 em 1,136 s.
  - É uma corrida só, fora do protocolo e atípica. Com o quali molhado, a referência do alvo veio dos treinos livres, e as voltas são as do fluxo ao vivo.
  - Mesmo assim, mostra que a vantagem do `pooled_noreg` é pequena e que a ordem pode mudar de uma corrida para outra.
- **O ensemble sem regulamento** está em `runs/eras_20261120/pooled_noreg_ens5` e roda com `--run eras_20261120/pooled_noreg_ens5`.
- **O modelo ao vivo da v1 é a LSTM com regulamento.** Ela foi escolhida depois do fatorial [célula × regulamento](#célula--regulamento).

### Célula × regulamento

Tirar o regulamento ajudou a GRU, e a LSTM tinha ganhado da GRU. Para ver se os dois ganhos se somam, a LSTM escolhida (estado 32, dropout 0,1) foi treinada sem o embedding de regulamento, com 5 sementes (`modeling.eras --cell lstm --strategies pooled_noreg`, em `runs/eras_20261120_lstm/`). As outras três casas do fatorial já existiam. A regra foi fixada antes do treino: ganha a casa com a menor média da pinball de validação por semente.

**Por semente** (média ± desvio entre 5 sementes)

| Célula | Regulamento | Pinball validação (s) | MAE validação (s) | Cobertura validação | MAE teste (s) | Cobertura teste |
|---|---|---|---|---|---|---|
| GRU | com (em uso) | 0,207 ± 0,007 | 0,610 ± 0,022 | 77,4% ± 2,7 | 0,496 ± 0,014 | 85,5% ± 1,8 |
| GRU | sem | **0,202 ± 0,005** (escolhida) | **0,588 ± 0,009** | 75,9% ± 3,0 | **0,485 ± 0,013** | 84,0% ± 2,0 |
| LSTM | com | 0,206 ± 0,008 | 0,606 ± 0,026 | 76,6% ± 3,2 | 0,488 ± 0,020 | 85,3% ± 1,1 |
| LSTM | sem | 0,209 ± 0,004 | 0,608 ± 0,010 | 74,1% ± 1,7 | 0,498 ± 0,015 | 82,4% ± 1,9 |

**Efeito de tirar o regulamento, pareado por semente** (sem − com, entre parênteses o erro-padrão). Com a mesma semente, a rede começa dos mesmos pesos e vê os lotes na mesma ordem; só muda a coluna de regulamento.

| Célula | Pinball validação (s) | MAE validação (s) | MAE teste (s) | Sementes em que a pinball de validação melhora |
|---|---:|---:|---:|---:|
| GRU | −0,0047 (0,0019) | −0,021 (0,006) | −0,010 (0,010) | 4 de 5 |
| LSTM | +0,0029 (0,0031) | +0,002 (0,011) | +0,010 (0,013) | 2 de 5 |
| Interação (LSTM − GRU) | +0,0076 (0,0037) | +0,023 (0,013) | +0,021 (0,017) | |

**Ensembles de 5 sementes**, nas mesmas voltas (`runs/compare_20261120_eras2x2/`). A última coluna é a reprodução de São Paulo com o previsor ao vivo, nas 7.661 voltas limpas previstas por todos.

| Célula | Regulamento | Pinball val. (s) | MAE val. (s) | MAE − em uso, val. (IC 95%) | Pinball teste (s) | MAE teste (s) | MAE − em uso, teste (IC 95%) | Cobertura val. / teste | São Paulo: MAE (s) e cobertura |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | com (em uso) | 0,1985 | 0,582 | — | 0,1593 | 0,472 | — | 79,3% / 87,7% | 1,037 · 57,8% |
| GRU | sem | **0,1951** | **0,565** | −0,017 (−0,024 a −0,010) | **0,1565** | 0,463 | −0,009 (−0,014 a −0,005) | 77,8% / 86,3% | 1,062 · 50,3% |
| LSTM | com | 0,1953 | 0,569 | −0,013 (−0,023 a −0,004) | 0,1566 | **0,457** | −0,015 (−0,022 a −0,008) | 78,6% / 87,4% | **1,001 · 61,0%** |
| LSTM | sem | 0,1994 | 0,581 | −0,001 (−0,012 a +0,010) | 0,1575 | 0,463 | −0,009 (−0,017 a −0,000) | 75,9% / 85,0% | 1,018 · 55,9% |

Comparação direta das duas melhores (`runs/compare_20261120_lstm_vs_noreg/`): o MAE da LSTM com regulamento menos o da GRU sem regulamento é +0,004 s na validação (IC −0,004 a +0,011) e −0,006 s no teste (−0,012 a +0,001).

**Tempo ao vivo**, na reprodução de São Paulo (CPU, 5 membros): 53–56 ms por previsão na GRU e 74–85 ms na LSTM. Para 20 pilotos, dá menos de 2 s por volta, contra voltas de 1,5 min.

Leitura:
- **Os ganhos não se somam.** Sem o regulamento, a LSTM piora:
  - a pinball de validação sobe em 3 das 5 sementes;
  - o ensemble perde a vantagem sobre a GRU em uso na validação (−0,001 s, IC de −0,012 a +0,010);
  - a cobertura cai mais (74% na validação, por semente).
- **O efeito do regulamento depende da célula.** A interação é de +0,008 na pinball (erro-padrão 0,004), maior que o próprio ganho na GRU. O ganho de tirar o regulamento é consistente na GRU (4 das 5 sementes, 5 das 5 no MAE de validação), mas não aparece na LSTM. Não é uma propriedade dos dados que valha para as duas redes.
- **Pela regra, a escolhida é a GRU sem regulamento:** 0,202 de pinball por semente, contra 0,206 da LSTM com regulamento.
- **Nos ensembles, as duas melhores empatam.**
  - A pinball de validação é 0,1951 contra 0,1953. As diferenças de MAE entre elas cruzam o zero na validação e no teste.
  - A LSTM com regulamento tem a cobertura mais perto de 80% na validação e ganha em São Paulo (1,001 s contra 1,062 s), mas São Paulo fica fora do protocolo.
- **Também foi avaliado um ensemble misto** com as 10 redes (as 5 GRUs sem regulamento e as 5 LSTMs com regulamento, média dos quantis).
  - Ele teve MAE de 0,562 s e pinball de 0,1937 na validação, e MAE de 0,455 s e pinball de 0,1554 no teste. Em São Paulo, deu 0,992 s.
  - O ganho sobre cada uma das duas é de 0,002 a 0,007 s. Parte dele pode vir só de ter o dobro de membros.
  - Não foi adotado.

**Modelo ao vivo da v1: a LSTM com regulamento (`lstm_20261120_ens5`)**, decidido em 21/11/2026.
- **Por que ela:** nos ensembles, que são o que vai ao vivo, ela empata com a GRU sem regulamento na validação. Ela tem o menor MAE de teste, a cobertura mais perto de 80% e o melhor resultado em São Paulo.
- **Ela não segue a regra por semente,** que apontava a GRU sem regulamento (0,202 contra 0,206).
- **Como roda:** é o `LIVE_RUN` de `live/predict.py`, o padrão do `python -m live.predict`. Os outros modelos rodam com `--run`. A reprodução de São Paulo com o padrão dá 1,001 s, igual à da tabela.
- **Leitura das tabelas acima:** "em uso" é a GRU com regulamento (`gru_20261106_ens5`), o modelo ao vivo até 20/11.

## Duelos

`modeling/duels.py` projeta o gap entre dois carros com as previsões da GRU e dá a volta em que o carro de trás deve chegar a menos de 1 s do da frente (a disputa), com uma janela de 80%. `modeling/duels_backtest.py` mede isso nas corridas de validação e teste e nas previsões feitas ao vivo em São Paulo, contra o ingênuo (ritmo médio das 3 últimas voltas limpas de cada um), e grava `runs/duels_<data>/metrics.json`.

| Corridas | Situações | Disputas reais | Precisão GRU (ingênuo) | Recall GRU (ingênuo) | A ±1 volta GRU (ingênuo) | Na janela de 80% | Brier GRU (taxa fixa) |
|---|---|---|---|---|---|---|---|
| Validação (Zandvoort, Monza, Madring, Baku) | 809 | 41% | 80% (74%) | 55% (55%) | 75% (70%) | 80% | 0,163 (0,241) |
| Teste (Singapura, Austin, México) | 515 | 38% | 82% (75%) | 78% (71%) | 86% (76%) | 93% | 0,120 (0,235) |
| São Paulo, previsões feitas ao vivo, pneu liso | 160 | 50% | 84% (78%) | 66% (65%) | 74% (73%) | 80% | 0,166 (0,250) |

Precisão: das disputas previstas nas 10 voltas seguintes (probabilidade de 50% ou mais), quantas aconteceram. Recall: das disputas que aconteceram, quantas foram previstas. "A ±1 volta": entre as previstas que aconteceram, a parte cuja volta real ficou a até 1 volta da prevista. A largura da janela foi escolhida na validação.

### Las Vegas: duelos previstos e duelos em curso (`modeling/duel_replay.py`)

O replay da aba Replay usa a mesma conta com as previsões da LSTM com regulamento (`lstm_20261120_ens5`). São as feitas ao vivo, mais as perdidas na queda do previsor, refeitas com `live.predict --once`. A gravação começou na volta 12, e o desfecho vem dos tempos oficiais (TracingInsights). Aqui só se medem as previsões de duelo, então não há recall nem Brier dos duelos que começam.

| Tipo | Situações (voltas) | Previstas | Sem desfecho | Resultado |
|---|---|---|---|---|
| Duelo que começa (1–5 s; a rede prevê a chegada a menos de 1 s) | 158 | 45 | 3 | 86% aconteceram, 94% delas dentro da janela de 80% |
| Duelo em curso (menos de 1 s; a rede prevê se o de trás passa) | 318 | 114 com ultrapassagem e 156 sem | 48 | 66% das ultrapassagens previstas aconteceram; 43% dos "segura" se confirmaram; acerto de 53% |

- **Duelo em curso:** a chance de ultrapassagem é P(gap na linha abaixo de 0) na mesma normal da projeção.
- **Contra a taxa base:** em 61% das 270 situações com desfecho, o carro de trás passou nas 10 voltas seguintes. Prever sempre ultrapassagem acertaria mais que a rede. O Brier da chance foi de 0,251, contra 0,238 da taxa fixa.
- **Por faixa de chance** (263 voltas):
  - **Extremos:** só eles funcionam. Acima de 80%, os 21 casos passaram; abaixo de 20%, 28% passaram.
  - **Meio:** no meio, a taxa real não acompanha a chance.
  - **Gap:** o gap na linha informa mais que a chance da rede. Com menos de 0,3 s, 76% passaram; com 0,6–1 s, 47%.
- **Leitura:** a rede prevê o ritmo de cada piloto sem o outro. Ar sujo, DRS e a dificuldade de passar não entram, e `SIGMA_SCALE` foi calibrado para a chegada a 1 s. Para os duelos em curso, a projeção não é um modelo de ultrapassagem.

## Relatórios de corrida

Cada corrida com a rede ao vivo tem um relatório em `docs/race_reports/<ano>_<circuito>.md`, com as figuras na pasta de mesmo nome. Ele traz as métricas por fase da corrida e estudos de caso.

| Corrida | Modelo | MAE GRU | MAE ingênuo | Cobertura p10–p90 | Relatório |
|---|---|---|---|---|---|
| GP de São Paulo, 8/11/2026 | `gru_20261106_ens5` | 1,115 s | 1,534 s | 50% | [2026_sao_paulo.md](race_reports/2026_sao_paulo.md) |

As medidas da tabela usam as voltas limpas (`clean`), que só incluem pneu liso. Em corridas com chuva, o relatório avalia as voltas de pneu de chuva à parte, com todas as voltas válidas: com tempo, fora do box, sem SC e sem volta apagada.
