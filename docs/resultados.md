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
  recording.jsonl      fluxo bruto do f1-dash (/api/realtime) no formato do simulador do f1-dash;
                       uma parte nova (recording_2.jsonl…) a cada reinício do gravador
  laps.parquet         histórico de voltas montado pelo live.reader, com os nomes de coluna de race_laps.parquet
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
| LSTM: a mesma rede com `--cell lstm`, 5 sementes da configuração escolhida na validação (`modeling.cells`, feita em 20/11) | `lstm_<data>_ens5` | ✅ |
| Modelo linear misto: composto × idade do pneu, progresso da corrida e temperatura, com intercepto e inclinação no progresso aleatórios por piloto:corrida (`modeling.lmm`, feito em 20/11) | `lmm_<data>` | ✅ |
| LightGBM com lags do ritmo, um modelo por quantil (`modeling.lgbm`, feito em 20/11) | `lgbm_<data>` | ✅ |
| Estratégias de eras `2026_only` e `pretrain_finetune` | | v1 |
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
| GRU `gru_20261106_ens5` | **0,582** | **0,863** | +0,089 | **1,92** | 79,3% | **0,198** | **+25,6%** | — |

**Teste** (Singapura, Austin, México; 28.220 voltas em comum)

| Modelo | MAE (s) | RMSE (s) | Viés (s) | Erro em 5 voltas (s) | Cobertura 80% | Pinball (s) | Skill vs. ingênuo | MAE − GRU (IC 95%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo `naive_20261106` | 0,655 | 0,933 | +0,195 | 2,07 | — | — | — | +0,183 (+0,159 a +0,207) |
| Linear misto `lmm_20261120` | 0,540 | 0,783 | −0,062 | 1,88 | 86,9% | 0,184 | +17,6% | +0,068 (+0,048 a +0,088) |
| LightGBM `lgbm_20261120` | 0,498 | 0,729 | +0,030 | 1,63 | 81,2% | 0,165 | +23,9% | +0,026 (+0,015 a +0,038) |
| GRU `gru_20261106_ens5` | **0,472** | **0,700** | +0,012 | **1,51** | 87,7% | **0,159** | **+28,0%** | — |

MAE por horizonte (s):

| Modelo | Divisão | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo | validação | 0,53 | 0,62 | 0,69 | 0,75 | 0,80 | 0,84 | 0,87 | 0,90 | 0,94 | 0,98 |
| Linear misto | validação | 0,62 | 0,66 | 0,70 | 0,74 | 0,77 | 0,80 | 0,83 | 0,86 | 0,89 | 0,92 |
| LightGBM | validação | 0,44 | 0,50 | 0,54 | 0,57 | 0,60 | 0,62 | 0,65 | 0,67 | 0,70 | 0,72 |
| GRU | validação | 0,43 | 0,49 | 0,53 | 0,56 | 0,59 | 0,61 | 0,63 | 0,66 | 0,68 | 0,70 |
| Ingênuo | teste | 0,47 | 0,54 | 0,59 | 0,63 | 0,66 | 0,68 | 0,71 | 0,74 | 0,77 | 0,80 |
| Linear misto | teste | 0,45 | 0,49 | 0,51 | 0,53 | 0,54 | 0,55 | 0,57 | 0,59 | 0,59 | 0,60 |
| LightGBM | teste | 0,40 | 0,44 | 0,47 | 0,49 | 0,51 | 0,52 | 0,54 | 0,55 | 0,55 | 0,55 |
| GRU | teste | 0,38 | 0,42 | 0,45 | 0,47 | 0,48 | 0,49 | 0,50 | 0,51 | 0,52 | 0,53 |

MAE por corrida e por tarefa (s; tarefa A: mesmo stint da volta `t`; tarefa B: depois de uma parada):

| Modelo | Zandvoort | Monza | Madring | Baku | Singapura | Austin | México | A val. | B val. | A teste | B teste |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ingênuo | 0,823 | 0,568 | 0,988 | 0,723 | 0,763 | 0,691 | 0,521 | 0,706 | 1,340 | 0,553 | 1,315 |
| Linear misto | 0,701 | 0,552 | 1,064 | 0,774 | 0,677 | 0,492 | 0,450 | 0,718 | 1,167 | 0,521 | 0,662 |
| LightGBM | 0,541 | **0,456** | 0,825 | 0,560 | 0,657 | 0,468 | 0,372 | 0,569 | 0,802 | 0,487 | 0,572 |
| GRU | **0,507** | 0,466 | **0,821** | **0,528** | **0,597** | **0,457** | **0,364** | **0,557** | **0,765** | **0,461** | **0,544** |

Leitura:
- **A GRU é a melhor em quase tudo, mas o LightGBM fica perto:** 0,015 s atrás na validação e 0,026 s no teste. A diferença é pequena e consistente: o intervalo não cruza o zero nas duas divisões, e o LightGBM só ganha em Monza.
- **O linear misto ganha do ingênuo no teste (+17,6%) e quase empata na validação (+1,4%).**
  - Ele perde na volta seguinte (h1): o seu nível de piloto:corrida é a média de todas as voltas limpas até `t`, e não o ritmo das últimas voltas.
  - Ganha nos horizontes longos e depois da parada (tarefa B), porque sabe o efeito do pneu novo e do composto.
- **Intervalos:** os três modelos ficam perto de 80% de cobertura, com desvios diferentes.
  - O LightGBM é estreito demais na validação (71%) e deixa 19% das voltas abaixo do p10.
  - O linear misto e a GRU são um pouco largos no teste (87–88%).
  - A pinball, que junta largura e cobertura, segue a ordem do MAE.

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

## Duelos

`modeling/duels.py` projeta o gap entre dois carros com as previsões da GRU e dá a volta em que o carro de trás deve chegar a menos de 1 s do da frente (a disputa), com uma janela de 80%. `modeling/duels_backtest.py` mede isso nas corridas de validação e teste e nas previsões feitas ao vivo em São Paulo, contra o ingênuo (ritmo médio das 3 últimas voltas limpas de cada um), e grava `runs/duels_<data>/metrics.json`.

| Corridas | Situações | Disputas reais | Precisão GRU (ingênuo) | Recall GRU (ingênuo) | A ±1 volta GRU (ingênuo) | Na janela de 80% | Brier GRU (taxa fixa) |
|---|---|---|---|---|---|---|---|
| Validação (Zandvoort, Monza, Madring, Baku) | 809 | 41% | 80% (74%) | 55% (55%) | 75% (70%) | 80% | 0,163 (0,241) |
| Teste (Singapura, Austin, México) | 515 | 38% | 82% (75%) | 78% (71%) | 86% (76%) | 93% | 0,120 (0,235) |
| São Paulo, previsões feitas ao vivo, pneu liso | 160 | 50% | 84% (78%) | 66% (65%) | 74% (73%) | 80% | 0,166 (0,250) |

Precisão: das disputas previstas nas 10 voltas seguintes (probabilidade de 50% ou mais), quantas aconteceram. Recall: das disputas que aconteceram, quantas foram previstas. "A ±1 volta": entre as previstas que aconteceram, a parte cuja volta real ficou a até 1 volta da prevista. A largura da janela foi escolhida na validação.

## Relatórios de corrida

Cada corrida com a rede ao vivo tem um relatório em `docs/race_reports/<ano>_<circuito>.md`, com as figuras na pasta de mesmo nome. Ele traz as métricas por fase da corrida e estudos de caso.

| Corrida | Modelo | MAE GRU | MAE ingênuo | Cobertura p10–p90 | Relatório |
|---|---|---|---|---|---|
| GP de São Paulo, 8/11/2026 | `gru_20261106_ens5` | 1,115 s | 1,534 s | 50% | [2026_sao_paulo.md](race_reports/2026_sao_paulo.md) |

As medidas da tabela usam as voltas limpas (`clean`), que só incluem pneu liso. Em corridas com chuva, o relatório avalia as voltas de pneu de chuva à parte, com todas as voltas válidas: com tempo, fora do box, sem SC e sem volta apagada.
