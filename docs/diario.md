# Diário de desenvolvimento

Registro do que foi feito e decidido em cada dia de trabalho, da entrada mais recente para a mais antiga.

## 5 de novembro de 2026

### Baseline ingênuo

- `modeling/naive.py`: ao fim da volta *t*, prevê para as voltas *t*+1 a *t*+10 a média do ritmo relativo das 3 últimas voltas limpas. Sem intervalo e sem treino; previsões na validação e no teste.
- Código comum a todos os modelos, no formato de [resultados.md](resultados.md): `modeling/frame.py` monta a tabela de alvos (uma linha por janela e horizonte) e `modeling/results.py` grava `runs/<run_id>/` (`config.json`, `predictions.parquet`, `metrics.json`, com escrita atômica) e calcula as métricas (MAE, RMSE e viés por horizonte, erro em 5 voltas, recortes por stint, composto e corrida, skill contra uma referência e as medidas de intervalo quando houver p10 e p90).
- Resultado (`runs/naive_20261105`):

  | Divisão | MAE | MAE h = 1 | MAE h = 10 | Viés | Erro em 5 voltas |
  |---|---|---|---|---|---|
  | Validação (Spa, Hungaroring, Zandvoort, Monza) | 0,725 s | 0,52 s | 0,87 s | +0,29 s | 2,51 s |
  | Teste (Madring, Baku, Singapura, Austin, México) | 0,732 s | 0,50 s | 0,92 s | +0,31 s | 2,37 s |

  O viés positivo é esperado: o carro fica mais rápido à medida que queima combustível, e o ingênuo repete o ritmo atual. O erro é bem maior depois de uma parada (1,37 s no teste) do que no mesmo stint (0,64 s), que é onde a rede tem mais a ganhar.

### GRU v0

- `modeling/gru.py`: GRU encoder–decoder com saída quantílica. O contexto estático (embeddings de circuito, piloto, equipe-ano e regulamento, mais as estáticas numéricas) inicializa o encoder e entra no decoder a cada passo. O decoder recebe só as variáveis conhecidas das voltas futuras. A saída tem p10 ≤ p50 ≤ p90 por construção (p50 mais e menos incrementos positivos). O encoder lê cada corrida uma vez e todas as previsões da sequência partem dos seus estados.
- Treino: perda pinball nas voltas limpas, alvo padronizado pela média e desvio do treino, Adam (taxa 0,001), corte do gradiente pela norma (1,0), dropout 0,1, estado de 64, contexto de 16, lotes de 32 sequências, early stopping pela perda de validação (paciência 8). Para que circuitos e pilotos novos tenham um embedding razoável, 10% das categorias são trocadas por `<unk>` no treino. São 41.365 parâmetros; o treino parou na época 21 (melhor época: 13), com ~6,5 s por época em CPU e 2 min e 17 s no total.
- Resultado (`runs/gru_20261105_s1`), contra o ingênuo:

  | Divisão | MAE GRU | MAE ingênuo | Skill | MAE h = 1 | MAE h = 10 | Viés | Cobertura 80% |
  |---|---|---|---|---|---|---|---|
  | Validação | 0,533 s | 0,725 s | +27% | 0,41 s | 0,60 s | −0,03 s | 78% |
  | Teste | 0,567 s | 0,732 s | +23% | 0,43 s | 0,66 s | +0,05 s | 80% |

  - O ganho cresce com o horizonte (skill de +15% em h = 1 e +29% em h = 10 no teste) e o viés do ingênuo (+0,3 s, o efeito do combustível) praticamente desaparece.
  - **Depois de uma parada (tarefa B), o erro cai pela metade:** 0,66 s contra 1,37 s do ingênuo no teste. A rede prevê o ganho do pneu novo a partir do cenário que o decoder recebe.
  - O intervalo p10–p90 está bem calibrado: cobre 80% das voltas no teste, com 13% abaixo do p10 e 93% abaixo do p90 (ideais: 10% e 90%).
  - Por corrida do teste: Austin 0,48 s (ingênuo 0,69 s) e México 0,38 s (ingênuo 0,52 s). O pior caso é Madring (0,87 s, cobertura de 59%), circuito novo que a rede só conhece como `<unk>`.

### Backtest em Austin e México

- `modeling/backtest.py` gera `runs/gru_20261105_s1/backtest.html`: erro por horizonte nas duas corridas, indicadores por corrida, gráficos volta a volta dos três primeiros colocados e uma tabela de erro por piloto. Os gráficos mostram as previsões feitas ao fim das voltas 10, 20, 30… para as 10 voltas seguintes (mediana e faixa p10–p90 da GRU, o ingênuo e os tempos reais), com as paradas e as voltas de SC/VSC marcadas.
- As figuras ficam em `modeling/figures.py`, em Plotly, para serem reaproveitadas no painel. Cores fixas por papel (real em preto, GRU em azul, ingênuo em laranja), validadas para daltonismo.
- Resultados, nas voltas limpas previstas pelos dois modelos:

  | Corrida | MAE GRU | MAE ingênuo | Melhora | Cobertura p10–p90 | Depois de uma parada (GRU × ingênuo) |
  |---|---|---|---|---|---|
  | Austin | 0,469 s | 0,691 s | +32% | 88% | 0,52 s × 1,53 s |
  | México | 0,379 s | 0,521 s | +27% | 90% | 0,48 s × 1,03 s |

  A GRU melhora sobre o ingênuo para todos os pilotos do pódio. Nos gráficos, o efeito da tarefa B é visível: depois de cada parada o ingênuo continua no ritmo do pneu velho, e a GRU já começa no ritmo do pneu novo. A cobertura nas duas corridas (88–90%) passa um pouco do alvo de 80%: aqui o intervalo é largo demais.

### Correção no pipeline: volta limpa calculável ao vivo

- A flag `clean` usa a mediana da corrida inteira (corte de 107%), que não existe durante a corrida. Ela entrava como variável de entrada da rede e no cálculo dos speed traps relativos, um pequeno vazamento de informação do futuro.
- Nova flag `clean_live` (corte sobre a mediana das voltas anteriores e teto de ritmo de 30% acima da referência). Ela substitui `clean` nas entradas da rede, nos speed traps e no baseline; `clean` fica só para a máscara da perda e as métricas. As duas concordam em 99,9% das voltas.
- O teto de ritmo veio de um caso real: em Zandvoort 2026, a volta de relargada depois da bandeira vermelha (~146 s, o dobro do normal) passava como limpa por ser a primeira volta sem histórico, e inflava o erro da validação (MAE de 1,10 s antes da correção).

## 2 de novembro de 2026

### Escolha do projeto e escopo

- Entre as ideias levantadas sobre os dados de F1 coletados, foi escolhida a **previsão do ritmo de corrida e da degradação de pneus com redes recorrentes (LSTM/GRU)**: prever o tempo das próximas voltas (tarefa A) e a curva do próximo stint com pneu novo (tarefa B), com intervalo de previsão.
- Arquitetura definida: encoder–decoder recorrente. O encoder lê as voltas já disputadas; o decoder recebe só o que se sabe sobre as voltas futuras (pneu, composto, paradas, voltas restantes). Alvo: ritmo relativo a uma volta de referência do piloto no fim de semana.
- **Dados de 2022 a 2026**: a era de efeito solo (2022–25) e o regulamento de 2026. Em vez de um modelo por era, uma única rede condicionada por embeddings de regulamento e de pneu; comparar com treinar só em 2026 e com pré-treino seguido de fine-tuning.
- Modelo **só para pista seca**. Um modelo para pista molhada (com o ponto de troca para pneu liso) fica como curiosidade para depois.
- Telemetria, pressões dos pneus e características da pista da Pirelli ficam fora da v0. A telemetria pode entrar como ablação offline em 2026.

### Prazo e testes ao vivo

- Entrega no início de dezembro. Testes ao vivo no **GP de São Paulo** (8/11, versão v0, modelo simplificado) e no **GP de Las Vegas** (22/11, versão v1).
- Fonte ao vivo: o [f1-dash](https://github.com/slowlydev/f1-dash) local, que roda sem Docker pelo `start-live.sh`. O serviço `realtime` expõe o estado da corrida em `localhost:4000/api/current` e devolve `{}` sem sessão.
- O arquivo histórico oficial do live timing exige autenticação (HTTP 401), então não dá para gerar replays de corridas passadas. As sessões serão gravadas pelo próprio projeto, e os treinos livres de cada fim de semana servem de ensaio.
- O plano completo e o cronograma foram gerados em PDF (`docs/plano_projeto_rn2.pdf`, mantido só localmente). O cronograma com o status de cada atividade fica no fim do README.

### Repositório e dados

- Repositório [Neural_Stints](https://github.com/schneiderweisse/Neural_Stints) criado, com README em português. Todos os arquivos do projeto ficam nele; os dados brutos e derivados ficam no `.gitignore`.
- `pipeline/fetch_tracinginsights.py`: sparse clone parcial do TracingInsights 2022–2026, só com tempos de volta, clima, race control e pilotos (sem telemetria). São 111 corridas, ~123 mil voltas de corrida e 226 MB, com dados até o GP do México de 2026. O script tenta de novo quando a conexão cai.
- Cópia de um subconjunto do OpenF1 (2023–2026) e da tabela de compostos do estudo de regulamento.
- `pipeline/sessions.py`: padroniza os nomes de sessão. Em 2023, a quali da sprint se chama `Sprint Shootout`; em 2022, a pasta `Sprint Qualifying` é uma cópia da corrida sprint e é descartada.
- Figura `docs/figures/variaveis_rede.svg`: como cada grupo de variáveis entra na rede.

### Pipeline de preparação

- `pipeline/build_laps.py`: tabela de voltas de todas as sessões (~304 mil voltas).
- `pipeline/race_laps.py`: voltas de corrida com referência, alvo, flags de limpeza e variáveis por volta (setores relativos, speed traps relativos ao passado do próprio piloto, gaps calculados pelos tempos acumulados, pneu, clima, progresso). Voltas limpas: 84,8% nas corridas secas.
- `pipeline/sequences.py`: matrizes normalizadas por volta, variáveis estáticas por piloto-corrida e índice das janelas encoder/decoder (horizonte de 10 voltas). São 1.989 sequências e 104 mil janelas. Divisão cronológica por corrida: 89 corridas de treino, 4 de validação (Spa, Hungaroring, Zandvoort e Monza 2026) e 5 de teste (Madring, Baku, Singapura, Austin e México 2026). As 13 corridas molhadas ficam de fora.
- Revisões feitas no mesmo dia:
  - **Referência do alvo**: a melhor volta seca no quali. Se o quali foi molhado (mais da metade das voltas com pneu de chuva), vale a melhor volta seca dos treinos livres. Isso corrigiu o ritmo negativo do Canadá 2023 e a referência vazia de Spa 2023.
  - **Circuitos em vez de GPs**: em 2026 o GP da Espanha foi em Madri e o do Bahrein em Kuala Lumpur. As corridas passaram a ser identificadas como `<ano> <circuito>` (`data/reference/circuits.csv`).

### Treino e painel

- Máquina: Intel i7-1165G7 (4 núcleos), 16 GB de RAM, sem GPU com CUDA. Um protótipo da GRU levou 3,8 s por época (hidden 64) e 7,8 s (hidden 128) em CPU. O treino será feito localmente; o Colab fica como plano B. PyTorch (CPU) instalado no `.venv`.
- Painel em **Streamlit**, desenvolvido junto com a rede e mostrando só o que estiver pronto. Abas: Exploração, Modelos, Backtest e Ao vivo. O serviço de previsão e o painel são processos separados: o serviço grava em disco e o painel só lê.
- Formato padrão dos resultados de todos os modelos e lista de métricas registrados em [resultados.md](resultados.md), com a seção do que existe na v0.

### Próximo passo

Atividade de quarta, 4/11: baseline ingênuo, GRU v0 e backtest em Austin e México, já gravando no formato padrão, e o esqueleto do painel.
