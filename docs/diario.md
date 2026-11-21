# Diário de desenvolvimento

Registro do que foi feito e decidido em cada dia de trabalho, da entrada mais recente para a mais antiga.

## 21 de novembro de 2026

### Modelo ao vivo da v1: a LSTM com regulamento

- **A decisão:** o previsor ao vivo passa a usar o ensemble de 5 LSTMs com o embedding de regulamento (`lstm_20261120_ens5`) no lugar da GRU com regulamento (`gru_20261106_ens5`). Vale a partir do GP de Las Vegas.
- **Por quê:** no fatorial célula × regulamento, os ensembles da LSTM com regulamento e da GRU sem regulamento empataram na validação (pinball de 0,1953 contra 0,1951). Entre as duas, a LSTM tem:
  - o menor MAE de teste (0,457 s);
  - a cobertura mais perto de 80% na validação;
  - o melhor resultado em São Paulo (1,001 s contra 1,062 s).
- **Ela não segue a regra por semente:** essa regra apontava a GRU sem regulamento (0,202 contra 0,206).
- **O ensemble misto também foi avaliado,** com as 10 redes: 5 GRUs sem regulamento e 5 LSTMs com regulamento.
  - Na validação, ganhou por pouco: MAE de 0,562 s contra 0,569 s.
  - Em São Paulo, deu 0,992 s.
  - Parte desse ganho pode vir só de ter o dobro de membros. Ficou de fora: a v1 fica com uma célula só.
- **No código:**
  - `live/predict.py` ganhou `LIVE_RUN = runs/lstm_20261120_ens5`, o padrão quando não se passa `--run`. Antes, o padrão era a GRU mais recente.
  - Os textos do painel que diziam "GRU" (previsão, duelos) agora dizem "rede".
- **Conferido:** a reprodução de São Paulo com o padrão dá MAE de 1,001 s. Foram 7.661 voltas, 5 membros e cerca de 80 ms por previsão.
- Detalhes em [resultados.md](resultados.md#célula--regulamento).

## 20 de novembro de 2026

### Célula × regulamento: a LSTM sem regulamento

- **A pergunta:** a LSTM ganhou da GRU, e tirar o regulamento ajudou a GRU. Os dois ganhos se somam? Para responder, faltava uma casa do fatorial 2×2: a LSTM sem regulamento.
- **O treino:** `modeling.eras` ganhou `--cell` e `--strategies`. A LSTM escolhida (estado 32, dropout 0,1) foi treinada sem o regulamento, 5 sementes, 3 em paralelo, em ~8 min.
- **A regra, fixada antes de treinar:** ganha a menor média da pinball de validação por semente. O teste e São Paulo só descrevem.

| Célula | Regulamento | Pinball val., por semente (s) | MAE val., ensemble (s) | MAE teste, ensemble (s) | São Paulo (s) |
|---|---|---:|---:|---:|---:|
| GRU | com (em uso) | 0,207 | 0,582 | 0,472 | 1,037 |
| GRU | sem | **0,202** | **0,565** | 0,463 | 1,062 |
| LSTM | com | 0,206 | 0,569 | **0,457** | **1,001** |
| LSTM | sem | 0,209 | 0,581 | 0,463 | 1,018 |

- **Os ganhos não se somam.** Sem o regulamento, a LSTM piora: a pinball de validação sobe em 3 das 5 sementes.
  - O efeito de tirar o regulamento, pareado por semente, é −0,005 na GRU e +0,003 na LSTM. A interação (+0,008, erro-padrão 0,004) é maior que o próprio ganho na GRU.
  - Ou seja, o ganho depende da célula; não é uma propriedade dos dados.
- **Pela regra, a escolhida é a GRU sem regulamento.**
- **Nos ensembles, ela empata com a LSTM com regulamento:**
  - pinball de validação de 0,1951 contra 0,1953;
  - diferença de MAE de +0,004 s na validação e −0,006 s no teste, os dois intervalos cruzando o zero.
  - A LSTM com regulamento ganha em São Paulo (1,001 s contra 1,062 s), mas essa corrida fica fora do protocolo.
- **Tempo ao vivo:** ~55 ms por previsão na GRU e ~80 ms na LSTM, menos de 2 s por volta para 20 pilotos.
- Detalhes em [resultados.md](resultados.md#célula--regulamento).

### Eras de regulamento

- **A pergunta:** a validação e o teste são de 2026, o primeiro ano do regulamento novo. Só 241 das 1.835 sequências de treino são de 2026, de 11 corridas. Como usar as temporadas de 2022–25?
- **`modeling.eras`:** seis estratégias com a GRU escolhida, 5 sementes cada, escolha pela pinball média na validação.
  - São 25 treinos novos, 3 em paralelo, ~1 h. O `pooled`, o modelo em uso, vem da bateria de 6/11.
  - Depois, mais 20 fine-tunings com taxas menores, para ver se o resultado dependia da taxa.
- **O `modeling.gru` ganhou três opções:** `--train-years` (só as corridas desses anos), `--init` (fine-tuning a partir de outro run) e `--zero-cat` (apaga uma categórica).
  - As categorias que o run não viu no treino viram `<unk>`, como ao vivo, e a lista vai no `config.json`.
  - O previsor ao vivo agora aplica a lista de cada membro do ensemble.

| Estratégia | Pinball val., por semente (s) | MAE val., ensemble (s) | MAE teste, ensemble (s) | São Paulo, reprodução (s) |
|---|---:|---:|---:|---:|
| Todas as eras, sem embedding de regulamento (`pooled_noreg`) | **0,202** | **0,565** | **0,463** | 1,062 |
| Só 2022–25 (`pre_2022_25`) | 0,205 | 0,572 | 0,465 | 1,136 |
| Todas as eras, com embedding de regulamento (`pooled`, em uso) | 0,207 | 0,582 | 0,472 | 1,037 |
| 2022–25 + fine-tuning em 2026 (`pretrain_finetune`) | 0,213 | 0,603 | 0,487 | 1,085 |
| Todas as eras + fine-tuning em 2026 (`pooled_finetune`) | 0,224 | 0,638 | 0,496 | 1,028 |
| Só 2026 (`2026_only`) | 0,357 | 1,011 | 0,592 | **0,954** |

- **Só 2026 não funciona:** com 11 corridas, a rede erra mais que o ingênuo na validação.
- **2022–25 sozinhas já dão uma rede tão boa quanto a em uso.** O alvo é relativo à volta de referência do quali, que deve absorver a mudança de velocidade entre as eras.
- **Sem o embedding de regulamento, a rede fica um pouco melhor**, e é a escolhida pela validação.
  - Ganha 0,017 s na validação e 0,009 s no teste, com os intervalos fora do zero.
  - Por semente, no teste, a diferença vai de −0,040 a +0,017 s.
- **O fine-tuning em 2026 piora, em todas as taxas (0,0003, 0,0001 e 0,00003).**
  - A validação piora desde a primeira época: a rede se ajusta às 11 corridas de treino.
  - O ponto de partida carrega certo: antes do primeiro passo, a perda é a do run de origem.
- **São Paulo inverte a ordem:** na reprodução, quanto mais peso em 2026, melhor, e o só 2026 é o melhor de todos. É uma corrida só e atípica: a referência do alvo veio dos treinos livres, por causa do quali molhado. Mas mostra que a vantagem do `pooled_noreg` é pequena e pode mudar de corrida para corrida.
- **O modelo ao vivo continua o `gru_20261106_ens5`.** A escolha da v1 (GRU ou LSTM, com ou sem regulamento) fica para o congelamento. Detalhes em [resultados.md](resultados.md#eras-de-regulamento).

### Calibração dos quantis

- **A previsão quantílica existe desde a v0:** p10, p50 e p90 com a perda pinball. Faltava medir a calibração de todos os modelos. O `modeling.compare` agora imprime a fração de voltas abaixo do p10, entre p10 e p90 e acima do p90, a largura do intervalo e a cobertura por horizonte.
- **As redes erram a cobertura por pouco, em direções opostas:** 79% na validação e 87–88% no teste, para um ideal de 80%.
  - Na validação, 13% das voltas saem abaixo do p10 e 7% acima do p90. É o mesmo sinal do viés positivo. A cobertura cai de 86% em h1 para 75% em h10.
  - No teste, fica em ~87% em todos os horizontes.
- **Sem recalibração:** um ajuste feito na validação alargaria os horizontes longos, que no teste já cobrem demais.
- **O LightGBM tem os intervalos mais estreitos:** cobre 71% na validação e 81% no teste.
- Tabelas em [resultados.md](resultados.md#calibração-dos-quantis).

### LSTM × GRU

- **A mesma rede com a célula trocada:** `modeling.gru --cell lstm`. O resto é igual: dados, variáveis, decoder e treino. O estado da LSTM é o par (h, c), e o decoder de cada volta parte dos dois. Por isso o encoder da LSTM anda volta a volta, para guardar o c de cada volta.
- **Conferências:**
  - a GRU já treinada dá as mesmas previsões de antes (diferença de 7·10⁻⁹);
  - na LSTM, a previsão do caminho ao vivo é igual à do treino;
  - o previsor ao vivo agora grava o nome certo do modelo (`gru` ou `lstm`).
- **Bateria** (`modeling.cells`, 20 treinos com 3 em paralelo, ~40 min). Cinco sementes em cada configuração:
  - estado 32, a configuração da GRU;
  - estado 26, com o mesmo número de parâmetros da GRU;
  - estado 64;
  - estado 32 com dropout 0,2.
  - Pela validação, a LSTM escolhida também é a de estado 32 e dropout 0,1.

| | GRU | LSTM |
|---|---:|---:|
| Parâmetros | 14.901 | 19.957 |
| Pinball validação, por semente (s) | 0,207 ± 0,007 | 0,206 ± 0,008 |
| MAE teste, por semente (s) | 0,496 ± 0,014 | 0,488 ± 0,020 |
| MAE validação, ensemble de 5 (s) | 0,582 | **0,569** |
| MAE teste, ensemble de 5 (s) | 0,472 | **0,457** |
| São Paulo, reprodução com o previsor ao vivo (s) | 1,037 | **1,001** |
| Segundos por época | 4,8 | 7,7 |

- **Uma rede sozinha: empate.** A diferença por semente (0,004 s na validação, 0,008 s no teste) é menor que a variação entre sementes. Com os mesmos parâmetros da GRU (estado 26), a LSTM fica um pouco atrás na validação.
- **No ensemble, a LSTM ganha por pouco, mas sempre:** 0,013 s na validação e 0,015 s no teste, com os intervalos fora do zero no bootstrap por piloto:corrida, e 0,036 s em São Paulo. As sementes dela variam mais entre si, e a média ganha mais com isso.
- **Custo:** ~60% mais tempo por época e algumas épocas a mais até parar.
- **O modelo ao vivo continua o ensemble da GRU.** O da LSTM (`runs/lstm_20261120_ens5`) está pronto e roda no previsor com `--run lstm_20261120_ens5`.

### Modelo linear misto e LightGBM

- **Os dois baselines do plano**, no formato padrão (`runs/lmm_20261120/`, `runs/lgbm_20261120/`), comparados com o ingênuo e a GRU nas mesmas voltas por um script novo, `modeling.compare`. Tabelas completas em [resultados.md](resultados.md#comparação-dos-modelos-na-validação-e-no-teste).

| Modelo | MAE validação (s) | MAE teste (s) | Skill vs. ingênuo (val. / teste) | Cobertura 80% (val. / teste) |
|---|---:|---:|---:|---:|
| Ingênuo | 0,783 | 0,655 | — | — |
| Linear misto | 0,772 | 0,540 | +1,4% / +17,6% | 75% / 87% |
| LightGBM | 0,597 | 0,498 | +23,7% / +23,9% | 71% / 81% |
| GRU (ensemble de 5) | **0,582** | **0,472** | **+25,6% / +28,0%** | 79% / 88% |

- **A GRU continua na frente, mas o LightGBM chega perto.** A diferença é de 0,015 s na validação (IC 95% de 0,002 a 0,028) e 0,026 s no teste (0,015 a 0,038), num bootstrap por piloto:corrida. O LightGBM só ganha da GRU em Monza.
- **Modelo linear misto** (`modeling.lmm`, statsmodels):
  - **Fórmula:** ritmo em função de composto × idade do pneu, progresso da corrida (combustível) e temperatura da pista, mais um intercepto e uma inclinação no progresso aleatórios por piloto:corrida.
  - **Ao vivo,** o efeito de cada piloto é o BLUP com as voltas limpas até a volta atual, em forma fechada. Confere com o do statsmodels até 4·10⁻¹².
  - **O que ele diz:**
    - o macio degrada 0,061 s por volta, 36% mais que o médio e o duro (0,045 s);
    - do começo ao fim da corrida, a volta fica 3,3 s mais rápida (combustível e pista);
    - cada °C de pista custa 0,022 s.
  - **A parte aleatória foi escolhida na validação.** O intercepto sozinho, o do plano, dava 0,799 s. Com a inclinação no progresso, 0,766 s. A inclinação na idade do pneu piorou (0,858 s): estimada com poucas voltas de um stint, ela extrapola mal.
  - **Onde ele perde:** na volta seguinte (h1) perde até do ingênuo, porque o nível do piloto é a média da corrida até ali, não o ritmo recente. Ganha nos horizontes longos e depois das paradas.
- **LightGBM** (`modeling.lgbm`):
  - **Previsão direta** com o horizonte como variável e um modelo por quantil.
  - **Alvo:** a mudança de ritmo em relação à média das 3 últimas voltas limpas.
  - **37 variáveis:** lags e médias do ritmo limpo, a inclinação do stint, o estado na volta atual e as mesmas variáveis conhecidas do futuro que a GRU recebe (composto, idade do pneu, paradas até a volta prevista).
  - **Treino:** 782 mil linhas; o número de árvores é escolhido por early stopping na validação. Treina em ~2 min.
  - **Mais importantes:** circuito, inclinação do stint e mesmo stint.
  - **Ablação sem o circuito:** piora a validação (0,626 s) e melhora um pouco o teste (0,489 s). Pela validação, o circuito ficou.
- **Achado no caminho:** o `race_laps.parquet` foi refeito em 10/11 (compostos C1–C6). Nesse meio-tempo, a fonte completou a volta 1 e três voltas finais do México 2026 (BOT, LIN, STR), que o `dataset.npz` de 06/11 não tem.
  - O `modeling/frame.py` agora casa as voltas pelo par (sequência, volta), em vez de exigir arquivos idênticos.
  - Todos os modelos são avaliados exatamente nas voltas da GRU. As voltas limpas batem uma a uma.
  - O ingênuo, refeito, reproduz o run de 06/11.
- **Instalados** statsmodels 0.15 e LightGBM 4.7 (em `requirements.txt`).

## 12 de novembro de 2026

### Duelos: quando um carro vai chegar no outro

- **Ideia do Eduardo:** um módulo embaixo da cronometragem com pares de pilotos, comparando o ritmo e prevendo em quantas voltas começa a disputa de posição. O alvo é a aproximação, não a ultrapassagem: o carro de trás chegar a menos de 1 s do da frente na linha, a zona de ataque.
- **Sem rede nova.** O gap projetado é o gap de agora mais a soma das diferenças de tempo previstas pela GRU para as próximas 10 voltas (`modeling/duels.py`). A incerteza vem dos quantis p10–p90 de cada piloto e dá a probabilidade de já estar a menos de 1 s em cada volta. Disso saem:
  - **volta prevista:** onde essa probabilidade passa de 50%;
  - **janela de 80%:** de onde passa de 10% até onde passa de 90%.
- **Calibração:** somar os desvios como se os erros de um piloto fossem totalmente persistentes deixava a janela larga demais (98% de cobertura na validação). O fator 0,5, escolhido na validação, dá 80%.
- **Backtest** (`modeling.duels_backtest`, resultados em `runs/duels_20261112/metrics.json`). Situações: ao fim de cada volta, cada carro a 1–5 s do da frente. Cada conta para na primeira parada, safety car, pneu de chuva ou volta sem tempo de algum dos dois, porque a projeção supõe que ninguém para.

| Corridas | Situações | Disputas reais | Precisão GRU (ingênuo) | Recall GRU (ingênuo) | A ±1 volta GRU (ingênuo) | Na janela de 80% | Brier GRU (taxa fixa) |
|---|---|---|---|---|---|---|---|
| Validação (Zandvoort, Monza, Madring, Baku) | 809 | 41% | 80% (74%) | 55% (55%) | 75% (70%) | 80% | 0,163 (0,241) |
| Teste (Singapura, Austin, México) | 515 | 38% | 82% (75%) | 78% (71%) | 86% (76%) | 93% | 0,120 (0,235) |
| São Paulo, previsões feitas ao vivo, pneu liso | 160 | 50% | 84% (78%) | 66% (65%) | 74% (73%) | 80% | 0,166 (0,250) |

  No teste, 82% das disputas que a rede previu aconteceram, e 86% delas começaram a até 1 volta da volta prevista. As probabilidades são bem calibradas nas faixas de 40% a 100%. Na faixa de 20% a 40% do teste, a disputa veio menos vezes do que a rede previu (8%).
- **No painel** (`dashboard/duels_view.py`), embaixo da cronometragem:
  - os duelos fixos escolhidos (sempre um par; na demo, NOR × VER);
  - os três pares vizinhos a até 5 s com mais chance de disputa;
  - a lista dos que já estão a menos de 1 s.
  - **Cada cartão** traz o gap, o ritmo das 3 últimas voltas de cada um, o gráfico do gap projetado com a faixa e a janela, e a volta prevista com a chance. Avisa quando a projeção não vale: pneu de chuva, box ou falta de previsão.
  - **Na gravação,** o cartão mostra também o que aconteceu depois: ✔ quando a disputa começou dentro da janela, ✘ quando veio fora dela ou não veio, ou o motivo de a conta ter parado.
- **Leitura do gráfico, ajustada a pedido do Eduardo:**
  - uma legenda única acima dos cartões explica cada elemento e os dois eixos;
  - o eixo horizontal traz o número de cada volta, com a volta da previsão em destaque;
  - além da linha tracejada laranja de 1 s (começa a disputa), há uma verde em 0 s, a ultrapassagem: abaixo dela, o carro de trás passou à frente;
  - o gap real segue depois do início da disputa, em vez de parar nele, então dá para ver quando ele cruza o zero;
  - o resultado agora diz também a volta da ultrapassagem ("A disputa começou na volta 44, dentro da janela · ultrapassagem na volta 45").
- **Testado:**
  - na reprodução de São Paulo, no painel completo e na demo;
  - o tempo de cada atualização fora do navegador ficou entre 0,16 e 0,24 s;
  - na demo, a reprodução seguiu a um quadro a cada ~1,5 s.

### Transição de troca de posição na cronometragem

- **Ideia do Eduardo,** vendo a reprodução de Interlagos na demo. Quando um piloto ganha ou perde posição, a linha dele primeiro ganha destaque ainda no lugar antigo, verde para quem subiu e vermelho para quem desceu. Depois ela desliza até o lugar novo e o destaque se apaga. Uma seta (▲2, ▼1) aparece ao lado da posição durante a transição.
- **Como funciona:** só com CSS, sem script, porque o Streamlit não roda JavaScript dentro do markdown.
  - O painel guarda a ordem anterior de cada tela no `session_state` (`live_tab.position_moves`).
  - Cada linha que mudou recebe uma classe e o deslocamento em linhas (`--d`). A animação parte desse deslocamento e volta a zero.
  - **Duas fases iguais:** as classes `mv-a` e `mv-b` reiniciam a animação quando o mesmo piloto troca de novo.
  - **Marcação mantida por 1,6 s:** a próxima atualização (ao vivo, a cada 1 s) não corta a transição, porque o navegador reaproveita a tabela e a animação continua.
  - **Saltos não animam:** na gravação, só momentos vizinhos (até 61 s de distância) são comparados. Um salto do controle de tempo não anima nada.
  - **Movimento reduzido:** quem prefere menos animação vê só o destaque.
- **Testado** na reprodução de São Paulo, no painel completo e na demo (stlite). As trocas observadas foram até o fim sem reiniciar no meio, e um retrato congelado no meio da transição mostra as linhas destacadas ainda no lugar antigo.

## 10 de novembro de 2026

### Compostos C1–C6 por GP

- **Tabela completa:** `data/reference/compounds.csv` passou de 14 para 115 corridas, todas as de 2022 a 2026, com a fonte de cada linha. As fontes são 62 páginas: comunicados da Pirelli, as páginas "What tyres will the teams and drivers have" da F1 e imprensa especializada que reproduz o comunicado.
  - **Conferência:** as linhas de 2025 e 2026 e uma amostra das outras foram conferidas no texto da própria página, e não só no resumo da busca.
  - **Erro evitado:** a busca dizia C2–C4 para Silverstone 2026, mas a página da F1 diz C1–C3.
  - **Pendentes:** Las Vegas, Catar e Abu Dhabi de 2026 ficam em branco até a Pirelli anunciar. Las Vegas precisa entrar antes do teste ao vivo.
- **Pulos de composto:** a coluna `note` marca as nominações não consecutivas (Austrália 2022: C2, C3 e C5; Spa e Austin 2025: C1, C3 e C4; Cidade do México 2025: C2, C4 e C5).
- **Especificação por temporada:** `data/reference/tyre_specs.csv` traz a faixa de cada ano, de C1–C5 em 2022 a C1–C6 em 2025 e C1–C5 em 2026, com os pneus novos. A numeração muda de sentido: em 2023 o antigo C1 virou C0, e entrou um C1 novo entre ele e o C2. Por isso a dureza só se compara dentro de uma temporada.
- **Pipeline:** `pipeline/race_laps.py` ganhou `add_compounds`, com a dureza nominal da volta (`compound_c`) e a nominação do fim de semana (`c_hard`, `c_medium`, `c_soft`). O caminho ao vivo (`live/history.py`) usa a mesma função.
  - **Cobertura:** 100% das voltas em pneu liso de 2022 a 2026.
  - **Paridade:** a conferência treino × ao vivo continua em 100%.
  - **Rede:** a dureza ainda não entra nela. Isso fica para os próximos treinos (LSTM × GRU e as eras), porque mudar as variáveis agora quebraria o ensemble da v0.

### Interlagos contra os dados oficiais

- **TracingInsights atualizado:** 2026 agora tem 20 GPs, até São Paulo, com 23.473 voltas de corrida com tempo. A corrida de São Paulo entrou na tabela como corrida molhada, então fica fora das sequências.
- **As voltas gravadas ao vivo batem com as oficiais:** as 1.435 voltas estão nas duas fontes, e tempos e setores são iguais ao milésimo nas 1.414 voltas que têm tempo nas duas. As outras 21 são a primeira volta, sem tempo no feed. Paradas e posições concordam em 100%, o composto em 99,9% e o stint em 99,5%.
- **Métricas com os tempos oficiais:** MAE de 1,110 s, contra 1,115 s com o feed gravado. O [relatório da corrida](race_reports/2026_sao_paulo.md) ganhou essa conferência.

## 9 de novembro de 2026

### Demo mais rápida

- **Sintoma:** a demo publicada demorava para responder. Trocar o piloto na aba GRU v0 levava 4,3 s no navegador.
- **Causas, medidas com o perfilador do Python:**
  - **Listas de pilotos:** quatro linhas montavam a lista refazendo o conjunto de pilotos da tabela inteira para cada piloto conferido, como em `[d for d in ordem if d in set(tabela["driver"])]`. Isso respondia por 40% do tempo de cada execução: 864 mil passos só nessas listas.
  - **Abas:** com `st.tabs`, qualquer clique roda o código das três abas e monta os 10 gráficos.
  - **Cópias de tabela:** o `st.cache_data` devolve uma cópia da tabela a cada chamada, e as 73 mil previsões eram copiadas a cada clique. A lista de corridas também percorria essas 73 mil linhas, uma a uma.
- **Correções:**
  - as listas usam o conjunto montado uma vez;
  - a demo troca as abas por um seletor de seção, e só a seção aberta roda;
  - o valor dos controles das seções fechadas é mantido (`layout.keep_widget_state`);
  - os gráficos da aba GRU v0 ficam em cache, porque só dependem dos runs, da corrida e do piloto;
  - as tabelas de previsões e de voltas usam `st.cache_resource`, sem cópia.
- **Resultado:**
  - **Fora do navegador:** cada clique caiu de 5,3 s para 0,08 a 0,23 s, conforme a seção.
  - **Trocar o piloto na aba GRU v0** caiu de 4,3 s para 0,7 s no navegador (versão publicada × versão nova servida localmente).
  - **Trocar de seção** leva de 0,9 a 1,2 s; abrir a GRU v0 pela primeira vez, 2 s.
  - **Reproduzir** agora acompanha o intervalo de 1,5 s (média de 1,4 s por passo).
  - **Abertura da página:** caiu de ~35 s para ~20 s, com os pacotes já no cache do navegador. A primeira visita ainda baixa o Python e os pacotes.
- **Painel completo:** continua com as abas, mas ganha as mesmas correções nas listas e nos caches.

## 8 de novembro de 2026

### GP de São Paulo: o primeiro teste ao vivo

- **Tudo no ar às 14:04, com a corrida ainda na volta 1.** O f1-dash, o gravador, o leitor, o previsor e o painel rodaram até o fim, sem queda.
  - Gravação: 73.750 mensagens, com no máximo 30 s sem Heartbeat. O modo acordado segurou as 2 h de corrida e as 2 h 40 do quali.
  - Leitor: 1.435 voltas de 21 carros. A correção de ontem no `write_atomic` segurou, com o painel lendo os arquivos a cada segundo.
  - Previsor: 12.603 previsões do ensemble de 5 sementes.
- **A corrida:** largada com intermediário, troca para macio nas voltas 14 a 17, quase todos com um segundo jogo de macios entre as voltas 32 e 53, e chuva de novo a partir da volta 58. Norris venceu, com Verstappen, que largou em 21º, a 14,6 s.
- **Resultados** (análise completa no [relatório da corrida](race_reports/2026_sao_paulo.md)):

  | Trecho | MAE GRU | MAE ingênuo | Viés GRU | Cobertura |
  |---|---|---|---|---|
  | Voltas limpas de pneu liso, corrida toda | 1,115 s | 1,534 s | +0,43 s | 50% |
  | Conjunto de teste, para comparar | 0,476 s | 0,655 s | +0,02 s | 88% |
  | Secagem (voltas 17–30) | 1,14 s | 2,97 s | +0,66 s | 50% |
  | Seco estável (voltas 31–57) | 0,76 s | 0,70 s | +0,49 s | 65% |
  | Chuva em pneu liso (voltas 58–62) | 2,51 s | 2,74 s | −2,51 s | 8% |
  | Intermediário (largada e final) | 9,6 e 11,7 s | 0,9 e 4,7 s | −9,6 e −11,7 s | 0% |

- **O que aprendi:**
  - A rede ganha quando o ritmo tem tendência e empata ou perde quando ele está estável.
  - Com pneu de chuva ela está fora do que aprendeu, porque o treino não tem corridas molhadas.
  - **Viés em seco:** das quatro explicações testadas, a idade do pneu não explica e a referência dos treinos livres explica pouco. A tendência que a rede não projeta explica o crescimento do viés com o horizonte. O nível de ritmo atípico é compatível, mas não está comprovado.
  - **Correção online rejeitada:** subtrair o erro recente melhora o seco desta corrida (0,858 → 0,726 s), mas piora a validação (0,558 → 0,605 s) e o teste (0,465 → 0,537 s).
  - **Alarme de chuva:** na volta 58, 80% do pelotão andou acima do p90 previsto, contra 3,8% em média nas voltas 31 a 56. O feed só marcou chuva 6 minutos depois. É uma corrida só, e falta medir os alarmes falsos nas outras.
  - **Cenário de parada:** rodado depois da corrida, ele previu cerca de metade do ganho real do pneu novo (0,72–0,84 s por volta, contra 1,41–1,82 s).
- **Correção do que eu tinha concluído durante a corrida:** o "erro ao vivo" do painel não mistura voltas molhadas e secas. A máscara `clean` só aceita pneu liso, então ele ignora o intermediário.
- **Painel, ajustes pedidos durante a corrida:**
  - Eixo de voltas só com inteiros, com uma marca por volta. Os rótulos rareiam acima de 30 e de 60 voltas para não se sobrepor.
  - Comparação de dois pilotos na previsão da GRU, no mesmo gráfico, cada um com sua cor, marcador e faixa.
  - Tempos de volta em minuto:segundo.milésimo no eixo e ao passar o mouse. Os gráficos usam um eixo de data do Plotly, que escolhe as marcas sozinho.
- **Demo atualizada:** a aba Corridas gravadas agora abre no GP de São Paulo, com as previsões feitas ao vivo, e mantém a corrida de 2019. Os dados da demo foram a 5,1 MB.
- **Defeito na reprodução de sessões lidas ao vivo.** Nessas sessões, o `laps.parquet` guarda o horário real, e a reprodução comparava com o tempo desde o início da gravação. Por isso nenhuma volta aparecia. Agora a reprodução usa o Heartbeat do momento para essas sessões. Testado na demo (corridas de 2026 e 2019) e no painel completo (corrida de 2026).
- **Relatório:** [`docs/race_reports/2026_sao_paulo.md`](race_reports/2026_sao_paulo.md), com métricas por fase, o estudo de caso Norris × Verstappen e como relacionar ritmo, pneu, stint, k voltas e paradas. O raciocínio das equipes é inferência a partir dos dados, porque não encontrei relatos da corrida de 2026.

## 7 de novembro de 2026

### TL3 e classificação

- **TL3:** gravado a partir das 11:42. O leitor caiu às 12:27, 3 minutos antes do fim.
  - **Causa:** o Windows recusou a troca do `laps.parquet` (`PermissionError`) porque outro programa estava com o arquivo aberto, provavelmente o painel lendo naquele instante.
  - **Correção:** o `write_atomic` tenta de novo por até ~5 s. Testado com o arquivo aberto por outro processo: a troca passa assim que ele fecha.
  - **Recuperação:** o `laps.parquet` foi refeito a partir da gravação, de 302 para 372 voltas. A volta 7 de Leclerc só apareceu no leitor ao vivo e foi mantida.
- **Atraso no painel:** uma volta no painel demorava mais que no ao vivo. `localhost` tenta IPv6 primeiro, e o `realtime` só escuta IPv4, o que custava ~2 s por conexão. Com `127.0.0.1`, cada atualização leva de 40 a 95 ms, e o painel pode atualizar a cada 1 s.
- **Gravador preso entre sessões:** ele não passou sozinho do TL3 para a classificação. Por isso, a regra para sessões seguidas é religar o gravador e o leitor depois que o f1-dash já mostrar a sessão nova.
- **Classificação:** toda na chuva, com bandeira vermelha aos 2 minutos do Q1 (15:02) e reinício às 16:26. Gravada do começo ao fim: 14.200 mensagens, no máximo 15 s sem Heartbeat e 391 voltas (296 de pneu de chuva extrema e 79 de intermediário).
- **Referência da corrida:** a regra do treino para quali molhado funcionou. Os 22 do grid ficaram com a melhor volta de pneu liso dos treinos livres, de 1:11.622 (VER) a 1:14.060 (STR).

### Quanto ainda dá para ganhar

- **Piso do erro:** um oráculo que vê o stint inteiro (curva quadrática ajustada a todas as voltas) erra 0,33 s no teste. A GRU erra 0,38 s na volta seguinte e 0,53 s em 10 voltas. Na volta seguinte, ela já está perto do piso; o que sobra é tendência em horizontes longos e depois das paradas.
- **Telemetria:** não temos histórico baixado, e o feed ao vivo não traz telemetria nem posição no mapa. Um modelo com telemetria só serviria como experimento fora do ao vivo.
- **GPU:** o gargalo hoje não é poder de cálculo. Cada treino leva poucos minutos em CPU, e a bateria mostrou que mudar hiperparâmetros não muda o erro.

### Demo no GitHub Pages

- O primeiro deploy falhou porque o Pages ainda não estava ativo. Com "GitHub Actions" escolhido como fonte, o workflow passou e a demo entrou no ar.
- **A parte ao vivo não pode ir para o Pages:** o navegador bloqueia a leitura do feed da F1 e do f1-dash público a partir de outro site (CORS). Por isso a demo fica com as corridas gravadas.

## 6 de novembro de 2026

### Demo do painel para o GitHub Pages

- **Pasta `demo/`:** versão pública do painel, sem a parte ao vivo. Tem três abas: a reprodução do GP da Alemanha de 2019, a GRU v0 (o ensemble de 5 sementes contra o ingênuo) e as Tarefas A e B.
- **Como roda:** inteiramente no navegador, com o stlite 1.12.0 (Streamlit 1.65 em Pyodide 314, Python 3.14, com pandas 3 e pyarrow). O site é estático (`index.html` e `app.zip`), então serve no GitHub Pages sem servidor. A primeira visita leva uns 30 s.
- **Código compartilhado:** a demo usa as mesmas abas de `dashboard/`.
  - O visual comum (CSS, cabeçalho e rodapé) passou para `dashboard/layout.py`.
  - A aba Ao Vivo ganhou um modo demo: só a reprodução, sem o f1-dash. Nele, em vez do aviso para rodar o previsor, aparece a explicação de que 2019 é anterior aos dados de treino.
- **Dados enxutos** (`demo/files/`, 4,5 MB, commitados), gerados por `demo/build.py export`:
  - os runs `gru_20261106_ens5` e `naive_20261106`, sem os pesos;
  - as voltas das 7 corridas de validação e teste;
  - a reprodução de 2019 já calculada a cada 30 s, só com o que o painel mostra: 294 momentos, 5,5 MB de JSON que viram 0,1 MB compactados, no lugar do fluxo bruto de 21 MB.
- **Defeito pego na exportação:** o `ReplaySource` altera o mesmo estado a cada mensagem, e a primeira versão do recorte guardava referências. Todos os momentos saíam com os valores do fim da corrida. Agora cada momento é copiado, e os 294 batem campo a campo com o estado completo.
- **Logos:** o cabeçalho da demo tem "F1" em texto vermelho no lugar do logo da Fórmula 1, que é marca registrada (o logo antigo também é). O logo da UnB vai junto.
- **Testado no navegador** (servidor local na porta 8770):
  - as três abas carregam sem erro, com 9 gráficos;
  - o botão Reproduzir avança a corrida;
  - no celular (375 px), a página não rola de lado.
- **Deploy:** o workflow `.github/workflows/demo.yml` monta o site (`demo/build.py site`, só com a biblioteca padrão) e publica no GitHub Pages a cada push na `main` que mexa na demo ou no código das abas. Falta escolher "GitHub Actions" como fonte em Settings → Pages.

### Ensaios ao vivo: TL1 e TL2 de Interlagos

- **Primeiro contato com o feed real da F1.** O f1-dash (`start-live.sh`), o gravador, o leitor e o painel ficaram no ar em ~1 min; o `realtime` compila em ~35 s.
- **Painel:**
  - **Atualização sem escurecer.** O Streamlit 1.67 deixa translúcido o que está sendo refeito (`data-stale`), e um CSS desliga isso. Medido: opacidade 1 em 515 amostras durante 7 s de atualizações.
  - **Setores com minissetores, como no f1-dash.** Cada minissetor vira uma barrinha com os códigos do feed: 2048 amarelo, 2049 verde, 2051 roxo e 2064 pit lane. Com o setor em andamento, o tempo da volta anterior aparece em cinza, e há uma legenda.
  - **Relógio da sessão.** O feed só manda o tempo restante quando o relógio começa ou para. Ao vivo, o painel desconta o tempo passado desde o envio, como o f1-dash.
  - **Selo "SEM DADOS HÁ X MIN"** em amarelo no lugar de AO VIVO, quando o último Heartbeat da F1 tem mais de 90 s.
- **TL1:** gravado das 12:31 às 12:51 (159 voltas, 22 pilotos), até o notebook ser fechado. A bandeira vermelha das ~12:40 apareceu no painel.
- **Defeito no leitor (`live/laps.py`), achado no TL1.** A volta de saída do box não recebe tempo do feed, então fica esperando o tempo atrasado. Na volta seguinte, o contador de voltas e o tempo chegam juntos, e o leitor dava esse tempo à volta que esperava. Daí em diante, cada volta do piloto ficava com o tempo da seguinte.
  - Exemplo: no VER, a volta 2 (de saída) recebeu 1:13.341, que é o tempo da volta 3.
  - Consequência no TL1: só 11 de 82 voltas batiam com a soma dos setores. Na corrida, isso aconteceria a cada parada.
  - **Correção:** o tempo que chega junto com uma volta nova é dessa volta. A que esperava fica com a soma dos setores, quando eles estão completos.
  - **Conferência extra:** se o tempo do feed difere da soma dos setores da mesma volta em mais de 0,05 s, fica a soma.
- **Verificação:**
  - TL1 refeito a partir da gravação: 86 de 86 voltas com setores completos batem ao milésimo, já só com a correção, sem precisar da conferência extra. O `laps.parquet` foi trocado pelo refeito.
  - TL2 e a corrida de 2019 refeitos com o leitor corrigido: nenhum tempo mudou (140 e 1.054 voltas).
- **TL2:** gravado só das 16:05 às 16:18 (13 min de feed, 140 voltas de 21 pilotos, soma dos setores batendo em 100%). O log de energia do Windows mostra o que aconteceu:
  - 16:10:13: a tela apagou depois de 5 min sem uso, e o notebook entrou em modo de espera (Modern Standby);
  - 16:18:23: a rede foi cortada;
  - 17:07:59: o notebook acordou.

  O f1-dash não reconectou à F1 depois disso e ficou com o estado das 16:18.
- **Modo acordado:**
  - `live/power.py` pede ao Windows sistema e tela acordados (`SetThreadExecutionState`). O gravador, o leitor e o previsor fazem esse pedido ao vivo. A bateria de treinos usa o mesmo módulo, mas só pede o sistema acordado.
  - O Windows aceitou o pedido. Não deu para isolar o efeito, porque outro programa já mantinha o sistema e a tela acordados no teste; a prova fica para o TL3.
  - Fechar a tampa ainda suspende.
  - O leitor avisa no terminal quando o feed fica mais de 90 s sem Heartbeat.
- **Referência de ritmo do plano B:** sai das gravações dos dois treinos para 27 pilotos (os 22 titulares e os estreantes do TL1). A mais rápida é a de ANT, 1:11.824, do TL2.

### Média das sementes (ensemble) no ao vivo

- **Código:** `modeling/ensemble.py` junta runs da GRU num ensemble.
  - A previsão é a média dos p10, p50 e p90 dos membros, volta a volta. A média de quantis em ordem continua em ordem.
  - Grava um run no formato padrão (`runs/gru_<data>_ens<n>/`), com as previsões de validação e teste, as métricas e as curvas de treino de cada membro.
  - O `model.pt` do run guarda os pesos de todos os membros.
- **Escolha pela validação**, a mesma regra da bateria: o ensemble do estado 32 (dropout 0,1) contra o do estado 64.

  | Ensemble de 5 sementes | Pinball validação | MAE validação | MAE teste | Cobertura teste |
  |---|---|---|---|---|
  | Estado 32 (escolhido) | 0,199 s | 0,585 s | 0,476 s | 88% |
  | Estado 64 | 0,204 s | 0,602 s | 0,472 s | 88% |

  No teste os dois empatam, e o estado 32 tem um terço dos parâmetros.
- **Run `gru_20261106_ens5`:**
  - teste: MAE de 0,476 s (uma semente: 0,496 ± 0,014 s), 28% melhor que o ingênuo, 0,54 s depois de uma parada;
  - validação: MAE de 0,585 s, cobertura de 79%.
- **Ao vivo:** o `live.predict` carrega todos os membros do `model.pt` e faz a mesma média. Um run de uma rede só continua funcionando como antes.
  - Refazendo o México 2026 pelo caminho ao vivo, as previsões batem com as do run nas janelas sem parada no horizonte, com diferença máxima de 1,5e-8.
  - No México, o erro ao vivo cai de 0,444 s (uma semente) para 0,426 s, com cobertura de 86%.
  - Cada previsão leva ~110 ms, contra ~80 ms. Quando os 19 pilotos fecham volta na mesma leitura, a atualização leva ~3 s.
  - O previsor e o painel usam a GRU mais recente, então domingo em Interlagos roda com o ensemble.
- **Painel:**
  - a aba GRU v0 mostra o ensemble (5 sementes, parâmetros por rede e melhores épocas) e a curva de treino de cada semente;
  - a aba Ao Vivo diz qual modelo fez as previsões.
- **Bateria:** a escolha da fase 3 agora fica gravada em `phase3.json` quando a fase 2 termina.
  - Rodar a bateria de novo só para refazer o relatório tinha escolhido outra vez, já contando as sementes extras, e treinado mais 2 sementes do estado 32 com dropout 0,2.
  - Com 5 sementes, essa configuração fez 0,208 s de pinball na validação (contra 0,207) e 0,515 ± 0,027 s no teste. Isso confirma a escolha original.
  - O relatório agora sugere o ensemble, não uma semente.
- **Correção:** uma gravação de um ano sem dados baixados do TracingInsights (como 2019) derrubava o previsor. Agora `iter_sessions` pula anos que não existem em `data/raw`, e o previsor passa às gravações ao vivo.

### Bateria de treinos: sementes, divisões e hiperparâmetros

- **Código:** `modeling/sweep.py` grava a bateria em `runs/sweep_<data>/`, fora do git.
  - Cada treino é um processo `modeling.gru` separado.
  - Rodar de novo pula os treinos prontos.
  - Com `--deadline HH:MM`, um treino só começa se a duração estimada couber no prazo.
  - A busca vai em largura: todas as configurações com uma semente, depois a semente seguinte. Assim, um corte no prazo tira sementes, não configurações.
  - No fim, grava `summary.csv` e `report.md`.

  Para isso:
  - `modeling.gru` e `modeling.naive` ganharam `--dataset`, `--out` e `--run-id`;
  - `pipeline.sequences` ganhou `--n-val`, `--n-test` e `--name`. A divisão de 5 corridas fica em `data/processed/dataset_split5`.
- **Execução:** 45 treinos em CPU, das 01:37 às 04:21, de 1,4 a 9 min cada.
- **Fase 1, divisão de 3 × de 5 corridas:** a configuração atual, com 5 sementes em cada divisão, avaliada nas 3 corridas de teste comuns às duas.
  - Divisão de 3: 0,490 ± 0,016 s. Divisão de 5: 0,499 ± 0,028 s.
  - Diferença de −0,010 s, com erro-padrão de 0,014 s.

  A impressão de que o modelo anterior era melhor vinha da variação entre sementes. Com uma semente de cada, a diferença de 0,009 s estava dentro desse ruído.
- **Fase 2, busca na validação:** estado 32, 64 ou 128 × dropout 0,1 ou 0,2 × taxa 0,001 ou 0,0003, com 3 sementes cada.
  - A pinball de validação ficou entre 0,206 e 0,217 s, com desvio entre sementes de até 0,010 s.
  - A taxa 0,001 ganhou da 0,0003 nos seis pares, e as 6 melhores configurações usam 0,001.
  - O tamanho do estado e o dropout ficaram dentro do ruído.
  - A escolhida foi estado 32 com dropout 0,1 (14.901 parâmetros, contra 41.365 da atual), a melhor média com 3 sementes. Com 5 sementes, caiu para o segundo lugar, a 0,001 s da primeira (estado 32 com dropout 0,2).
- **Fase 3, teste:**
  - Escolhida: 0,496 ± 0,014 s. Atual: 0,490 ± 0,016 s.
  - Cobertura p10–p90 de 85–86% nas duas.

  Nenhuma configuração da grade melhora a atual: nesta faixa, mexer nos hiperparâmetros não muda o erro.
- **Média das 5 sementes** (ensemble, média dos quantis):

  | Configuração | Validação: 1 semente → média | Teste: 1 semente → média |
  |---|---|---|
  | Atual (estado 64) | 0,624 → 0,602 s | 0,490 → 0,472 s |
  | Estado 32 | 0,610 → 0,585 s | 0,496 → 0,476 s |

  - É o maior ganho da bateria, e aparece também na validação; não é escolha feita olhando o teste.
  - A cobertura no teste fica em 88%.
- **Determinismo:** a mesma configuração com a mesma semente na bateria repete o MAE de `gru_20261106_s1` (0,497 s no teste).
- **Cobertura:** segue acima de 80% no teste (85–88%) e abaixo na validação (72–80%) em todas as configurações. É efeito das corridas, não dos hiperparâmetros.
- **Ao vivo:** nesse momento o previsor ainda usava uma rede só. A média das sementes entrou logo depois (seção acima).

### Painel: aba Tarefas A e B

- Nova aba `dashboard/tasks_tab.py`, com a corrida e até dois pilotos à escolha (só corridas da validação e do teste, que a rede não viu no treino):
  - **Tarefa A:** escolhe o ponto de partida de cada piloto (o começo da corrida ou uma das paradas) e k de 1 a 10. Mostra a previsão feita naquele ponto para as k voltas seguintes, contra os tempos reais, com uma tabela volta a volta. Exemplo: "parou na volta 30, k = 5" mostra a previsão feita ao fim da volta 30 para as voltas 31 a 35.
  - **Tarefa B:** escolhe um stint de cada piloto e vê o stint inteiro, coberto por previsões feitas no começo do stint e a cada 10 voltas (o horizonte da rede). A tabela traz MAE, viés, cobertura e a degradação real e a prevista (inclinação de uma reta ajustada ao tempo de volta ao longo do stint). No México 2026, segundo stint: Verstappen com degradação real de +0,027 s/volta e prevista de +0,028; Hamilton com +0,058 real e +0,031 prevista.
- Os dois pilotos usam cores validadas para daltonismo, e o segundo também tem linha tracejada e marcador em losango.
- A corrida de Madring foi tirada da lista desta aba, a pedido.

### Painel: visual

- **Cabeçalho:** logos da F1 (à esquerda, o atual em vermelho) e da UnB (à direita, a versão para fundo escuro), lidos de `dashboard/assets/`. Os arquivos vêm do Wikimedia Commons (fontes no README da pasta); por serem marcas registradas, ficam fora do git. Sem eles, o painel funciona igual.
- **Selo ao lado do título da sessão:**
  - **AO VIVO**, com um ponto vermelho piscando, quando os dados chegam do f1-dash;
  - **SESSÃO ENCERRADA** (cinza) quando a sessão terminou;
  - **GRAVAÇÃO** ou **REPRODUZINDO GRAVAÇÃO** (cinza, parado) no modo gravado.
  A animação é desligada para quem pede menos movimento no sistema. Testado com o simulador e o `realtime` do f1-dash servindo a corrida de 2019 como se fosse ao vivo.
- **Rodapé:** "Eduardo Schneider Arruda · engenheiro de computação e pesquisador, graduando de Estatística pela UnB", com a disciplina e o aviso de projeto não oficial.

### Nova divisão: teste com as 3 últimas corridas, e a GRU retreinada

- **Divisão nova, ainda cronológica:**
  - teste: Singapura, Austin e México 2026 (as 3 últimas corridas secas);
  - validação: Zandvoort, Monza, Madring e Baku;
  - treino: tudo até a Hungria 2026 (Spa e Hungaroring passaram para o treino).
  A normalização e os vocabulários foram reajustados ao treino novo.
- **Ingênuo** (`runs/naive_20261106`): MAE de 0,655 s no teste e 0,783 s na validação.
- **GRU** (`runs/gru_20261106_s1`, mesma configuração e semente): melhor época 14, parou na 22.

  | | MAE | Melhora sobre o ingênuo | MAE h = 1 / h = 10 | Viés | Cobertura p10–p90 | Depois de uma parada (GRU × ingênuo) |
  |---|---|---|---|---|---|---|
  | Teste | 0,497 s | +24% | 0,39 / 0,56 s | −0,02 s | 85% | 0,54 × 1,32 s |
  | Validação | 0,618 s | +21% | 0,46 / 0,74 s | +0,10 s | 72% | |

- **Por corrida do teste:** Singapura 0,613 s (ingênuo 0,763), Austin 0,497 s (0,691) e México 0,381 s (0,521).
- **GRU nova × anterior nas mesmas três corridas** (28.220 voltas): 0,495 s contra 0,486 s, com cobertura de 85% nas duas. As duas corridas a mais no treino não mudaram o resultado além da variação esperada entre treinos com uma única semente.
- A cobertura do intervalo ficou acima do alvo no teste (85%) e abaixo na validação (72%), onde está Madring, o circuito que a rede só conhece como `<unk>`. Calibrar melhor o intervalo fica para a v1.
- O backtest (`python -m modeling.backtest`) passou a usar por padrão as corridas do teste do run, e o painel monta o título do teste a partir do run escolhido. O previsor ao vivo usa a GRU mais recente, então domingo em Interlagos roda com este modelo.

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

### Serviço ao vivo: leitura do f1-dash e histórico de voltas

- `live/feed.py`: duas fontes do estado da corrida com a mesma saída. `HttpSource` lê o `localhost:4000/api/current` do f1-dash a cada 3 s e trata o `{}` (sem sessão ou sem conexão) como "aguardando". `ReplaySource` reproduz uma gravação no formato do simulador do f1-dash, aplicando as atualizações com a mesma regra de merge do serviço `realtime` (portada do Rust), para testar sem sessão ao vivo.
- `live/laps.py`: monta o histórico de voltas, porque o feed só traz a última. Quando o contador de voltas de um piloto avança, a volta vira uma linha com tempo, setores, speed traps, composto, stint, idade do pneu, posição, gaps, status de pista (todos os códigos vistos na volta), entrada e saída dos boxes e clima, com os nomes de coluna do treino.
- `live/reader.py`: liga as duas partes e grava `data/live/<ano>_<circuito>_<sessão>/laps.parquet` a cada volta, com escrita atômica.
- **Validação com o GP da Alemanha de 2019**, gravado na pasta do f1-dash: as 1.054 voltas reproduzidas batem uma a uma (piloto e volta) com o Jolpica, e 1.027 dos 1.034 tempos são idênticos ao milésimo. As 7 diferenças estão nas voltas finais, e a soma dos setores do próprio feed confirma o valor ao vivo.
- Dois defeitos de sincronia do feed, encontrados na validação e corrigidos:
  - às vezes o contador de voltas avança antes do tempo da volta ser atualizado. A volta fica pendente até o tempo mudar; se a volta seguinte terminar antes, usa a soma dos setores;
  - a idade do pneu (`TotalLaps`) chega atrasada. Ela passou a ser contada no próprio leitor, volta a volta, a partir das voltas que o jogo já tinha; como no FastF1, a volta de entrada fica no stint antigo e a de saída começa em 1.
- **Intervalo de leitura:** no replay, ler o estado a cada 1, 3, 5 ou 10 s dá exatamente o mesmo histórico; a cada 30 s, setores e idade do pneu começam a errar. O padrão ficou em 3 s.
- O caminho HTTP foi testado de ponta a ponta subindo o simulador e o `realtime` do f1-dash com a gravação de 2019: a sessão é detectada e as voltas são gravadas. O teste foi acelerado (~30×), o que equivale a ler a cada ~30 s de corrida e explica as diferenças de setores nesse teste.

### Serviço ao vivo: histórico de voltas

- **Retomada:** se o leitor for reiniciado no meio da sessão, ele carrega o `laps.parquet` já gravado e continua dele, sem duplicar voltas. As voltas que terminaram com o leitor parado não voltam (o feed só traz a última). Teste no GP da Alemanha de 2019, com o leitor caindo aos 70 min e voltando aos 85: nenhuma volta duplicada, 147 voltas perdidas e o resto igual ao da corrida sem queda.
- **Voltas deletadas por limite de pista:** lidas das mensagens do race control (`CAR 81 (PIA) TIME 1:22.907 DELETED … LAP 5`) e reinstauradas quando houver mensagem. A mensagem chega depois da volta, e a linha já gravada é corrigida. Nas 13 mensagens do GP do México de 2026, as voltas marcadas são exatamente as 13 do TracingInsights.
- `live/history.py` transforma o histórico ao vivo nas variáveis do treino, chamando as próprias funções do `pipeline/race_laps.py`:
  - **referência do fim de semana:** pelo TracingInsights, se o quali já foi publicado, ou pelas sessões de treino e quali gravadas pelo leitor. No México 2026, a referência é idêntica à do pipeline para os 22 pilotos (volta e setores);
  - **progresso da corrida:** pelo total de voltas anunciado no feed, porque ao vivo a corrida ainda não acabou;
  - **voltas perdidas:** entram como voltas sem tempo, para a sequência continuar contínua. O pneu delas é reconstruído pela volta conhecida mais próxima, usando as voltas de uso do jogo montado (`StartLaps`); no teste da queda, composto, stint e idade das 147 voltas perdidas ficaram 100% corretos.
- **Paridade com o pipeline:** as cinco corridas do teste foram refeitas pelo caminho ao vivo (`python -m live.history <race_id>`). Ritmo, setores e speed traps relativos, flags de limpeza, SC/VSC, tráfego, progresso, voltas no stint e composto concordam em 100% das voltas. A única diferença inicial eram as voltas deletadas, resolvida pela leitura do race control.

### Serviço ao vivo: gravador das sessões

- `live/recorder.py` assina o fluxo SSE do f1-dash (`/api/realtime`) e grava cada sessão em `data/live/<ano>_<circuito>_<sessão>/recording.jsonl`. O arquivo segue o formato do simulador do f1-dash: primeira linha com o estado completo no momento da conexão, depois uma mensagem por linha com o tópico, os dados e os milissegundos desde o início da gravação.
- Comportamento:
  - se o gravador for reiniciado na mesma sessão, a nova parte vai para `recording_2.jsonl`, com o seu próprio estado inicial, porque as atualizações perdidas no intervalo tornariam uma continuação inconsistente;
  - quando o f1-dash troca de sessão, o gravador reconecta e abre o arquivo da sessão nova;
  - sem sessão ou sem conexão, ele espera e tenta de novo.
- **Teste:** subindo o simulador e o `realtime` do f1-dash com a gravação de 2019 (acelerada), o gravador salvou 14.415 mensagens em 4 min, até a volta 12.
  - Reproduzida pelo `ReplaySource`, a gravação dá o mesmo histórico da gravação original: 211 voltas, nenhuma faltando, e tempos, setores, pneu, posição, box e status de pista idênticos.
  - Reproduzida pelo próprio simulador do f1-dash, o painel carrega a sessão e as atualizações chegam até as voltas. O simulador anda mais devagar que o fator pedido no Windows (cada pausa curta vira ~15 ms), o que não afeta a gravação.
- O `simulator save` do f1-dash não foi usado: pelo código, ele passa host e hub trocados em relação ao `realtime` e provavelmente não conecta no feed ao vivo.

### Serviço ao vivo: previsões da GRU

- `live/predict.py` acompanha o `laps.parquet` da corrida e, a cada volta nova de um piloto, prevê as 10 voltas seguintes no cenário A (mesmo pneu, sem parada). Grava `predictions.parquet` no formato padrão (`split = live`, mais `predicted_at`, `scenario` e o run do modelo) e preenche o valor real de cada volta prevista quando ela termina, o que dá o erro ao vivo.
- **Mesmo cálculo do treino:** as variáveis vêm de `live/history.py`, a codificação usa a normalização e os vocabulários guardados no `model.pt`, e as colunas derivadas saem de `pipeline.sequences.model_columns`, separada do `prepare` para os dois caminhos usarem o mesmo código (o `dataset.npz` refeito ficou idêntico). A GRU ganhou `forecast()`, que prevê a partir da última volta conhecida sem precisar das voltas futuras; ele dá o mesmo resultado do `forward` do treino (diferença de 1e-7).
- O leitor passou a gravar `session.json` (GP, circuito, data e total de voltas), de onde o previsor tira o circuito (pelo `circuits.csv`) e a referência do fim de semana.
- Não há previsão na volta de entrada nos boxes (o pneu ainda vai mudar) nem em treinos livres (sem total de voltas).
- **Validação com o México 2026:** refazendo a corrida pelo caminho ao vivo, as previsões são idênticas às do backtest (diferença máxima de 2e-8 em p10, p50 e p90) em todas as janelas sem parada no horizonte, que são as em que o cenário A coincide com o que aconteceu. Contando todas as janelas, o erro ao vivo seria de 0,443 s, com cobertura de 83%.
- **Modo contínuo com a Alemanha 2019:** com o histórico crescendo volta a volta, foram 943 previsões, com valor real preenchido à medida que as voltas terminavam, e ~0,4 s por atualização (máximo de 1,5 s), folgado para leituras a cada 3 s.
- Dois defeitos encontrados e corrigidos:
  - o total de voltas anunciado no feed era trocado pela maior volta já disputada, então ao vivo a rede acharia que a corrida terminava na volta atual. O teste de paridade não pegava porque lá a corrida estava completa;
  - um leitor reiniciado contra uma fonte que recomeça do início (como um replay) duplicava voltas já gravadas. Volta já gravada não entra de novo.

### Painel Streamlit (v0)

- `dashboard/app.py`, com tema escuro nas cores do GitHub (`.streamlit/config.toml` e um pouco de CSS) e duas abas:
  - **Ao Vivo:** escolhe a fonte, "f1-dash ao vivo" (lê `/api/current` a cada 3 s, só leitura) ou "sessão gravada" (reproduz uma gravação de `data/live/`, com controle de momento e botão de reproduzir). Mostra o cabeçalho da sessão (GP, volta ou tempo restante, clima e status da pista), uma tabela de cronometragem no estilo do f1-dash (posição, piloto na cor da equipe, melhor e última volta, gaps, setores com roxo e verde, pneu e idade, paradas, box/abandono), os tempos de volta dos pilotos escolhidos com o composto em cada ponto e, na corrida, a previsão mais recente da GRU para o piloto escolhido, com o erro ao vivo acumulado.
  - **GRU v0:** indicadores do teste (MAE, melhora sobre o ingênuo, cobertura, erro depois da parada, tamanho e treino), erro por horizonte na validação e no teste, calibração do intervalo por horizonte, curva de treino, erro por composto e por tarefa A × B, tabela por corrida e exemplos volta a volta de qualquer corrida e piloto da validação e do teste.
- As figuras de `modeling/figures.py` passaram a aceitar tema claro (relatório de backtest) e escuro (painel); as cores do GRU e do ingênuo foram validadas para daltonismo nos dois fundos.
- A gravação do GP da Alemanha de 2019 foi copiada para `data/live/2019_hockenheim_race/` (fora do git), para a aba Ao Vivo ter uma sessão gravada para testar sem sessão ao vivo.
- Conferido no navegador: a cronometragem mostra os 20 carros, os gráficos das duas abas renderizam sem erro, e a seção de previsões aparece quando há `predictions.parquet` (testada com um arquivo temporário, já apagado).

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
