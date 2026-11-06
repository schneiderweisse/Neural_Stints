# Diário de desenvolvimento

Registro do que foi feito e decidido em cada dia de trabalho, da entrada mais recente para a mais antiga.

## 6 de novembro de 2026

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
