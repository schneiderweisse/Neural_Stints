# Diário de desenvolvimento

Registro do que foi feito e decidido em cada dia de trabalho, da entrada mais recente para a mais antiga.

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
