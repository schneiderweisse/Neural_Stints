# Neural Stints

Previsão de ritmo de corrida e degradação de pneus na Fórmula 1 com redes neurais recorrentes (LSTM/GRU).

> Projeto final da disciplina **Redes Neurais 2**, do Bacharelado em Estatística da Universidade de Brasília (UnB). Em desenvolvimento.

## Motivação

Numa equipe de Fórmula 1, a estratégia de corrida depende de saber como o tempo de volta vai evoluir: quanto o carro ganha com um jogo de pneus novos, quanto perde à medida que o pneu envelhece e quando a degradação passa a custar mais do que uma parada nos boxes. Este projeto tenta responder isso com dados públicos e uma rede neural.

## O que a rede faz

A partir das voltas já disputadas por um piloto numa corrida, a rede prevê:

- **o ritmo das próximas voltas** dentro do stint atual;
- **a curva de desempenho do próximo stint** com um pneu novo de um dado composto: o aquecimento inicial, a degradação e a queda final de rendimento (*cliff*).

Cada previsão vem com um intervalo de incerteza (quantis p10, p50 e p90). Com essas curvas, é possível comparar cenários de estratégia, como "parar agora com o composto X" ou "ficar mais *k* voltas na pista".

## Abordagem

![Como as variáveis entram na rede](docs/figures/variaveis_rede.svg)

- **Arquitetura:** rede recorrente encoder–decoder (LSTM e GRU). O encoder lê a corrida até o momento; o decoder recebe o que já se sabe sobre as voltas futuras (idade do pneu, composto, voltas restantes) e prevê os tempos de volta.
- **Alvo:** ritmo relativo à melhor volta do piloto no treino classificatório, para comparar circuitos e temporadas diferentes.
- **Eras de regulamento:** os dados cobrem as temporadas de 2022 a 2026, que têm regulamentos técnicos e especificações de pneu diferentes. Em vez de um modelo por era, uma única rede é condicionada pelo regulamento, pela especificação do pneu e pela dureza do composto (C1 a C6).
- **Comparação:** os modelos são avaliados contra baselines estatísticos (média móvel, modelo linear misto e gradient boosting), com validação por corrida em ordem cronológica.

## Teste ao vivo

Além da avaliação com dados históricos, a rede é testada durante corridas reais. Um serviço em Python lê o estado da corrida de uma instância local do [f1-dash](https://github.com/slowlydev/f1-dash), que consome o feed oficial de live timing da F1, e gera as previsões a cada volta completada.

## Dados

| Fonte | Uso |
|---|---|
| [TracingInsights](https://github.com/TracingInsights) | Tempos de volta, compostos, idade dos pneus, clima e status de pista (2022–2026) |
| [OpenF1](https://openf1.org) | Voltas, stints e pit stops para checagem cruzada |
| [Jolpica F1](https://github.com/jolpica/jolpica-f1) | Tempos de volta oficiais para checagem cruzada |
| Pirelli / F1.com | Compostos nominados para cada Grande Prêmio |
| Live timing da F1, via f1-dash | Dados ao vivo durante as corridas |

Os dados brutos não ficam neste repositório. Cada fonte tem a sua própria licença ou termos de uso. Como baixar e organizar os dados está em [data/README.md](data/README.md).

## Como rodar

Requer Python 3.11 e git. A partir da pasta do repositório:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
py pipeline/fetch_tracinginsights.py
.venv/Scripts/python -m pipeline.build_laps
.venv/Scripts/python -m pipeline.race_laps
.venv/Scripts/python -m pipeline.sequences
```

No macOS ou Linux, use `python3` e `.venv/bin/python`.

## Organização

```
pipeline/   scripts de coleta e preparação dos dados
data/       dados brutos, intermediários e de referência (ver data/README.md)
docs/       documentos e figuras do projeto
```

## Aviso

Projeto acadêmico e não oficial, sem qualquer vínculo com a Fórmula 1, a FIA, a Pirelli ou as equipes. Fórmula 1 e marcas relacionadas pertencem aos seus respectivos donos.
