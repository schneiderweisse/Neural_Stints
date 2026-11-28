# Demo do painel (GitHub Pages)

Versão pública do painel, com tudo o que o painel completo (`dashboard/`) tem, menos a parte ao vivo. Fica em **https://schneiderweisse.github.io/Neural_Stints/** e roda inteiramente no navegador: o [stlite](https://github.com/whitphx/stlite) executa o Streamlit em WebAssembly (Pyodide), sem servidor. A primeira visita leva uns 30 segundos para carregar o Python e os pacotes.

| Aba | O que mostra |
|---|---|
| Corridas gravadas | Os GPs de Las Vegas e de São Paulo de 2026 e o GP da Alemanha de 2019, gravados do feed oficial pelo f1-dash, com cronometragem, pneus, setores e tempos de volta. Dá para arrastar o momento ou apertar Reproduzir. Nas corridas de 2026 aparecem as previsões que a rede fez ao vivo: na cronometragem (a previsão da última volta, com o erro, e a da volta atual), na comparação de dois pilotos e nos duelos (em quantas voltas um carro deve chegar a menos de 1 s do outro, com o que aconteceu de fato). |
| Replay | A corrida de Las Vegas em time-lapse no mapa, pela telemetria, acompanhando um piloto: mais lenta e com zoom nos duelos, com bandeiras, box e mensagens do race control. No modo **Duelos previstos pela rede**, cada duelo que a rede previu é reproduzido desde a volta da previsão, com o gap previsto contra o real. |
| Stints | Um stint de um piloto (o médio de Verstappen em Las Vegas, por exemplo): os tempos de volta reais e a projeção da rede, feita no começo do stint, volta a volta ou a partir de uma volta escolhida, com a degradação real e a projetada. Nas corridas ao vivo e nas de validação e teste. |
| LSTM | O modelo ao vivo por dentro: a arquitetura com as dimensões reais, os pesos por portão, os embeddings, as curvas de treino com a parada antecipada, a busca de hiperparâmetros e a avaliação contra a GRU e o ingênuo, inclusive o erro ao vivo em Las Vegas. |
| GRU | O primeiro modelo, usado ao vivo em São Paulo (média de 5 sementes), contra o baseline ingênuo nas corridas de validação e teste de 2026. |
| Tarefas A e B | Previsto × real para até dois pilotos: as próximas k voltas e o stint inteiro. |

Ficam só no painel completo a leitura do f1-dash durante as sessões (aba Ao Vivo) e a aba Mapa, que estima a posição dos carros ao vivo pelos minissetores. Para as corridas gravadas, o mapa está na aba Replay, com a posição real da telemetria. A corrida de 2019 não tem previsões, porque é anterior aos dados de treino (2022 a 2026). A análise da corrida de São Paulo está no [relatório](../docs/race_reports/2026_sao_paulo.md).

## Arquivos

| Arquivo | Papel |
|---|---|
| `app.py` | Entrada da demo: cabeçalho, as seis seções e o rodapé. |
| `index.html` | Carrega o stlite e monta o app a partir de `app.zip`. |
| `build.py` | `export` gera `files/`; `site` monta o site estático. |
| `files/` | Dados enxutos da demo (commitados, ~13 MB): runs da GRU e do ingênuo; o da LSTM com as previsões enxutas, o histórico, a ficha do modelo (`model_card.json`) e o resumo da busca de hiperparâmetros; voltas das corridas de validação e teste; as reproduções das três corridas a cada 30 s; as previsões ao vivo das corridas de 2026 (e as refeitas de Las Vegas); os dois arquivos da aba Replay e o logo da UnB. |

O cabeçalho usa "F1" em texto, e não o logo da Fórmula 1, que é marca registrada. O logo da UnB vai em `files/assets/`.

## Atualizar

A partir da pasta `Neural_Stints`. Os dados da aba Replay vêm antes do export (depois de uma corrida nova, com a telemetria baixada):

```bash
.venv/Scripts/python -m modeling.model_card runs/lstm_20261120_ens5   # ficha do modelo (aba LSTM), depois de um modelo novo
.venv/Scripts/python -m pipeline.race_replay 2026 "Las Vegas Grand Prix"
.venv/Scripts/python -m live.predict --session 2026_las_vegas_race --once
.venv/Scripts/python -m modeling.duel_replay 2026 "Las Vegas Grand Prix" --live 2026_las_vegas_race
```

```bash
.venv/Scripts/python demo/build.py export --gru runs/gru_20261106_ens5 --naive runs/naive_20261106 --lstm runs/lstm_20261120_ens5   # depois de um modelo novo ou de uma corrida nova (sem as opções, usa os runs mais recentes)
.venv/Scripts/python demo/build.py site     # monta demo/_site para testar
```

O `export` refaz `files/` inteiro. Uma gravação em partes (o gravador reiniciado no meio da corrida, como em Las Vegas) vira uma reprodução só, com o relógio contínuo pelo Heartbeat do feed. Os `replay.json.gz` levam a data no cabeçalho do gzip: se o conteúdo não mudou, volte os antigos com `git checkout` para não criar commits só de data.

Para testar localmente, sirva `demo/_site` por HTTP (o stlite não carrega o zip por `file://`):

```bash
.venv/Scripts/python -m http.server 8770 --directory demo/_site
```

e abra http://localhost:8770. O deploy é automático: a cada push na `main` que mexa em `demo/`, `dashboard/` ou no código que as abas usam, o workflow `.github/workflows/demo.yml` monta o site e publica no GitHub Pages. Antes do primeiro deploy, é preciso escolher **GitHub Actions** em Settings → Pages → Build and deployment → Source.
