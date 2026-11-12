# Demo do painel (GitHub Pages)

Versão pública do painel, sem a parte ao vivo. Fica em **https://schneiderweisse.github.io/Neural_Stints/** e roda inteiramente no navegador: o [stlite](https://github.com/whitphx/stlite) executa o Streamlit em WebAssembly (Pyodide), sem servidor. A primeira visita leva uns 30 segundos para carregar o Python e os pacotes.

| Aba | O que mostra |
|---|---|
| Corridas gravadas | O GP de São Paulo de 2026 e o GP da Alemanha de 2019, gravados do feed oficial pelo f1-dash, com cronometragem, pneus, setores e tempos de volta. Dá para arrastar o momento ou apertar Reproduzir. Na corrida de 2026 aparecem as previsões que a GRU fez ao vivo, com a comparação de dois pilotos, e os duelos: em quantas voltas um carro deve chegar a menos de 1 s do outro, com o que aconteceu de fato. |
| GRU v0 | O ajuste da rede (média de 5 sementes) contra o baseline ingênuo nas corridas de validação e teste de 2026. |
| Tarefas A e B | Previsto × real para até dois pilotos: as próximas k voltas e o stint inteiro. |

As abas são as mesmas do painel completo (`dashboard/`). A aba Ao Vivo, que lê o f1-dash durante as sessões, fica só no painel completo. A corrida de 2019 não tem previsões da GRU, porque é anterior aos dados de treino (2022 a 2026). A análise da corrida de 2026 está no [relatório](../docs/race_reports/2026_sao_paulo.md).

## Arquivos

| Arquivo | Papel |
|---|---|
| `app.py` | Entrada da demo: cabeçalho, as três abas e o rodapé. |
| `index.html` | Carrega o stlite e monta o app a partir de `app.zip`. |
| `build.py` | `export` gera `files/`; `site` monta o site estático. |
| `files/` | Dados enxutos da demo (commitados, ~5 MB): runs da GRU e do ingênuo, voltas das corridas de validação e teste, as reproduções das duas corridas a cada 30 s, as previsões ao vivo de 2026 e o logo da UnB. |

O cabeçalho usa "F1" em texto, e não o logo da Fórmula 1, que é marca registrada. O logo da UnB vai em `files/assets/`.

## Atualizar

A partir da pasta `Neural_Stints`:

```bash
.venv/Scripts/python demo/build.py export --gru runs/gru_20261106_ens5 --naive runs/naive_20261106   # depois de um modelo novo ou de uma corrida nova (sem as opções, usa os runs mais recentes)
.venv/Scripts/python demo/build.py site     # monta demo/_site para testar
```

Para testar localmente, sirva `demo/_site` por HTTP (o stlite não carrega o zip por `file://`):

```bash
.venv/Scripts/python -m http.server 8770 --directory demo/_site
```

e abra http://localhost:8770. O deploy é automático: a cada push na `main` que mexa em `demo/`, `dashboard/` ou no código que as abas usam, o workflow `.github/workflows/demo.yml` monta o site e publica no GitHub Pages. Antes do primeiro deploy, é preciso escolher **GitHub Actions** em Settings → Pages → Build and deployment → Source.
