# Demo do painel (GitHub Pages)

Versão pública do painel, sem a parte ao vivo. Fica em **https://schneiderweisse.github.io/Neural_Stints/** e roda inteiramente no navegador: o [stlite](https://github.com/whitphx/stlite) executa o Streamlit em WebAssembly (Pyodide), sem servidor. A primeira visita leva uns 30 segundos para carregar o Python e os pacotes.

| Aba | O que mostra |
|---|---|
| Reprodução 2019 | O GP da Alemanha de 2019, gravado do feed oficial pelo f1-dash, com cronometragem, pneus, setores e tempos de volta. Dá para arrastar o momento ou apertar Reproduzir. |
| GRU v0 | O ajuste da rede (média de 5 sementes) contra o baseline ingênuo nas corridas de validação e teste de 2026. |
| Tarefas A e B | Previsto × real para até dois pilotos: as próximas k voltas e o stint inteiro. |

As abas são as mesmas do painel completo (`dashboard/`). A aba Ao Vivo, que lê o f1-dash durante as sessões, fica só no painel completo. A corrida de 2019 não tem previsões da GRU, porque é anterior aos dados de treino (2022 a 2026).

## Arquivos

| Arquivo | Papel |
|---|---|
| `app.py` | Entrada da demo: cabeçalho, as três abas e o rodapé. |
| `index.html` | Carrega o stlite e monta o app a partir de `app.zip`. |
| `build.py` | `export` gera `files/`; `site` monta o site estático. |
| `files/` | Dados enxutos da demo (commitados, ~4,5 MB): runs da GRU e do ingênuo, voltas das corridas de validação e teste, a reprodução de 2019 a cada 30 s e o logo da UnB. |

O cabeçalho usa "F1" em texto, e não o logo da Fórmula 1, que é marca registrada. O logo da UnB vai em `files/assets/`.

## Atualizar

A partir da pasta `Neural_Stints`:

```bash
.venv/Scripts/python demo/build.py export   # depois de treinar um modelo novo (usa a GRU e o ingênuo mais recentes)
.venv/Scripts/python demo/build.py site     # monta demo/_site para testar
```

Para testar localmente, sirva `demo/_site` por HTTP (o stlite não carrega o zip por `file://`):

```bash
.venv/Scripts/python -m http.server 8770 --directory demo/_site
```

e abra http://localhost:8770. O deploy é automático: a cada push na `main` que mexa em `demo/`, `dashboard/` ou no código que as abas usam, o workflow `.github/workflows/demo.yml` monta o site e publica no GitHub Pages. Antes do primeiro deploy, é preciso escolher **GitHub Actions** em Settings → Pages → Build and deployment → Source.
