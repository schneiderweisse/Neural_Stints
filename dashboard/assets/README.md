# Logos do painel

O cabeçalho do painel mostra dois logos, se os arquivos estiverem nesta pasta:

| Arquivo | Logo |
|---|---|
| `f1_logo.svg` ou `f1_logo.png` | Fórmula 1 (versão atual, em vermelho) |
| `unb_logo.svg` ou `unb_logo.png` | Universidade de Brasília |

Os arquivos usados vêm do Wikimedia Commons:

- `f1_logo.svg`: [File:Formula One logo.svg](https://commons.wikimedia.org/wiki/File:Formula_One_logo.svg), o logo atual (desde 2023, com o símbolo ®);
- `unb_logo.png`: [File:Universidade of Brasilia logo.png](https://commons.wikimedia.org/wiki/File:Universidade_of_Brasilia_logo.png), a versão que a UnB fornece para fundos escuros.

Os dois são marcas registradas dos seus donos. Por isso os arquivos ficam fora do git (ver `.gitignore`) e não são distribuídos com o repositório: cada um baixa os logos das fontes oficiais. O painel funciona sem eles. Este é um projeto acadêmico e não oficial, sem vínculo com a Fórmula 1.

A demo pública (`demo/`) não usa o logo da F1: o cabeçalho dela tem "F1" em texto. O logo da UnB da demo fica em `demo/files/assets/`.
