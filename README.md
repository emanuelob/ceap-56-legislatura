# Análise de Padrões e Anomalias na CEAP - 56ª Legislatura (2019–2023)

**Disciplina:** CIC0203 - Computação Experimental

**Autores:** Emanuel de Oliveira · Leandro Kornelius

---

## Sobre o projeto

Este projeto conduz uma análise empírica das despesas da Cota para o Exercício
da Atividade Parlamentar (CEAP) do Senado Federal durante a **56ª Legislatura
completa (2019-2023)**, seguindo os padrões metodológicos do
[ACM SIGSOFT Empirical Standards - Data Science](https://www2.sigsoft.org/EmpiricalStandards/docs/standards?standard=DataScience#).

## Perguntas de pesquisa

| #   | Pergunta                                                                                   |
| --- | ------------------------------------------------------------------------------------------ |
| RQ1 | Quais categorias de despesa concentram o maior volume financeiro da CEAP?                  |
| RQ2 | É possível identificar grupos de senadores com perfis de gasto estatisticamente distintos? |
| RQ3 | Quais despesas se desviam significativamente do comportamento médio de sua categoria?      |
| RQ4 | Como os padrões de gasto evoluíram ao longo dos 5 anos da legislatura?                     |

## Estrutura do repositório

```
ceap-56-legislatura/
├── data/
│   ├── processed/     # Dados limpos e integrados (os arquivos aparecerão após rodar os notebooks localmente)
│   ├── raw/           # CSVs originais da API
├── database/
│   └── ceap.db        # SQLite com todos os dados (o arquivo aparecerá após rodar os notebooks localmente)
├── logs/
│   └── coleta_metadata.json   # Log de coleta
├── notebooks/
│   ├── 01_coleta.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_feature_engineering.ipynb
├── src/
│   ├── collect.py      # Coleta via API
│   ├── preprocess.py   # Limpeza e padronização
│   ├── features.py     # Engenharia de atributos
│   ├── utils.py        # Funções utilitárias
├── .gitignore
├── README.md
└── requirements.txt
```

## Como reproduzir

```bash
# 1. Clone o repositório
git https://github.com/emanuelob/ceap-56-legislatura.git
cd ceap-56-legislatura

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a coleta de dados
python src/collect.py

# 5. Abra os notebooks em ordem
jupyter lab
#ou
jupyter notebook            # Interface do Jupyter Notebook
```

## Fonte dos dados

- **Portal de Dados Abertos do Senado Federal**
- BASE_URL: https://adm.senado.gov.br/adm-dadosabertos
- Endpoint: `GET /api/v1/senadores/despesas_ceaps/{ano}/csv`
- Acesso: público, sem autenticação
- Log de coleta: `logs/coleta_metadata.json`
