# Análise Exploratória de Dados Aplicada a Gastos Públicos: O Caso da CEAP no Senado Federal (56ª Legislatura)

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

> **Nota:** a RQ3 foi explorada nos notebooks e está refletida nos artefatos do
> repositório (`figures/rq3_anomalias.png`, seções de detecção de anomalias),
> mas **não foi incluída no artigo final**.

## Estrutura do repositório

```
ceap-56-legislatura/
├── data/
│   ├── processed/     # Dados limpos e integrados (os arquivos aparecerão após rodar os notebooks localmente)
│   └── raw/           # CSVs originais da API
├── database/          # SQLite com todos os dados (o arquivo aparecerá após rodar os notebooks localmente)
│   └── ceap.db        
├── figures/           # Gráficos gerados nas fases de Feature Engineering e EDA
│   ├── fe_benford.png              
│   ├── fe_comportamentais.png      
│   ├── rq1_categorias.png
│   ├── rq2_senadores.png
│   ├── rq2_top_fornecedores.png
│   ├── rq3_anomalias.png
│   ├── rq4_pandemia.png
│   ├── rq4_sazonalidade.png
│   └── rq4_serie_temporal.png
├── logs/              # Logs de rastreabilidade
│   ├── coleta_metadata.json                
│   ├── eda_stats.json
│   ├── feature_engineering_log.json        
│   └── preprocessamento_qualidade.json     
├── notebooks/        
│   ├── 01_coleta.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_eda.ipynb
├── src/
│   ├── collect.py      # Coleta via API
│   ├── eda.py          # Análise exploratória de dados
│   ├── features.py     # Engenharia de atributos
│   ├── preprocess.py   # Limpeza e padronização
│   ├── utils.py        # Funções utilitárias
├── .gitignore
├── README.md
└── requirements.txt
```

## Como reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/emanuelob/ceap-56-legislatura.git
cd ceap-56-legislatura

# 2. Crie e ative um ambiente virtual
python3 -m venv .venv       # Linux/macOS
python -m venv .venv        # Windows
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a coleta de dados
python3 src/collect.py      # Linux/macOS
python src/collect.py       # Windows

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
