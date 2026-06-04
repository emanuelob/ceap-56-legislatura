"""
DOCUMENTAÇÃO (ACM SIGSOFT — 'describes the feature engineering approaches and transformations'):

    Este módulo deriva todas as variáveis analíticas a partir da tabela fato_despesa.
    Nenhuma feature é calculada diretamente nos notebooks de análise; toda a lógica
    de derivação é centralizada aqui para garantir reprodutibilidade e rastreabilidade.

FEATURES DERIVADAS E JUSTIFICATIVAS:

    GRUPO 1 — TEMPORAIS (responde RQ4: evolução ao longo da legislatura)
    periodo_ano_mes    : str "YYYY-MM" — granularidade mensal para séries temporais.
    trimestre          : int 1–4 — agrupamento trimestral para análise sazonal.
    semestre           : int 1–2 — distinção primeiro/segundo semestre.
    ano_eleitoral      : bool — marcador para 2022 (ano de eleições presidenciais),
                         variável de controle para RQ4.
    periodo_pandemia   : str categorica — classifica cada registro em:
                         'pre_pandemia' (< 2020-03), 'pandemia' (2020-03 a 2022-04),
                         'pos_pandemia' (>= 2022-05). Baseado na Portaria GM/MS Nº 913/2022.

    GRUPO 2 — POR SENADOR (responde RQ2: perfis de gasto)
    Calculadas com transform('sum'/'mean'/'count') para preservar granularidade
    do registro original (sem colapsar linhas):

    gasto_total_senador         : float — soma histórica de gastos do senador.
    gasto_medio_mensal_senador  : float — média por período ano-mês.
    n_fornecedores_distintos    : int — diversidade de fornecedores (proxy de
                                  dispersão vs. concentração de gastos).
    n_categorias_distintas      : int — amplitude de uso das categorias CEAP.
    proporcao_maior_categoria   : float [0,1] — concentração na categoria principal
                                  (feature para clustering; alto valor = especialização).

    GRUPO 3 — POR CATEGORIA (responde RQ1: categorias dominantes)
    gasto_total_categoria       : float — volume total por tipo_despesa.
    percentual_categoria_global : float — participação percentual no total geral.
    rank_categoria              : int — ranking por volume (1 = maior).

    GRUPO 4 — DETECÇÃO DE ANOMALIAS (responde RQ3: desvios significativos)
    zscore_categoria     : float — Z-score do valor dentro da categoria.
                           Valores |z| > 3 são candidatos a anomalias.
    iqr_outlier_flag     : bool — flag IQR: valor > Q3 + 1.5*IQR dentro da categoria.
    valor_log1p          : float — log(1 + valor), transformação para normalizar
                           a distribuição fortemente assimétrica dos valores monetários.
                           Necessário antes de algoritmos que assumem normalidade.

    JUSTIFICATIVA DO Z-SCORE VS. IQR:
    Z-score assume distribuição aproximadamente normal; IQR é não-paramétrico.
    Ambos são calculados para que o artigo possa discutir a concordância/discordância
    entre os dois métodos.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# OMS declarou estado de pandemia em 11/03/2020
# Fim da Emergência de Saúde Pública de Importância Nacional (ESPIN) no Brasil foi declarado pelo MS em 22/04/2022.
INICIO_PANDEMIA = pd.Timestamp("2020-03-01")
FIM_PANDEMIA    = pd.Timestamp("2022-04-30")
ANO_ELEITORAL   = 2022

# Features temporais

def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva variáveis temporais a partir de 'data_despesa', 'ano' e 'mes'.

    Registros com data_despesa = NaT recebem NaN nas features baseadas em
    datetime, mas mantêm as features baseadas em 'ano'/'mes' (que vêm
    diretamente das colunas inteiras do CSV original).
    """
    df = df.copy()

    # Granularidade mensal (para séries temporais)
    df['periodo_ano_mes'] = df['data_despesa'].dt.to_period('M').astype(str)

    # Trimestre e semestre (baseados na coluna 'mes' inteira — mais robusta)
    df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype('Int64')
    df['semestre']  = df['mes'].apply(lambda m: 1 if pd.notna(m) and m <= 6 else 2)

    # Marcadores de eventos externos (ano eleitoral e pandemia)
    df['ano_eleitoral'] = (df['ano'] == ANO_ELEITORAL)

    df['periodo_pandemia'] = pd.cut(
        df['data_despesa'],
        bins=[
            pd.Timestamp.min,
            INICIO_PANDEMIA - pd.Timedelta(days=1),
            FIM_PANDEMIA,
            pd.Timestamp.max,
        ],
        labels=['pre_pandemia', 'pandemia', 'pos_pandemia'],
        right=True,
    )

    return df

# Features por senador

def adicionar_features_senador(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas agregadas por senador usando transform() para preservar
    o índice original (sem colapsar linhas — operação window, não group-by).
    """
    df = df.copy()
    grp = df.groupby('cod_senador')

    df['gasto_total_senador'] = grp['valor_limpo'].transform('sum')

    # Média mensal: soma por (senador, periodo) → média sobre os períodos
    soma_mensal = (
        df.groupby(['cod_senador', 'periodo_ano_mes'])['valor_limpo']
          .sum()
          .reset_index()
          .rename(columns={'valor_limpo': '_soma_mes'})
    )
    media_mensal = (
        soma_mensal.groupby('cod_senador')['_soma_mes']
                   .mean()
                   .reset_index()
                   .rename(columns={'_soma_mes': 'gasto_medio_mensal_senador'})
    )
    df = df.merge(media_mensal, on='cod_senador', how='left')

    # Diversidade de fornecedores
    df['n_fornecedores_distintos'] = grp['documento_fornecedor'].transform('nunique')

    # Amplitude categórica
    df['n_categorias_distintas'] = grp['tipo_despesa_limpo'].transform('nunique')

    # Concentração na categoria principal (proporção do gasto total na categoria mais frequente)
    gasto_por_categoria = (
        df.groupby(['cod_senador', 'tipo_despesa_limpo'])['valor_limpo']
          .sum()
          .reset_index()
    )
    gasto_total_por_senador = (
        gasto_por_categoria.groupby('cod_senador')['valor_limpo']
                           .sum()
                           .reset_index()
                           .rename(columns={'valor_limpo': '_total'})
    )
    gasto_por_categoria = gasto_por_categoria.merge(gasto_total_por_senador, on='cod_senador')
    gasto_por_categoria['_prop'] = (
        gasto_por_categoria['valor_limpo'] / gasto_por_categoria['_total']
    )
    proporcao_max = (
        gasto_por_categoria.groupby('cod_senador')['_prop']
                           .max()
                           .reset_index()
                           .rename(columns={'_prop': 'proporcao_maior_categoria'})
    )
    df = df.merge(proporcao_max, on='cod_senador', how='left')

    return df

# Features por categoria

def adicionar_features_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula volume e participação percentual por tipo de despesa.
    """
    df = df.copy()
    grp = df.groupby('tipo_despesa_limpo')

    df['gasto_total_categoria'] = grp['valor_limpo'].transform('sum')

    total_geral = df['valor_limpo'].sum()
    df['percentual_categoria_global'] = (df['gasto_total_categoria'] / total_geral * 100).round(4)

    # Ranking por volume (calculado no nível de categoria, depois mapeado)
    rank_map = (
        df[['tipo_despesa_limpo', 'gasto_total_categoria']]
          .drop_duplicates()
          .sort_values('gasto_total_categoria', ascending=False)
          .reset_index(drop=True)
    )
    rank_map['rank_categoria'] = rank_map.index + 1
    df = df.merge(
        rank_map[['tipo_despesa_limpo', 'rank_categoria']],
        on='tipo_despesa_limpo', how='left'
    )

    return df

# Features de detecção de anomalias

def adicionar_features_anomalia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de desvio dentro de cada categoria de despesa.

    Z-score e IQR são calculados separadamente por categoria porque distribuições 
    de valores diferem substancialmente entre categorias (ex.: passagens aéreas vs. refeições). 
    Calcular sobre a base inteira produziria falsos positivos nas categorias de alto valor médio.

    Antipadrão evitado: não usar apenas Z-score sem verificar normalidade da distribuição. 
    Por isso, também calculamos a flag IQR, que é não-paramétrica.
    """
    df = df.copy()

    # Transformação logarítmica (pré-requisito para métricas que assumem simetria)
    df['valor_log1p'] = np.log1p(df['valor_limpo'])

    # Z-score por categoria (assume distribuição aproximadamente normal)
    def zscore(x):
        std = x.std()
        if std == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    df['zscore_categoria'] = (
        df.groupby('tipo_despesa_limpo')['valor_log1p']
          .transform(zscore)
    )

    # Flag IQR por categoria (não-paramétrico)
    def iqr_flag(x):
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        return x > (q3 + 1.5 * iqr)

    df['iqr_outlier_flag'] = (
        df.groupby('tipo_despesa_limpo')['valor_limpo']
          .transform(iqr_flag)
    )

    return df

# Pipeline completo de features

def engenharia_de_features(df_fato: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica sequencialmente todos os grupos de features.

    A ordem importa: features temporais são pré-requisito para features
    de senador (que usam 'periodo_ano_mes').

    Parâmetros:
    df_fato : pd.DataFrame
        Saída limpa do pipeline de pré-processamento.

    Retorno:
    pd.DataFrame com todas as features derivadas.
    """
    print("  [1/4] Features temporais...")
    df = adicionar_features_temporais(df_fato)

    print("  [2/4] Features por senador...")
    df = adicionar_features_senador(df)

    print("  [3/4] Features por categoria...")
    df = adicionar_features_categoria(df)

    print("  [4/4] Features de detecção de anomalias...")
    df = adicionar_features_anomalia(df)

    return df

# Inventário de features para documentação e referência nos notebooks de análise

INVENTARIO_FEATURES = {
    "temporais": [
        "periodo_ano_mes", "trimestre", "semestre",
        "ano_eleitoral", "periodo_pandemia",
    ],
    "por_senador": [
        "gasto_total_senador", "gasto_medio_mensal_senador",
        "n_fornecedores_distintos", "n_categorias_distintas",
        "proporcao_maior_categoria",
    ],
    "por_categoria": [
        "gasto_total_categoria", "percentual_categoria_global", "rank_categoria",
    ],
    "anomalia": [
        "valor_log1p", "zscore_categoria", "iqr_outlier_flag",
    ],
}
