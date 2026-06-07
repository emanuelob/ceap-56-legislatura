"""
DOCUMENTAÇÃO (ACM SIGSOFT — 'describes the feature engineering approaches and transformations'):

    Este módulo deriva todas as variáveis analíticas a partir da tabela fato_despesa.
    Nenhuma feature é calculada diretamente nos notebooks de análise; toda a lógica
    de derivação é centralizada aqui para garantir reprodutibilidade e rastreabilidade.

    As feaatures são organizadas em quatro grupos temáticos, cada um respondendo a uma questão de pesquisa específica:
        RQ1	Quais categorias de despesa concentram o maior volume financeiro da CEAP?
        RQ2	É possível identificar grupos de senadores com perfis de gasto estatisticamente distintos?
        RQ3	Quais despesas se desviam significativamente do comportamento médio de sua categoria?
        RQ4	Como os padrões de gasto evoluíram ao longo dos 5 anos da legislatura?

GRUPO 1 — TEMPORAIS  (RQ4)
periodo_ano_mes  : str "YYYY-MM" — granularidade mensal para séries temporais.
trimestre        : int 1-4 — agrupamento trimestral para análise sazonal.
semestre         : int 1-2 — distinção primeiro/segundo semestre.
ano_eleitoral    : bool — marcador para 2022; variável de controle para RQ4.

is_fim_de_semana : bool — True se data_despesa for sábado (dayofweek=5) ou
                   domingo (dayofweek=6).
                   JUSTIFICATIVA: a CEAP cobre atividade parlamentar formal,
                   concentrada em dias úteis (sessões, audiências, viagens de
                   trabalho). Gastos elevados em fins de semana nas categorias
                   de hospedagem e alimentação constituem desvio comportamental
                   mensurável. A feature é calculada por registro e depois
                   sumarizada por senador para RQ2 (proporcao_fds_por_senador).
                   Não implica fraude per se, é apenas curiosidade.

periodo_pandemia : str categórica — classifica cada registro em:
                   'pre_pandemia'  (< 2020-03-01)
                   'pandemia'      (2020-03-01 a 2022-04-30)
                   'pos_pandemia'  (>= 2022-05-01)
                   BASE LEGAL: OMS declarou pandemia em 11/03/2020. O Ministério
                   da Saúde encerrou a ESPIN pela Portaria GM/MS nº 913/2022 em 22/04/2022.

GRUPO 2 — POR SENADOR  (RQ2)
Calculadas com transform() para preservar granularidade do registro original.

gasto_total_senador          : float — soma histórica da legislatura.
gasto_medio_mensal_senador   : float — média de gasto por período ano-mês.
n_fornecedores_distintos     : int — diversidade de CNPJ/CPF usados.
n_categorias_distintas       : int — amplitude de categorias CEAP utilizadas.
proporcao_maior_categoria    : float [0,1] — concentração na categoria de maior
                               gasto (specialization index).
proporcao_fds_por_senador    : float [0,1] — fração dos gastos do senador que
                               ocorreu em fins de semana. Sumariza is_fim_de_semana
                               no nível de perfil para clustering (RQ2).
proporcao_fim_mes_senador    : float [0,1] — fração do gasto mensal concentrada
                               nos últimos 7 dias do mês (dias >= 25).
                               JUSTIFICATIVA: senadores com cota não utilizada têm
                               economizado (sem utilizar a cota) ou concentrado 
                               gastos no fim do mês para evitar estorno?
proporcao_maior_fornecedor   : float [0,1] — proporção do gasto total direcionada
                               a um único CNPJ/CPF.
                               JUSTIFICATIVA: enquanto proporcao_maior_categoria
                               mede especialização setorial (pode ser legítima),
                               esta feature mede dependência de um único fornecedor,
                               padrão associado a direcionamento de contratos e
                               empresas de fachada em auditorias forenses (OCDE,
                               2016 — Detecting Bid Rigging in Public Procurement).
                               Fornecedor "NAO_INFORMADO" é excluído do cálculo
                               para evitar distorção sistemática.

GRUPO 3 — POR CATEGORIA  (responde RQ1)
gasto_total_categoria        : float — volume total acumulado da legislatura.
percentual_categoria_global  : float — participação % no total geral.
rank_categoria               : int — ranking por volume (1 = maior).

GRUPO 4 — DETECÇÃO DE ANOMALIAS  (responde RQ3)
valor_log1p          : float — log(1 + valor_limpo). Pré-condicionamento para
                       algoritmos que assumem distribuições simétricas. Necessário
                       porque a distribuição de valores CEAP é fortemente
                       assimétrica à direita (skewness > 5 na base bruta).

zscore_categoria     : float — Z-score de valor_log1p dentro da categoria.
                       Calculado por categoria para evitar falsos positivos
                       entre categorias de escalas distintas. |z| > 3 = candidato
                       a anomalia sob hipótese de normalidade aproximada.

iqr_outlier_flag     : bool — valor > Q3 + 1.5xIQR dentro da categoria.
                       Método não-paramétrico complementar ao Z-score. A
                       concordância entre ambos (anomalia_consenso) será
                       discutida na seção de resultados.

primeiro_digito      : int 1-9 — primeiro dígito significativo de valor_limpo.
                       JUSTIFICATIVA (Lei de Benford): em conjuntos financeiros
                       naturais com múltiplas ordens de magnitude, o primeiro
                       dígito segue P(d) = log10(1 + 1/d) (Benford, 1938;
                       Nigrini, 2012). A CEAP é adequada: valores de R$0,50 a
                       R$30.000+ cobrem ~5 ordens de magnitude. LIMITAÇÃO:
                       desvio de Benford é sinal de alerta, nunca conclusão
                       isolada de fraude; será tratado como feature de entrada
                       para a modelagem (RQ3).

anomalia_consenso    : bool — True se |zscore_categoria| > 3 E iqr_outlier_flag.
                       Registros sinalizados por ambos os métodos possuem maior
                       grau de evidência e são priorizados na análise qualitativa.
"""

import numpy as np
import pandas as pd

# Constantes — eventos externos documentados
INICIO_PANDEMIA    = pd.Timestamp("2020-03-01")
FIM_PANDEMIA       = pd.Timestamp("2022-04-30")
ANO_ELEITORAL      = 2022
DIA_INICIO_FIM_MES = 25  # dias >= 25 classificados como "fim de mês"

# GRUPO 1 — Features Temporais
def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva variáveis temporais a partir de 'data_despesa', 'ano' e 'mes'.

    Registros com data_despesa = NaT recebem NaN nas features baseadas em
    datetime, mas mantêm as features baseadas em 'ano'/'mes' (colunas inteiras
    do CSV original), garantindo que nenhum registro seja descartado.
    """
    df = df.copy()

    # Granularidade mensal — pré-requisito para features de senador (Grupo 2)
    df['periodo_ano_mes'] = df['data_despesa'].dt.to_period('M').astype(str)

    # Trimestre e semestre — baseados em 'mes' inteiro (mais robusto que datetime)
    df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype('Int64')
    df['semestre']  = df['mes'].apply(
        lambda m: 1 if pd.notna(m) and m <= 6 else 2
    )

    # Variável de controle: ano eleitoral
    df['ano_eleitoral'] = (df['ano'] == ANO_ELEITORAL)

    # Classificação de período pandemia
    df['periodo_pandemia'] = pd.cut(
        df['data_despesa'],
        bins=[
            pd.Timestamp("1900-01-01"),
            INICIO_PANDEMIA - pd.Timedelta(days=1),
            FIM_PANDEMIA,
            pd.Timestamp("2100-01-01"),
        ],
        labels=['pre_pandemia', 'pandemia', 'pos_pandemia'],
        right=True,
    )

    # Feature: gasto em fim de semana
    # NaT produz NaN — registros sem data não são marcados como FDS
    df['is_fim_de_semana'] = df['data_despesa'].dt.dayofweek >= 5

    return df

# GRUPO 2 — Features por Senador
def adicionar_features_senador(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas agregadas por senador via transform() para preservar
    a granularidade do registro original.

    Todas as features são escalares por senador mapeados de volta ao DataFrame
    completo, tornando-as diretamente utilizáveis como input de clustering (RQ2).
    """
    df = df.copy()
    grp = df.groupby('cod_senador')

    # 2.1 Gasto total histórico
    df['gasto_total_senador'] = grp['valor_limpo'].transform('sum')

    # 2.2 Média mensal de gastos (dois passos para evitar inflação pelo número de registros por mês)
    # passo 1: soma por (senador, mês); 
    # passo 2: média das somas mensais
    soma_mensal = (
        df.groupby(['cod_senador', 'periodo_ano_mes'])['valor_limpo']
          .sum()
          .reset_index(name='_soma_mes')
    )
    media_mensal = (
        soma_mensal.groupby('cod_senador')['_soma_mes']
                   .mean()
                   .reset_index()
                   .rename(columns={'_soma_mes': 'gasto_medio_mensal_senador'})
    )
    df = df.merge(media_mensal, on='cod_senador', how='left')

    # 2.3 Diversidade de fornecedores
    df['n_fornecedores_distintos'] = grp['documento_fornecedor'].transform('nunique')

    # 2.4 Amplitude categórica
    df['n_categorias_distintas'] = grp['tipo_despesa_limpo'].transform('nunique')

    # 2.5 Concentração na categoria principal
    gasto_cat = (
        df.groupby(['cod_senador', 'tipo_despesa_limpo'])['valor_limpo']
          .sum()
          .reset_index(name='_gasto_cat')
    )
    total_sen = (
        gasto_cat.groupby('cod_senador')['_gasto_cat']
                 .sum()
                 .reset_index(name='_total')
    )
    gasto_cat = gasto_cat.merge(total_sen, on='cod_senador')
    gasto_cat['_prop'] = gasto_cat['_gasto_cat'] / gasto_cat['_total']
    prop_max_cat = (
        gasto_cat.groupby('cod_senador')['_prop']
                 .max()
                 .reset_index()
                 .rename(columns={'_prop': 'proporcao_maior_categoria'})
    )
    df = df.merge(prop_max_cat, on='cod_senador', how='left')

    # 2.6 Proporção de gastos em fins de semana (perfil comportamental)
    # Exclui NaT do denominador para não subestimar a proporção
    fds_por_senador = (
        df.dropna(subset=['is_fim_de_semana'])
          .groupby('cod_senador')['is_fim_de_semana']
          .mean()
          .reset_index()
          .rename(columns={'is_fim_de_semana': 'proporcao_fds_por_senador'})
    )
    df = df.merge(fds_por_senador, on='cod_senador', how='left')

    # 2.7 Burn-rate: proporção do gasto mensal nos últimos dias do mês
    # Calculada como media(soma_fim_mes / soma_mes) por senador
    # Registros sem data são excluídos desta janela
    df_com_data = df.dropna(subset=['data_despesa']).copy()
    df_com_data['_is_fim_mes'] = df_com_data['data_despesa'].dt.day >= DIA_INICIO_FIM_MES

    def _prop_fim_mes(g):
        total = g['valor_limpo'].sum()
        if total <= 0:
            return 0.0
        return g.loc[g['_is_fim_mes'], 'valor_limpo'].sum() / total

    prop_fim_mes_mensal = (
        df_com_data
        .groupby(['cod_senador', 'periodo_ano_mes'])
        .apply(_prop_fim_mes, include_groups=False)
        .reset_index(name='_prop_fim_mes')
    )
    prop_fim_mes_senador = (
        prop_fim_mes_mensal
        .groupby('cod_senador')['_prop_fim_mes']
        .mean()
        .reset_index()
        .rename(columns={'_prop_fim_mes': 'proporcao_fim_mes_senador'})
    )
    df = df.merge(prop_fim_mes_senador, on='cod_senador', how='left')

    # 2.8 Concentração no maior fornecedor
    #Obs.: Exclui "NAO_INFORMADO"
    df_com_cnpj = df[df['documento_fornecedor'] != 'NAO_INFORMADO'].copy()
    gasto_fornecedor = (
        df_com_cnpj
        .groupby(['cod_senador', 'documento_fornecedor'])['valor_limpo']
        .sum()
        .reset_index(name='_gasto_cnpj')
    )
    max_fornecedor = (
        gasto_fornecedor.groupby('cod_senador')['_gasto_cnpj']
                        .max()
                        .reset_index(name='_max_cnpj')
    )
    total_com_cnpj = (
        df_com_cnpj.groupby('cod_senador')['valor_limpo']
                   .sum()
                   .reset_index(name='_total_com_cnpj')
    )
    max_fornecedor = max_fornecedor.merge(total_com_cnpj, on='cod_senador', how='left')
    max_fornecedor['proporcao_maior_fornecedor'] = (
        max_fornecedor['_max_cnpj'] / max_fornecedor['_total_com_cnpj']
    )
    df = df.merge(
        max_fornecedor[['cod_senador', 'proporcao_maior_fornecedor']],
        on='cod_senador', how='left'
    )

    return df

# GRUPO 3 — Features por Categoria
def adicionar_features_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula volume total, participação percentual e ranking por categoria.
    """
    df = df.copy()

    df['gasto_total_categoria'] = (
        df.groupby('tipo_despesa_limpo')['valor_limpo'].transform('sum')
    )

    total_geral = df['valor_limpo'].sum()
    df['percentual_categoria_global'] = (
        df['gasto_total_categoria'] / total_geral * 100
    ).round(4)

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

# GRUPO 4 — Features de Detecção de Anomalias
def adicionar_features_anomalia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores de desvio estatístico para cada registro.

    Z-score e IQR são calculados DENTRO de cada categoria, não sobre a base
    inteira, para evitar falsos positivos entre categorias de escalas distintas
    """
    df = df.copy()

    # 4.1 Transformação logarítmica
    df['valor_log1p'] = np.log1p(df['valor_limpo'])

    # 4.2 Z-score por categoria (sobre valor_log1p)
    def _zscore_group(x: pd.Series) -> pd.Series:
        std = x.std(ddof=1)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    df['zscore_categoria'] = (
        df.groupby('tipo_despesa_limpo')['valor_log1p']
          .transform(_zscore_group)
    )

    # 4.3 Flag IQR por categoria (não-paramétrico)
    def _iqr_flag(x: pd.Series) -> pd.Series:
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        return x > (q3 + 1.5 * iqr)

    df['iqr_outlier_flag'] = (
        df.groupby('tipo_despesa_limpo')['valor_limpo']
          .transform(_iqr_flag)
    )

    # 4.4 Primeiro dígito significativo (Lei de Benford)
    # decimal e zeros à esquerda, para capturar o dígito correto em valores < 1.
    # Exemplo: 0.85 → "0.8500000000" → "08500000000" → "8500000000" → '8'
    def _primeiro_digito(x: float) -> int:
        if pd.isna(x) or x <= 0:
            return 0
        s = f"{x:.10f}".replace('.', '').lstrip('0')
        return int(s[0]) if s else 0

    df['primeiro_digito'] = df['valor_limpo'].apply(_primeiro_digito)

    # 4.5 Concordância entre métodos de detecção (flag de consenso)
    zscore_flag = df['zscore_categoria'].abs() > 3
    df['anomalia_consenso'] = zscore_flag & df['iqr_outlier_flag']

    return df

# PIPELINE PRINCIPAL
def engenharia_de_features(df_fato: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica sequencialmente todos os grupos de features.

    ORDEM IMPORTA:
      Grupo 1 (temporais) precede Grupo 2 (senador) pois 'periodo_ano_mes'
      e 'is_fim_de_semana' são pré-requisitos das features de senador.
      Grupos 3 e 4 são independentes entre si.

    Parâmetros:
    df_fato : pd.DataFrame
        Saída do pipeline de pré-processamento (tabela fato_despesa).

    Retorno:
    pd.DataFrame com todas as features derivadas, sem colapso de linhas.
    """
    print("[1/4] Features temporais.")
    df = adicionar_features_temporais(df_fato)

    print("[2/4] Features por senador.")
    df = adicionar_features_senador(df)

    print("[3/4] Features por categoria.")
    df = adicionar_features_categoria(df)

    print("[4/4] Features de anomalia.")
    df = adicionar_features_anomalia(df)

    return df

# Inventário para documentação e referência nos notebooks
INVENTARIO_FEATURES = {
    "temporais": [
        "periodo_ano_mes", "trimestre", "semestre",
        "ano_eleitoral", "is_fim_de_semana", "periodo_pandemia",
    ],
    "por_senador": [
        "gasto_total_senador", "gasto_medio_mensal_senador",
        "n_fornecedores_distintos", "n_categorias_distintas",
        "proporcao_maior_categoria", "proporcao_fds_por_senador",
        "proporcao_fim_mes_senador", "proporcao_maior_fornecedor",
    ],
    "por_categoria": [
        "gasto_total_categoria", "percentual_categoria_global", "rank_categoria",
    ],
    "anomalia": [
        "valor_log1p", "zscore_categoria", "iqr_outlier_flag",
        "primeiro_digito", "anomalia_consenso",
    ],
}

# Subconjunto de features contínuas por senador para clustering (RQ2)
FEATURES_CLUSTERING_SENADOR = [
    "gasto_total_senador",
    "gasto_medio_mensal_senador",
    "n_fornecedores_distintos",
    "n_categorias_distintas",
    "proporcao_maior_categoria",
    "proporcao_fds_por_senador",
    "proporcao_fim_mes_senador",
    "proporcao_maior_fornecedor",
]
