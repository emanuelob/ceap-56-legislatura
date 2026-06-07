"""
DOCUMENTAÇÃO (ACM SIGSOFT — 'explains how the data was pre-processed, filtered, and categorized'):

    Este módulo implementa a camada de transformação entre a área de staging (stg_ceap)
    e a tabela analítica (fato_despesa). Todas as operações são determinísticas,
    reversíveis na raiz (dado bruto preservado) e documentadas abaixo.

DECISÕES DE DESIGN E JUSTIFICATIVAS:

    1. CONVERSÃO MONETÁRIA
       O CSV do Senado usa o padrão pt-BR: separador de milhar = '.' e decimal = ','.
       Ex.: "2.102,89" → 2102.89. A remoção do ponto antes da vírgula é aplicada
       nessa ordem para evitar ambiguidade (ex.: "1.200" seria lido como 1.2 pelo
       parser padrão do Python sem essa etapa prévia).

    2. SANITIZAÇÃO DE DOCUMENTOS (CNPJ/CPF)
       Máscaras como "09.798.307/0001-54" são removidas via regex [^0-9], retendo
       apenas dígitos. Isso normaliza chaves de junção com outras bases públicas
       (ex.: CNPJ da Receita Federal) que armazenam documentos sem formatação.
       Documentos ausentes ou inválidos recebem a sentinel "NAO_INFORMADO" em vez
       de NaN, para preservar a contagem de registros em GROUP BY sem descarte
       silencioso de linhas.

    3. DATAS
       Conversão para datetime com errors='coerce': datas malformadas viram NaT
       e são sinalizadas no relatório de qualidade, em vez de lançar exceção que
       abortaria o pipeline inteiro.

    4. PADRONIZAÇÃO TEXTUAL
       upper() + strip() nos campos categóricos evita a fragmentação de entidades
       em agrupamentos futuros. Ex.: "Humberto Costa" e "HUMBERTO COSTA " seriam
       contados separadamente em um GROUP BY sem essa etapa.

    5. REMOÇÃO DE VALORES MONETÁRIOS NEGATIVOS E ZERADOS 
       Registros com valor_limpo <= 0 representam estornos ou entradas de dados
       inválidas. São isolados em um DataFrame separado (df_estornos) e excluídos
       da tabela fato para não contaminar estatísticas descritivas. 

    6. RELATÓRIO DE QUALIDADE DE DADOS 
       A função gera um dicionário com métricas de qualidade: total de registros
       antes/depois da filtragem, NaT em datas, nulos por coluna. Esse relatório
       atende ao requisito ACM SIGSOFT de 'manually inspects some non-trivial
       portion of the data (i.e. data sanity checks)'.

    7. TIPAGEM DE COLUNAS INTEIRAS 
       'ano' e 'mes' são explicitamente convertidos para Int64 (nullable integer
       do pandas) para possibilitar operações temporais sem coerção para float.
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime

def limpar_moeda(valor_str) -> float:
    """
    Converte strings monetárias do formato brasileiro (pt-BR) para float IEEE 754.

    Obs.:: O separador de milhar '.' deve ser removido ANTES da
    substituição da vírgula por ponto; caso contrário, "1.200,50" → "1.200.50"
    geraria ValueError no float().
    """
    if pd.isna(valor_str):
        return 0.0
    valor_str = str(valor_str).strip().replace('.', '').replace(',', '.')
    try:
        return float(valor_str)
    except ValueError:
        return 0.0

def limpar_documento(doc_str) -> str:
    """
    Remove pontuações de CNPJ/CPF para padronização de chaves de junção.

    Retorna "NAO_INFORMADO" para valores ausentes ou strings vazias após
    limpeza, evitando NaN que seria descartado silenciosamente em GROUP BY.
    """
    if pd.isna(doc_str):
        return "NAO_INFORMADO"
    limpo = re.sub(r'[^0-9]', '', str(doc_str))
    return limpo if limpo else "NAO_INFORMADO"

def padronizar_string(texto_str) -> str:
    """
    Converte para MAIÚSCULAS e remove espaços excedentes (leading/trailing/internal).

    Aplica também normalização de múltiplos espaços internos para evitar que
    "HUMBERTO  COSTA" (dois espaços) e "HUMBERTO COSTA" sejam tratados como
    entidades distintas em agrupamentos.
    """
    if pd.isna(texto_str):
        return "NAO_INFORMADO"
    return re.sub(r'\s+', ' ', str(texto_str).strip()).upper()

#PIPELINE PRINCIPAL
def preprocessar_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Recebe o DataFrame bruto (stg_ceap) e retorna três artefatos:

    Returnos:
    df_fato    : pd.DataFrame
        Registros limpos e válidos para análise (valor_limpo > 0).
    df_estornos : pd.DataFrame
        Registros com valor_limpo <= 0 isolados para auditoria separada.
    relatorio  : dict
        Métricas de qualidade de dados para o log de pré-processamento.
    """
    df = df_raw.copy()

    # Etapa 0: Normalização de nomes de colunas
    # Remove acentos, espaços e converte para snake_case minúsculo.
    # O CSV do Senado usa nomes com acento (ex.: 'MÊS').
    col_map = {
        col: col.strip()
                .lower()
                .replace(' ', '_')
                .replace('ê', 'e')
                .replace('ç', 'c')
                .replace('ã', 'a')
                .replace('á', 'a')
                .replace('é', 'e')
                .replace('í', 'i')
                .replace('ó', 'o')
                .replace('ú', 'u')
        for col in df.columns
    }
    df = df.rename(columns=col_map)

    total_bruto = len(df)

    # Etapa 1: Tipagem de campos numéricos
    # Int64 (nullable) permite NaN sem coerção para float.
    df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
    df['mes'] = pd.to_numeric(df['mes'], errors='coerce').astype('Int64')
    df['cod_senador'] = pd.to_numeric(df['cod_senador'], errors='coerce').astype('Int64')

    # Etapa 2: Conversão monetária
    df['valor_limpo'] = df['valor_reembolsado'].apply(limpar_moeda)

    # Etapa 3: Sanitização de documentos fiscais
    df['documento_fornecedor'] = df['cpf_cnpj_fornecedor'].apply(limpar_documento)

    # Etapa 4: Conversão de datas
    # errors='coerce' transforma datas sintaticamente inválidas em NaT (rastreável).
    df['data_despesa'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')

    """
    Etapa 4b: Anulação de anos implausíveis em datas tecnicamente válidas

    Existem datas que passam no parsing, ou seja, são strings sintaticamente corretas, mas possuem data corrompida da fonte.
    (ex: "0202-05-04" em vez de "2020-05-04". Dígitos trocados/truncados no sistema do Senado, visíveis já no CSV bruto)

    Para isso, foi aplicada uma tolerância de ±1 ano entre a data da despesa e o ano reportado para evitar falsos positivos decorrentes de lançamentos contábeis entre anos consecutivos.
    Desvios fora dessa janela tendem a indicar erros de digitação ou datas potencialmente corrompidas.

    Esses registros são capturados para o relatório de qualidade (auditoria)
    e então anulados (NaT) — o mesmo tratamento dado às datas sintaticamente
    malformadas — para não propagar 'data_despesa'/'periodo_ano_mes' corrompidos
    (ex.: "0202-10") para a tabela fato e para o feature engineering.
    """
    ano_min, ano_max = int(df['ano'].min()), int(df['ano'].max())
    mask_ano_implausivel = (
        df['data_despesa'].notna()
        & ~df['data_despesa'].dt.year.between(ano_min - 1, ano_max + 1)
    )
    n_anos_implausiveis = int(mask_ano_implausivel.sum())
    exemplos_anos_implausiveis = (
        df.loc[mask_ano_implausivel,
               ['ano', 'mes', 'cod_senador', 'nome_senador', 'data', 'data_despesa']]
          .astype(str)
          .to_dict(orient='records')
    )

    df.loc[mask_ano_implausivel, 'data_despesa'] = pd.NaT
    n_nat = int(df['data_despesa'].isna().sum())

    # Etapa 5: Padronização textual de campos categóricos
    colunas_texto = {
        'nome_senador'   : 'nome_senador_limpo',
        'tipo_despesa'   : 'tipo_despesa_limpo',
        'nome_fornecedor': 'nome_fornecedor_limpo',
        'tipo_documento' : 'tipo_documento_limpo',
        'detalhamento'   : 'detalhamento_limpo',
    }
    for col_orig, col_dest in colunas_texto.items():
        if col_orig in df.columns:
            df[col_dest] = df[col_orig].apply(padronizar_string)

    # Etapa 6: Separação de estornos (valor_limpo <= 0)
    mask_estorno = df['valor_limpo'] <= 0
    df_estornos = df[mask_estorno].copy()
    df_fato = df[~mask_estorno].copy()

    # Etapa 7: Relatório de qualidade de dados
    relatorio = {
        "timestamp"              : datetime.now().isoformat(),
        "total_registros_brutos" : total_bruto,
        "registros_validos"      : len(df_fato),
        "registros_estorno"      : len(df_estornos),
        "datas_invalidas_nat"    : int(n_nat),
        "anos_implausiveis": {
            "intervalo_observado_ano" : [ano_min, ano_max],
            "intervalo_tolerado_ano"  : [ano_min - 1, ano_max + 1],
            "justificativa_tolerancia": (
                "±1 ano em torno do intervalo de 'ano' absorve referências "
                "cruzadas legítimas entre dezembro/janeiro (nota fiscal de "
                "um ano, despesa reportada no ano seguinte ou anterior)."
            ),
            "total_registros": n_anos_implausiveis,
            "exemplos"       : exemplos_anos_implausiveis,
        },
        "percentual_estornos"    : round(len(df_estornos) / total_bruto * 100, 4),
        "nulos_por_coluna": {
            col: int(df_fato[col].isna().sum())
            for col in df_fato.columns
            if df_fato[col].isna().sum() > 0
        },
        "colunas_geradas": list(colunas_texto.values()) + [
            'valor_limpo', 'documento_fornecedor', 'data_despesa',
            'ano', 'mes', 'cod_senador'
        ],
        "decisoes_filtragem": [
            "Registros com valor_limpo <= 0 isolados em tabela 'estornos_ceap' "
            "(não descartados — preservados para auditoria).",
            "Datas malformadas convertidas para NaT via errors='coerce' "
            "(não removidas da tabela fato).",
            "Datas com ano fora do intervalo observado em 'ano' (provável "
            "corrupção de dígitos na fonte; ex.: '0202-05-04' em vez de "
            "'2020-05-04') são registradas em 'anos_implausiveis' para "
            "auditoria e então anuladas para NaT — mesmo tratamento dado "
            "às datas sintaticamente malformadas — para não propagar "
            "'data_despesa'/'periodo_ano_mes' corrompidos para a tabela "
            "fato e o feature engineering.",
        ],
    }

    return df_fato, df_estornos, relatorio

# Seleção de colunas para a tabela fato
COLUNAS_FATO = [
    'ano',
    'mes',
    'cod_senador',
    'nome_senador_limpo',
    'tipo_despesa_limpo',
    'nome_fornecedor_limpo',
    'documento_fornecedor',
    'data_despesa',
    'valor_limpo',
    'tipo_documento_limpo',
    'detalhamento_limpo',
    'source_file',
]

COLUNAS_ESTORNO = COLUNAS_FATO  #mesma estrutura para facilitar auditoria

def salvar_relatorio_preprocessamento(relatorio: dict, logs_dir: Path) -> Path:
    """Persiste o relatório de qualidade de dados em JSON estruturado."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    caminho = logs_dir / "preprocessamento_qualidade.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    return caminho
