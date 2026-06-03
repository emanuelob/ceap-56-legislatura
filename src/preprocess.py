import pandas as pd
import re

def limpar_moeda(valor_str) -> float:
    """
    Converte strings monetárias do formato brasileiro para float.
    Assim, evita erros de parsing onde '1.200,50' vira 1.2.
    """
    if pd.isna(valor_str):
        return 0.0
    valor_str = str(valor_str).replace('.', '').replace(',', '.')
    try:
        return float(valor_str)
    except ValueError:
        return 0.0

def limpar_documento(doc_str) -> str:
    """
    Remove pontuações de CNPJ/CPF para padronização de chaves.
    """
    if pd.isna(doc_str):
        return "NAO_INFORMADO"
    return re.sub(r'[^0-9]', '', str(doc_str))

def padronizar_strings(texto_str) -> str:
    """
    Converte para maiúsculas e remove espaços excedentes.
    """
    if pd.isna(texto_str):
        return "NAO_INFORMADO"
    return str(texto_str).strip().upper()

def preprocessar_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame bruto e aplica as transformações de tipagem e limpeza.
    """
    df = df_raw.copy()
    
    # 1. Padronização de nomes de colunas (remove acentos, caixa baixa)
    df.columns = [col.strip().lower().replace(' ', '_').replace('ê', 'e') for col in df.columns]
    
    # 2. Tipagem e limpeza
    df['valor_limpo'] = df['valor_reembolsado'].apply(limpar_moeda)
    df['documento_fornecedor'] = df['cpf_cnpj_fornecedor'].apply(limpar_documento)
    df['data_despesa'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    
    # 3. Padronização Textual
    colunas_texto = ['nome_senador', 'tipo_despesa', 'nome_fornecedor', 'tipo_documento', 'detalhamento']
    for col in colunas_texto:
        df[f'{col}_limpo'] = df[col].apply(padronizar_strings)
            
    return df