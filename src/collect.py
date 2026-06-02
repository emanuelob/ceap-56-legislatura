"""
DOCUMENTAÇÃO:
    Fonte: Portal de Dados Abertos do Senado Federal
    URL base: https://adm.senado.gov.br/adm-dadosabertos
    Endpoint: /api/v1/senadores/despesas_ceaps/{ano}/csv
    Acesso: público, sem autenticação
    Período: 56ª Legislatura completa (2019–2023)
    Formato: CSV com separador ponto-e-vírgula (;), encoding UTF-8

JUSTIFICATIVA DO PERÍODO (ACM SIGSOFT — 'explains how and why data was selected'):
    A 56ª Legislatura (2019–2023) representa um ciclo legislativo encerrado,
    permitindo análise integral e evitando viés de período incompleto.

JUSTIFICATIVA DA ROTA CSV (vs JSON):
    O endpoint /csv retorna dados tabulares diretamente, eliminando etapa
    de serialização/desserialização JSON. Mais eficiente para grandes volumes
    e carregamento direto com pandas.read_csv(). Os campos são idênticos
    em ambas as rotas; a escolha é puramente operacional.
"""

import requests
import time
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

BASE_URL = "https://adm.senado.gov.br/adm-dadosabertos"
ENDPOINT = "/api/v1/senadores/despesas_ceaps/{ano}/csv"

ANOS_56_LEGISLATURA = [2019, 2020, 2021, 2022, 2023]

ROOT_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT_DIR / "data" / "raw"
LOGS_DIR    = ROOT_DIR / "logs"

PAUSA_ENTRE_REQUISICOES = 1.5   # segundos 
TIMEOUT_REQUISICAO      = 90    # segundos
MAX_TENTATIVAS          = 3     # retentativas em caso de falha transiente (obs.: API estável, rede não falhou)

def _sha256(filepath: Path) -> str:
    """Calcula SHA-256 do arquivo para garantia de integridade."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()

def _contar_linhas_csv(filepath: Path) -> int:
    """Conta linhas do CSV (incluindo cabeçalho) para verificação de sanidade."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)

def download_ano(ano: int, session: requests.Session) -> dict:
    """
    Baixa o CSV de despesas CEAP de um ano e salva em data/raw/.

    Retorna um dicionário de metadados para o log de coleta.
    Implementa retentativas com backoff exponencial para robustez.

    Parâmetros:
    ano     : int — ano da 56ª Legislatura (2019–2023)
    session : requests.Session — sessão HTTP reutilizável

    Retorno:
    dict com: ano, url, data_coleta, status_code, tamanho_bytes,
              total_linhas, sha256, caminho_arquivo, erro (se houver)
    """
    url      = BASE_URL + ENDPOINT.format(ano=ano)
    filepath = RAW_DIR / f"ceap_{ano}.csv"
    ts_inicio = datetime.now()

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_REQUISICAO)
            response.raise_for_status()

            # Salva exatamente como recebido — nunca modificar dados brutos
            with open(filepath, "wb") as f:
                f.write(response.content)

            total_linhas = _contar_linhas_csv(filepath)
            checksum     = _sha256(filepath)
            tamanho_kb   = len(response.content) / 1024

            print(f"  ✓ {tamanho_kb:.1f} KB  |  {total_linhas - 1:,} registros  |  {filepath.name}")

            return {
                "ano"            : ano,
                "url"            : url,
                "data_coleta"    : ts_inicio.isoformat(),
                "status_http"    : response.status_code,
                "tamanho_bytes"  : len(response.content),
                "total_linhas"   : total_linhas - 1,   # exclui cabeçalho
                "sha256"         : checksum,
                "caminho_arquivo": str(filepath),
                "erro"           : None,
            }

        except requests.exceptions.HTTPError as e:
            print(f"  ✗ Tentativa {tentativa}/{MAX_TENTATIVAS} — HTTP {e.response.status_code}: {e}")
            if e.response.status_code in (400, 404):
                break   # erro permanente — não retenta
        except requests.exceptions.Timeout:
            print(f"  ✗ Tentativa {tentativa}/{MAX_TENTATIVAS} — Timeout após {TIMEOUT_REQUISICAO}s")
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Tentativa {tentativa}/{MAX_TENTATIVAS} — {type(e).__name__}: {e}")

        if tentativa < MAX_TENTATIVAS:
            espera = 2 ** tentativa   # backoff: 2s, 4s
            print(f"  ↻ Aguardando {espera}s antes de retentar...")
            time.sleep(espera)

    return {
        "ano"         : ano,
        "url"         : url,
        "data_coleta" : ts_inicio.isoformat(),
        "erro"        : "Falha após todas as tentativas",
    }

def salvar_log_coleta(metadados: list[dict]) -> Path:
    """
    Persiste o log de coleta em JSON estruturado.

    Padrão ACM SIGSOFT: 'presents the experimental setup' e 'explains how the data was selected'.

    Parâmetros
    metadados : lista de dicts retornados por download_ano()

    Retorno
    Path do arquivo de log gerado
    """
    log = {
        "projeto": "Análise de Padrões e Anomalias na CEAP — 56ª Legislatura",
        "autores": ["Emanuel de Oliveira", "Leandro Kornelius"],
        "fonte_dados": {
            "nome"       : "Portal de Dados Abertos do Senado Federal",
            "url_base"   : BASE_URL,
            "endpoint"   : ENDPOINT,
            "acesso"     : "público, sem autenticação",
            "formato"    : "CSV ; separador ponto-e-vírgula",
            "encoding"   : "UTF-8",
        },
        "escopo_temporal": {
            "legislatura"   : "56ª (2019–2023)",
            "anos_coletados": ANOS_56_LEGISLATURA,
            "justificativa" : (
                "Período encerrado e completo. Evita viés de truncamento "
                "presente em legislaturas em curso. Cobre eventos de alto "
                "impacto sobre gastos parlamentares (pandemia COVID-19 em "
                "2020–2021, eleições presidenciais em 2022)."
            ),
        },
        "criterios_exclusao": [
            "57ª Legislatura (2023–2027): período em curso, dados incompletos.",
            "Legislaturas anteriores à 56ª: fora do escopo temporal definido.",
        ],
        "data_execucao"   : datetime.now().isoformat(),
        "parametros_coleta": {
            "timeout_segundos"   : TIMEOUT_REQUISICAO,
            "max_tentativas"     : MAX_TENTATIVAS,
            "pausa_entre_anos_s" : PAUSA_ENTRE_REQUISICOES,
        },
        "resultados": metadados,
        "resumo": {
            "anos_baixados"  : sum(1 for m in metadados if not m.get("erro")),
            "anos_com_erro"  : sum(1 for m in metadados if m.get("erro")),
            "total_registros": sum(
                m.get("total_linhas", 0) for m in metadados if not m.get("erro")
            ),
        },
    }

    log_path = LOGS_DIR / "coleta_metadata.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return log_path

def main():
    # Garante que os diretórios necessários existem
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("-" * 30)
    print("COLETA DE DADOS — CEAP SENADO FEDERAL")
    print(f"56ª Legislatura | Anos: {ANOS_56_LEGISLATURA}")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 30)

    # Session reutilizável com headers de identificação
    session = requests.Session()
    session.headers.update({
        "User-Agent" : "pesquisa-ceap/1.0",
        "Accept"     : "text/csv",
        "Accept-Encoding": "gzip, deflate",
    })

    metadados = []

    for i, ano in enumerate(ANOS_56_LEGISLATURA):
        resultado = download_ano(ano, session)
        metadados.append(resultado)

        if i < len(ANOS_56_LEGISLATURA) - 1:
            time.sleep(PAUSA_ENTRE_REQUISICOES)

    # Persiste o log de coleta
    log_path = salvar_log_coleta(metadados)

    # Resumo final
    sucessos = sum(1 for m in metadados if not m.get("erro"))
    total_registros = sum(
        m.get("total_linhas", 0) for m in metadados if not m.get("erro")
    )

    print("\n" + "-" * 30)
    print(f"Coleta concluída: {sucessos}/{len(ANOS_56_LEGISLATURA)} anos")
    print(f"Total de registros coletados: {total_registros:,}")
    print(f"Dados brutos: {RAW_DIR}/")
    print(f"Log de coleta: {log_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()