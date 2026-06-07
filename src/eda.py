"""
Uma função de tabela + uma função de gráfico por análise.

    RQ1  → rq1_categorias()          + plot_rq1()
    RQ2  → rq2_senadores()           + plot_rq2()
           rq2_top_fornecedores()    + plot_rq2_fornecedores()
    RQ3  → rq3_anomalias()           + plot_rq3()
    RQ4  → rq4_serie_temporal()      + plot_rq4()
           rq4_pandemia()            + plot_rq4_pandemia()
           rq4_sazonalidade()        + plot_rq4_sazonalidade()
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

#Paleta daltônica-segura
AZUL     = '#0072B2'
LARANJA  = '#E69F00'
VERDE    = '#009E73'
VERMELHO = '#D55E00'
CEU      = '#56B4E9'
ROSA     = '#CC79A7'
PALETA   = [AZUL, LARANJA, VERDE, VERMELHO, CEU, ROSA, '#F0E442', '#000000']

plt.rcParams.update({
    'figure.dpi'        : 150,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : '#F9F9F9',
    'axes.grid'         : True,
    'grid.alpha'        : 0.35,
    'grid.linestyle'    : '--',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 10,
    'legend.fontsize'   : 9,
})

_BRL  = mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}')
_MILH = mticker.FuncFormatter(lambda x, _: f'R${x:,.1f}M')

def _salvar(fig, nome, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    p = figures_dir / f'{nome}.png'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    return p

# RQ1 — Categorias de despesa
def rq1_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela com uma linha por categoria: total, %, rank e % acumulado (Pareto).
    Ordenada do maior para o menor gasto.
    """
    total_geral = df['valor_limpo'].sum()
    resumo = (
        df.groupby('tipo_despesa_limpo')['valor_limpo']
          .agg(n_registros='count', total='sum', media='mean', mediana='median')
          .reset_index()
          .rename(columns={'tipo_despesa_limpo': 'categoria'})
    )
    resumo = resumo.sort_values('total', ascending=False).reset_index(drop=True)
    resumo['rank']          = resumo.index + 1
    resumo['percentual']    = (resumo['total'] / total_geral * 100).round(2)
    resumo['pct_acumulado'] = resumo['percentual'].cumsum().round(2)
    return resumo

def plot_rq1(df: pd.DataFrame, figures_dir: Path) -> tuple:
    """
    Figura RQ1 — Volume total por categoria + curva de Pareto.

    Painel esquerdo: barras horizontais com percentual anotado.
    Painel direito:  curva de Pareto — responde "N categorias = 80% do gasto".
    """
    resumo = rq1_categorias(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Painel esquerdo — barras horizontais
    cores = [VERMELHO if r <= 3 else AZUL for r in resumo['rank']]
    bars = ax1.barh(
        resumo['categoria'].str[:50][::-1],
        resumo['total'][::-1] / 1e6,
        color=cores[::-1], edgecolor='white', linewidth=0.6, height=0.65
    )
    for bar, pct in zip(bars, resumo['percentual'][::-1]):
        ax1.text(bar.get_width() + 0.2,
                 bar.get_y() + bar.get_height() / 2,
                 f'{pct:.1f}%', va='center', fontsize=8.5, color='#333')
    ax1.set_xlabel('Volume Total (R$ milhões)')
    ax1.set_title('Volume Total por Categoria\n(Top 3 em vermelho)')
    ax1.xaxis.set_major_formatter(_MILH)

    # Painel direito — curva de Pareto
    x   = np.arange(len(resumo))
    ax2b = ax2.twinx()
    ax2.bar(x, resumo['total'] / 1e6,
            color=AZUL, alpha=0.75, edgecolor='white', width=0.7)
    ax2b.plot(x, resumo['pct_acumulado'], 'o-',
              color=VERMELHO, lw=2, markersize=6, label='% acumulado')
    ax2b.axhline(80, color=LARANJA, ls='--', lw=1.5, label='80%')
    cruzamento = resumo[resumo['pct_acumulado'] >= 80].index[0]
    ax2b.axvline(cruzamento, color=LARANJA, ls=':', lw=1.5, alpha=0.7)
    ax2b.text(cruzamento + 0.15, 82,
              f'{cruzamento+1} cat.\n= 80%', fontsize=8, color=VERMELHO)
    ax2.set_xticks(x)
    ax2.set_xticklabels(resumo['categoria'].str[:30],
                        rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Volume (R$ milhões)')
    ax2.yaxis.set_major_formatter(_MILH)
    ax2b.set_ylabel('% Acumulado')
    ax2b.set_ylim(0, 105)
    ax2b.legend(loc='center right', fontsize=8)
    ax2.set_title('Curva de Pareto\n(concentração do gasto)')

    fig.suptitle(
        'RQ1 — Concentração de Gastos por Categoria\nCEAP 56ª Legislatura (2019–2023)',
        fontsize=12)
    fig.tight_layout()
    _salvar(fig, 'rq1_categorias', figures_dir)

    stats = {
        'n_categorias_80pct': int(cruzamento + 1),
        'top3_pct'          : float(resumo.head(3)['percentual'].sum()),
        'tabela'            : resumo.to_dict('records'),
    }
    return fig, stats

# RQ2 — Perfis de senadores
def rq2_senadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela com uma linha por senador: gasto total e features de perfil.
    Remove proporcao_fim_mes e proporcao_fds — não fazem parte da análise.
    """
    perfil = (
        df.groupby('cod_senador')
          .agg(nome=('nome_senador_limpo', 'first'),
               total=('valor_limpo', 'sum'),
               n_registros=('valor_limpo', 'count'))
          .reset_index()
    )
    # Features de perfil relevantes (sem burn-rate e sem FDS)
    features = [
        'gasto_total_senador',
        'gasto_medio_mensal_senador',
        'n_fornecedores_distintos',
        'n_categorias_distintas',
        'proporcao_maior_categoria',
        'proporcao_maior_fornecedor',
    ]
    features_agg = (
        df.groupby('cod_senador')[features]
          .first()
          .reset_index()
    )
    perfil = perfil.merge(features_agg, on='cod_senador', how='left')
    perfil = perfil.sort_values('total', ascending=False).reset_index(drop=True)
    perfil['rank'] = perfil.index + 1
    return perfil

def plot_rq2(df: pd.DataFrame, figures_dir: Path, top_n: int = 20) -> tuple:
    """
    Figura RQ2 — Ranking de senadores + scatter de perfil.

    Painel esquerdo: top_n senadores por gasto total.
    Painel direito:  gasto total × n_fornecedores, colorido por
                     proporcao_maior_fornecedor (azul=distribuído, vermelho=concentrado).
    """
    perfil = rq2_senadores(df)
    top    = perfil.head(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Painel esquerdo — ranking
    cores = [VERMELHO if r <= 5 else AZUL for r in top['rank']]
    ax1.barh(top['nome'].str[:30][::-1],
             top['total'][::-1] / 1e3,
             color=cores[::-1], edgecolor='white', linewidth=0.5, height=0.65)
    ax1.set_xlabel('Gasto Total (R$ mil)')
    ax1.set_title(f'Top {top_n} Senadores por Gasto Total\n(Top 5 em vermelho)')
    ax1.xaxis.set_major_formatter(_BRL)

    # Painel direito — scatter perfil
    sc = ax2.scatter(
        perfil['n_fornecedores_distintos'],
        perfil['total'] / 1e3,
        c=perfil['proporcao_maior_fornecedor'],
        cmap='RdYlBu_r', s=60, alpha=0.75,
        edgecolors='#666', linewidths=0.4,
        vmin=0, vmax=1,
    )
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Concentração no\nmaior fornecedor [0–1]', fontsize=8)
    for _, row in perfil.head(5).iterrows():
        ax2.annotate(row['nome'].split()[-1],
                     (row['n_fornecedores_distintos'], row['total'] / 1e3),
                     xytext=(5, 3), textcoords='offset points',
                     fontsize=7.5, color=VERMELHO)
    ax2.set_xlabel('Nº de Fornecedores Distintos')
    ax2.set_ylabel('Gasto Total (R$ mil)')
    ax2.yaxis.set_major_formatter(_BRL)
    ax2.set_title('Gasto Total × Diversidade de Fornecedores\n'
                  '(Cor = concentração no maior CNPJ)')

    fig.suptitle('RQ2 — Perfis de Gasto por Senador\nCEAP 56ª Legislatura (2019–2023)',
                 fontsize=12)
    fig.tight_layout()
    _salvar(fig, 'rq2_senadores', figures_dir)

    stats = {
        'n_senadores'        : int(len(perfil)),
        'media_por_senador'  : float(perfil['total'].mean()),
        'mediana_por_senador': float(perfil['total'].median()),
        'top5_pct_total'     : float(perfil.head(5)['total'].sum()
                                     / perfil['total'].sum() * 100),
    }
    return fig, stats

def rq2_top_fornecedores(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Tabela dos N fornecedores que mais receberam recursos na legislatura.

    A coluna 'n_senadores' revela o padrão de uso:
    - Muitos senadores, alto valor: fornecedor amplamente utilizado (comportamento normal)
    - Poucos senadores, alto valor: fornecedor concentrado (sinal de atenção para RQ3)
    """
    df_cnpj = df[df['documento_fornecedor'] != 'NAO_INFORMADO'].copy()
    top = (
        df_cnpj.groupby(['documento_fornecedor', 'nome_fornecedor_limpo'])
               .agg(total=('valor_limpo', 'sum'),
                    n_registros=('valor_limpo', 'count'),
                    n_senadores=('cod_senador', 'nunique'))
               .reset_index()
               .sort_values('total', ascending=False)
               .head(n)
               .reset_index(drop=True)
    )
    top['rank'] = top.index + 1
    top_geral   = df_cnpj['valor_limpo'].sum()
    top['percentual'] = (top['total'] / top_geral * 100).round(2)
    return top

def plot_rq2_fornecedores(df: pd.DataFrame, figures_dir: Path, n: int = 15) -> tuple:
    """
    Figura RQ2-B — Top fornecedores por volume recebido.

    Barras horizontais. Cor diferenciada quando n_senadores <= 3,
    sinalizando concentração de recebimento em poucos parlamentares.
    """
    top = rq2_top_fornecedores(df, n)
    fig, ax = plt.subplots(figsize=(11, 7))

    cores = [VERMELHO if row['n_senadores'] <= 3 else AZUL
             for _, row in top.iterrows()]
    bars = ax.barh(
        top['nome_fornecedor_limpo'].str[:45][::-1],
        top['total'][::-1] / 1e6,
        color=cores[::-1], edgecolor='white', linewidth=0.5, height=0.65
    )
    for bar, (_, row) in zip(bars, top[::-1].iterrows()):
        ax.text(bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{row['percentual']:.1f}% | {row['n_senadores']} sen.",
                va='center', fontsize=8, color='#333')

    ax.set_xlabel('Total Recebido (R$ milhões)')
    ax.set_title(
        f'Top {n} Fornecedores por Volume Recebido\n'
        f'Vermelho = recebido de ≤ 3 senadores distintos'
    )
    ax.xaxis.set_major_formatter(_MILH)

    fig.suptitle('RQ2 — Concentração de Pagamentos por Fornecedor\n'
                 'CEAP 56ª Legislatura (2019–2023)', fontsize=12)
    fig.tight_layout()
    _salvar(fig, 'rq2_top_fornecedores', figures_dir)

    stats = top.to_dict('records')
    return fig, stats

# RQ3 — Anomalias e desvios
def rq3_anomalias(df: pd.DataFrame) -> dict:
    """
    Resumo das anomalias: contagem por método (IQR e consenso Z+IQR).
    """
    return {
        'n_total'            : int(len(df)),
        'n_outlier_iqr'      : int(df['iqr_outlier_flag'].sum()),
        'pct_outlier_iqr'    : round(float(df['iqr_outlier_flag'].mean() * 100), 2),
        'n_anomalia_consenso': int(df['anomalia_consenso'].sum()),
        'pct_consenso'       : round(float(df['anomalia_consenso'].mean() * 100), 2),
    }

def plot_rq3(df: pd.DataFrame, figures_dir: Path) -> tuple:
    """
    Figura RQ3 — Distribuição de valores + heatmap de anomalias.

    Painel esquerdo: histograma original (azul) sobreposto ao log (laranja).
                     Mostra por que a transformação logarítmica é necessária
                     antes do Z-score: a escala original é extremamente assimétrica.
    Painel direito:  heatmap de anomalias de consenso por categoria e ano.
                     Concentração em poucos anos = evento pontual.
                     Distribuição uniforme ao longo dos anos = padrão sistemático.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Painel esquerdo — histograma duplo
    ax1b    = ax1.twinx()
    valores = df['valor_limpo']
    v_log   = df['valor_log1p']
    p99     = valores.quantile(0.99)

    ax1.hist(valores[valores <= p99], bins=70,
             color=AZUL, alpha=0.6, edgecolor='white', lw=0.3,
             label='Escala original')
    ax1b.hist(v_log, bins=70,
              color=LARANJA, alpha=0.5, edgecolor='white', lw=0.3,
              label='log(1+valor)')
    ax1.axvline(valores.median(), color=AZUL, lw=2, ls='--',
                label=f'Mediana: R${valores.median():,.0f}')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    ax1.set_xlabel('Valor (R$) / log(1+R$)')
    ax1.set_ylabel('Frequência — escala original', color=AZUL)
    ax1b.set_ylabel('Frequência — log', color=LARANJA)
    ax1.xaxis.set_major_formatter(_BRL)
    ax1.set_title(f'Distribuição dos Valores\n'
                  f'(assimetria: {valores.skew():.1f} → log: {v_log.skew():.2f})')

    # Painel direito — heatmap anomalias de consenso
    hm = (
        df.groupby(['tipo_despesa_limpo', 'ano'])['anomalia_consenso']
          .sum().unstack(fill_value=0).astype(int)
    )
    hm = hm.loc[hm.sum(axis=1).sort_values(ascending=False).index]
    sns.heatmap(hm, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
                linewidths=0.5,
                cbar_kws={'label': 'Nº Anomalias\n(Z-score E IQR)'},
                annot_kws={'size': 9})
    ax2.set_yticklabels([c[:40] for c in hm.index], fontsize=8, rotation=0)
    ax2.set_xlabel('Ano')
    ax2.set_title('Anomalias de Consenso\npor Categoria e Ano')
    ax2.tick_params(axis='x', rotation=0)

    fig.suptitle('RQ3 — Distribuição de Valores e Desvios Significativos\n'
                 'CEAP 56ª Legislatura', fontsize=12)
    fig.tight_layout()
    _salvar(fig, 'rq3_anomalias', figures_dir)

    return fig, rq3_anomalias(df)

# RQ4 — Evolução temporal
def rq4_serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Série anual: total, n registros, ticket médio, mediana e variação YoY.
    """
    serie = (
        df.groupby('ano')['valor_limpo']
          .agg(total='sum', n='count', media='mean', mediana='median')
          .reset_index()
    )
    serie['yoy_pct'] = (serie['total'].pct_change() * 100).round(2)
    return serie

def plot_rq4(df: pd.DataFrame, figures_dir: Path) -> tuple:
    """
    Figura RQ4 — Série mensal com eventos + barras anuais com YoY.

    Painel superior: linha do gasto total mensal.
                     Faixa vermelha = pandemia (mar/2020–abr/2022).
                     Linha tracejada laranja = início de 2022 (ano eleitoral).
    Painel inferior: barras anuais com total em R$ e variação YoY dentro da barra.
    """
    serie_mensal = (
        df.groupby('periodo_ano_mes')['valor_limpo']
          .sum().reset_index()
          .sort_values('periodo_ano_mes').reset_index(drop=True)
    )
    meses      = serie_mensal['periodo_ano_mes'].tolist()
    x          = np.arange(len(serie_mensal))
    serie_anual = rq4_serie_temporal(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Painel superior — série mensal
    ax1.plot(x, serie_mensal['valor_limpo'] / 1e6, color=AZUL, lw=2)
    try:
        i_ini = next(i for i, m in enumerate(meses) if m >= '2020-03')
        i_fim = next(i for i, m in enumerate(meses) if m >  '2022-04')
        ax1.axvspan(i_ini, i_fim, alpha=0.10, color=VERMELHO, label='Pandemia')
    except StopIteration:
        pass
    try:
        i_ele = next(i for i, m in enumerate(meses) if m >= '2022-01')
        ax1.axvline(i_ele, color=LARANJA, lw=1.2, ls='--',
                    alpha=0.8, label='Início 2022 (eleições)')
    except StopIteration:
        pass
    ticks = list(range(0, len(x), 6))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([meses[i] for i in ticks], rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('Total Mensal (R$ milhões)')
    ax1.yaxis.set_major_formatter(_MILH)
    ax1.set_title('Gasto Total Mensal — 56ª Legislatura')
    ax1.legend(fontsize=8)

    # Painel inferior — barras anuais
    anos = serie_anual['ano'].astype(str).tolist()
    xa   = np.arange(len(anos))
    bars = ax2.bar(xa, serie_anual['total'] / 1e6,
                   color=PALETA[:len(anos)], edgecolor='white', width=0.65)
    for bar, (_, row) in zip(bars, serie_anual.iterrows()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'R${row["total"]/1e6:.1f}M',
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        if pd.notna(row['yoy_pct']):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() / 2,
                     f'{row["yoy_pct"]:+.1f}%',
                     ha='center', va='center', fontsize=8,
                     color='white', fontweight='bold')
    ax2.set_xticks(xa)
    ax2.set_xticklabels(anos, fontsize=9)
    ax2.set_ylabel('Total Anual (R$ milhões)')
    ax2.yaxis.set_major_formatter(_MILH)
    ax2.set_title('Gasto Total por Ano (variação YoY dentro das barras)')

    fig.suptitle('RQ4 — Evolução dos Gastos CEAP ao Longo da 56ª Legislatura', fontsize=12)
    fig.tight_layout()
    _salvar(fig, 'rq4_serie_temporal', figures_dir)

    return fig, rq4_serie_temporal(df).to_dict('records')

def rq4_pandemia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela comparativa entre os três períodos de pandemia.
    Uma linha por período: total gasto, n registros, ticket médio e mediana.
    """
    ordem = ['pre_pandemia', 'pandemia', 'pos_pandemia']
    
    df_p = df[df['periodo_pandemia'].isin(ordem)].copy()

    tabela = (
        df_p.groupby('periodo_pandemia')['valor_limpo']
            .agg(total='sum', n_registros='count',
                 media='mean', mediana='median')
            .reset_index()
    )
    
    tabela['periodo_pandemia'] = pd.Categorical(
        tabela['periodo_pandemia'], categories=ordem, ordered=True
    )
    tabela = tabela.sort_values('periodo_pandemia').reset_index(drop=True)

    total_geral = tabela['total'].sum()
    tabela['percentual'] = (tabela['total'] / total_geral * 100).round(2)
    return tabela

def plot_rq4_pandemia(df: pd.DataFrame, figures_dir: Path) -> tuple:
    """
    Figura RQ4-B — Impacto da pandemia nos gastos CEAP.
    """
    ordem  = ['pre_pandemia', 'pandemia', 'pos_pandemia']
    df_p = df[df['periodo_pandemia'].isin(ordem)].copy()

    labels = ['Pré-pandemia\n(até fev/2020)',
              'Pandemia\n(mar/2020–abr/2022)',
              'Pós-pandemia\n(a partir mai/2022)']
    cores_p = [VERDE, VERMELHO, AZUL]

    dados_box = [
        df_p.loc[df_p['periodo_pandemia'] == p, 'valor_limpo'].dropna()
        for p in ordem
    ]
    tabela = rq4_pandemia(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Painel esquerdo — boxplot
    bp = ax1.boxplot(
        dados_box, labels=labels, patch_artist=True, notch=False,
        medianprops=dict(color='black', lw=2.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.2, color='grey'),
    )
    for patch, cor in zip(bp['boxes'], cores_p):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)
    for i, dados in enumerate(dados_box):
        if not dados.empty:
            med = dados.median()
            ax1.text(i + 1, med,
                     f'R${med:,.0f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Valor da Despesa (R$)')
    ax1.yaxis.set_major_formatter(_BRL)
    ax1.set_title('Distribuição do Ticket por Período\n(linha = mediana)')

    # Painel direito — barras com total e %
    bars = ax2.bar(
        labels,
        tabela['total'] / 1e6,
        color=cores_p, edgecolor='white', linewidth=0.5
    )
    for bar, (_, row) in zip(bars, tabela.iterrows()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'R${row["total"]/1e6:.1f}M\n({row["percentual"]:.1f}%)',
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('Total Gasto (R$ milhões)')
    ax2.yaxis.set_major_formatter(_MILH)
    ax2.set_title('Total Gasto por Período')

    fig.suptitle(
        'RQ4 — Impacto da Pandemia nos Gastos CEAP\n'
        'Comparação Pré / Durante / Pós-Pandemia',
        fontsize=12
    )
    fig.tight_layout()
    _salvar(fig, 'rq4_pandemia', figures_dir)

    return fig, tabela.to_dict('records')

def rq4_sazonalidade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela pivot com gasto médio por (mês, ano): linhas = meses (1–12), colunas = anos.
    Revela padrões que se repetem todo ano (ex: pico em dezembro, queda em janeiro).
    """
    pivot = (
        df.groupby(['ano', 'mes'])['valor_limpo']
          .sum()
          .unstack(level='ano')
          .fillna(0)
    )
    pivot.index.name = 'mes'
    return pivot

def plot_rq4_sazonalidade(df: pd.DataFrame, figures_dir: Path) -> tuple:
    """
    Figura RQ4-C — Heatmap de sazonalidade: gasto por (mês, ano).

    Cada célula = total gasto naquele mês/ano em R$ milhões.
    Padrões que se repetem em todas as colunas = sazonalidade real.
    Célula isoladamente escura = evento pontual naquele mês específico.
    """
    pivot = rq4_sazonalidade(df)

    nomes_mes = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                 7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
    pivot.index = [nomes_mes.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot / 1e6,
        annot=True, fmt='.1f', cmap='Blues',
        ax=ax, linewidths=0.5,
        cbar_kws={'label': 'Total Gasto (R$ milhões)'},
        annot_kws={'size': 9},
    )
    ax.set_xlabel('Ano')
    ax.set_ylabel('Mês')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title(
        'RQ4 — Sazonalidade dos Gastos CEAP\n'
        'Total por Mês e Ano (R$ milhões)'
    )

    fig.tight_layout()
    _salvar(fig, 'rq4_sazonalidade', figures_dir)

    return fig, pivot.to_dict()