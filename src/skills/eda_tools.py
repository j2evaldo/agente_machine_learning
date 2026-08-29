"""
Módulo de ferramentas analíticas para Auditoria de Dados, Prevenção de Leakage,
Testes Bivariados Formais, WoE/IV e Dashboards Visuais.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def audit_data_integrity(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Audita volumetria, missing values, cardinalidade e variância quase nula.
    """
    n_total = len(df)
    audit_records = []

    for col in df.columns:
        n_null = int(df[col].isnull().sum())
        pct_null = (n_null / n_total) * 100
        n_unique = int(df[col].nunique())
        
        # Concentração do modo
        mode_counts = df[col].value_counts(normalize=True, dropna=False)
        mode_freq = float(mode_counts.iloc[0] * 100) if not mode_counts.empty else 0.0

        status = "OK"
        if target_col and col == target_col:
            status = "TARGET"
        elif pct_null > 40.0:
            status = "CRÍTICO (>40% nulos)"
        elif mode_freq > 95.0:
            status = "ALERTA (Baixa Variância >95%)"
        elif n_unique > 30 and df[col].dtype == 'object':
            status = "ALTA CARDINALIDADE (>30)"

        audit_records.append({
            "Coluna": col,
            "Tipo": str(df[col].dtype),
            "Nulos": n_null,
            "% Nulos": round(pct_null, 2),
            "Cardinais": n_unique,
            "% Modo Conc.": round(mode_freq, 2),
            "Diagnóstico": status
        })

    return pd.DataFrame(audit_records)


def compute_woe_iv(
    df: pd.DataFrame, 
    feature_col: str, 
    target_col: str, 
    q: int = 10
) -> Tuple[pd.DataFrame, float]:
    """
    Calcula Weight of Evidence (WoE) e Information Value (IV) com correção de continuidade.
    """
    df_temp = df[[feature_col, target_col]].dropna().copy()
    
    if pd.api.types.is_numeric_dtype(df_temp[feature_col]):
        df_temp['bin'] = pd.qcut(df_temp[feature_col], q=q, duplicates='drop')
    else:
        df_temp['bin'] = df_temp[feature_col].astype(str)

    grouped = df_temp.groupby('bin', observed=False).agg(
        total=('bin', 'count'),
        events=(target_col, 'sum')
    ).reset_index()

    grouped['non_events'] = grouped['total'] - grouped['events']
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    if total_events == 0 or total_non_events == 0:
        return pd.DataFrame(), 0.0

    # Correção de continuidade para evitar divisão por zero / log(0)
    eps = 0.5
    grouped['dist_events'] = (grouped['events'] + eps) / (total_events + eps)
    grouped['dist_non_events'] = (grouped['non_events'] + eps) / (total_non_events + eps)

    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv_component'] = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']
    total_iv = float(grouped['iv_component'].sum())

    return grouped[['bin', 'total', 'events', 'non_events', 'woe', 'iv_component']], total_iv


def run_bivariate_screening(
    df: pd.DataFrame,
    target_col: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    is_binary_target: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Executa testes bivariados formais e triagem de poder preditivo para alvos binários ou contínuos.
    """
    results = []
    woe_dict = {}

    if is_binary_target:
        # 1. Alvo Binário: Mann-Whitney, Qui-Quadrado e WoE/IV
        for col in numerical_cols:
            df_valid = df[[col, target_col]].dropna()
            g0 = df_valid[df_valid[target_col] == 0][col]
            g1 = df_valid[df_valid[target_col] == 1][col]
            
            stat, pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            woe_df, iv = compute_woe_iv(df, col, target_col)
            woe_dict[col] = woe_df

            iv_tier = "Descartar (<0.02)" if iv < 0.02 else (
                "Fraco (0.02-0.10)" if iv < 0.1 else (
                    "Médio (0.10-0.30)" if iv < 0.3 else (
                        "Forte (0.30-0.50)" if iv < 0.5 else "SUSPEITO LEAKAGE (>0.50)"
                    )
                )
            )

            results.append({
                "Feature": col,
                "Tipo": "Numérica",
                "Teste Formal": "Mann-Whitney U",
                "Estatística": round(stat, 2),
                "p-valor": f"{pval:.4e}",
                "Poder / Métrica": f"IV = {iv:.4f}",
                "Score_Rank": iv,
                "Recomendação": iv_tier
            })

        for col in categorical_cols:
            df_valid = df[[col, target_col]].dropna()
            contingency = pd.crosstab(df_valid[col], df_valid[target_col])
            
            chi2, pval, _, _ = stats.chi2_contingency(contingency)
            n_obs = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2 / (n_obs * min_dim)) if min_dim > 0 else 0.0

            woe_df, iv = compute_woe_iv(df, col, target_col)
            woe_dict[col] = woe_df

            iv_tier = "Descartar (<0.02)" if iv < 0.02 else (
                "Fraco (0.02-0.10)" if iv < 0.1 else (
                    "Médio (0.10-0.30)" if iv < 0.3 else (
                        "Forte (0.30-0.50)" if iv < 0.5 else "SUSPEITO LEAKAGE (>0.50)"
                    )
                )
            )

            results.append({
                "Feature": col,
                "Tipo": "Categórica",
                "Teste Formal": f"Qui-Quadrado (V={cramer_v:.2f})",
                "Estatística": round(chi2, 2),
                "p-valor": f"{pval:.4e}",
                "Poder / Métrica": f"IV = {iv:.4f}",
                "Score_Rank": iv,
                "Recomendação": iv_tier
            })

    else:
        # 2. Alvo Contínuo (Econometria/OLS): Pearson, Spearman e ANOVA
        for col in numerical_cols:
            df_valid = df[[col, target_col]].dropna()
            r_pearson, p_pearson = stats.pearsonr(df_valid[col], df_valid[target_col])
            r_spearman, _ = stats.spearmanr(df_valid[col], df_valid[target_col])

            results.append({
                "Feature": col,
                "Tipo": "Numérica",
                "Teste Formal": "Pearson / Spearman",
                "Estatística": round(r_pearson, 3),
                "p-valor": f"{p_pearson:.4e}",
                "Poder / Métrica": f"r={r_pearson:.3f} | rho={r_spearman:.3f}",
                "Score_Rank": abs(r_pearson),
                "Recomendação": "Forte Correlação" if abs(r_pearson) > 0.5 else ("Média" if abs(r_pearson) > 0.2 else "Fraca")
            })

        for col in categorical_cols:
            df_valid = df[[col, target_col]].dropna()
            groups = [group[target_col].values for _, group in df_valid.groupby(col)]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
            else:
                f_stat, p_val = 0.0, 1.0

            results.append({
                "Feature": col,
                "Tipo": "Categórica",
                "Teste Formal": "ANOVA One-Way (F)",
                "Estatística": round(f_stat, 2),
                "p-valor": f"{p_val:.4e}",
                "Poder / Métrica": f"F-Stat = {f_stat:.2f}",
                "Score_Rank": f_stat,
                "Recomendação": "Diferença Significante" if p_val < 0.05 else "Sem Diferença Relevante"
            })

    screening_df = pd.DataFrame(results).sort_values(by="Score_Rank", ascending=False)
    return screening_df, woe_dict


def get_leakage_and_discard_features(screening_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Retorna listas separadas de variáveis suspeitas de Leakage e variáveis inúteis para descarte.
    """
    leakage = screening_df[screening_df["Recomendação"].str.contains("LEAKAGE", na=False)]["Feature"].tolist()
    discard = screening_df[screening_df["Recomendação"].str.contains("Descartar", na=False)]["Feature"].tolist()
    return leakage, discard


def plot_eda_dashboard(
    df: pd.DataFrame,
    target_col: str,
    screening_df: pd.DataFrame,
    woe_dict: Optional[Dict[str, pd.DataFrame]] = None,
    top_n: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Gera um dashboard visual 2x2 focado em auditoria estatística executiva.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Ranking de Poder Preditivo
    ax1 = axes[0, 0]
    top_features = screening_df.head(top_n).sort_values(by="Score_Rank")
    colors = ['#d62728' if 'LEAKAGE' in str(r) or 'Descartar' in str(r) else '#2ca02c' for r in top_features['Recomendação']]
    ax1.barh(top_features['Feature'], top_features['Score_Rank'], color=colors, alpha=0.85)
    ax1.set_title("1. Ranking de Preditores por Poder Estatístico", fontweight='bold')
    ax1.set_xlabel("Métrica de Força (IV ou |r|)")
    ax1.grid(True, alpha=0.3)

    # 2. Distribuição do Target
    ax2 = axes[0, 1]
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 2:
        ax2.hist(df[target_col].dropna(), bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
        ax2.set_title(f"2. Distribuição do Target Contínuo ({target_col})", fontweight='bold')
    else:
        counts = df[target_col].value_counts(normalize=True) * 100
        bars = ax2.bar(counts.index.astype(str), counts.values, color=['#1f77b4', '#ff7f0e'], alpha=0.85)
        for b in bars:
            ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 1, f"{b.get_height():.1f}%", ha='center', fontweight='bold')
        ax2.set_title(f"2. Taxa Basal da Resposta ({target_col})", fontweight='bold')
        ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # 3. Monotonicidade de WoE ou Dispersão
    ax3 = axes[1, 0]
    top_col = screening_df.iloc[0]['Feature'] if not screening_df.empty else None
    if woe_dict and top_col in woe_dict and not woe_dict[top_col].empty:
        w_df = woe_dict[top_col]
        ax3.plot(range(len(w_df)), w_df['woe'], marker='o', color='#9467bd', lw=2.5)
        ax3.axhline(0, color='gray', linestyle='--', lw=1)
        ax3.set_xticks(range(len(w_df)))
        ax3.set_xticklabels([str(b) for b in w_df['bin']], rotation=30, ha='right', fontsize=8)
        ax3.set_title(f"3. Monotonicidade do WoE: {top_col}", fontweight='bold')
        ax3.set_ylabel("Weight of Evidence (WoE)")
    elif top_col:
        ax3.scatter(df[top_col], df[target_col], alpha=0.4, color='#9467bd')
        ax3.set_title(f"3. Dispersão: {top_col} vs {target_col}", fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Boxplot de Separação
    ax4 = axes[1, 1]
    if top_col and pd.api.types.is_numeric_dtype(df[top_col]) and df[target_col].nunique() == 2:
        data = [df[df[target_col] == 0][top_col].dropna(), df[df[target_col] == 1][top_col].dropna()]
        ax4.boxplot(data, labels=['Target = 0', 'Target = 1'], patch_artist=True, boxprops=dict(facecolor='#aec7e8'))
        ax4.set_title(f"4. Separação Distribucional: {top_col}", fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "Dashboard Pronto", ha='center', va='center')
    ax4.grid(True, alpha=0.3)

    plt.suptitle("EXECUTIVE DASHBOARD: EDA & STATISTICAL SCREENING", fontsize=13, fontweight='bold', y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()