"""
Módulo de ferramentas analíticas para Auditoria de Dados, Prevenção de Leakage,
Testes Bivariados Formais, WoE/IV e Dashboards Visuais.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="darkgrid")


def audit_data_integrity(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
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
) -> tuple[pd.DataFrame, float]:
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
    numerical_cols: list[str],
    categorical_cols: list[str],
    is_binary_target: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
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


def get_leakage_and_discard_features(screening_df: pd.DataFrame) -> tuple[list[str], list[str]]:
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
    woe_dict: dict[str, pd.DataFrame] | None = None,
    top_n: int = 8
) -> None:
    """
    Gera um dashboard visual 2x2 focado em auditoria estatística executiva.
    """
    _, axes = plt.subplots(2, 2, figsize=(15, 11))
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
        ax4.boxplot(data, patch_artist=True, boxprops={"facecolor": '#aec7e8'})
        ax4.set_xticklabels(['Target = 0', 'Target = 1'])
        ax4.set_title(f"4. Separação Distribucional: {top_col}", fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "Dashboard Pronto", ha='center', va='center')
    ax4.grid(True, alpha=0.3)

    plt.suptitle("EXECUTIVE DASHBOARD: EDA & STATISTICAL SCREENING", fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def plot_class_imbalance(
    df: pd.DataFrame,
    target_col: str = "Class",
    figsize: tuple[float, float] = (14, 5)
) -> None:
    """
    Exibe o desbalanceamento de classes em barras de contagem (escala log)
    e rosca de proporcao. Renderiza inline, sem exportar arquivo.
    """
    counts = df[target_col].value_counts().sort_index()
    labels = [f"Classe {i}" for i in counts.index]
    palette = ['#1f77b4', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    bars = ax1.bar(labels, counts.values, color=palette, alpha=0.9)
    ax1.set_yscale("log")
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2.0, v * 1.15, f"{int(v):,}".replace(",", "."),
                 ha="center", fontweight="bold", fontsize=9)
    ax1.set_title("Contagem por Classe (Escala Log)", fontweight="bold")
    ax1.set_ylabel("Numero de Transacoes (log)")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    props = (counts / counts.sum() * 100).values
    _wedges, _, autotexts = ax2.pie(
        props, labels=labels, autopct=lambda p: f"{p:.3f}%",
        colors=palette, startangle=90, counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontweight": "bold"},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Proporcao de Classes (Rosca)", fontweight="bold")
    ax2.axis("equal")

    fig.suptitle("DESCOMPENSAMENTO EXTREMO DO ALVO", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_fraud_rate_by_hour(
    df: pd.DataFrame,
    target_col: str = "Class",
    time_col: str = "Time",
    figsize: tuple[float, float] = (13, 5.5)
) -> None:
    """
    Cruza o volume de transacoes por hora do dia com a taxa de fraude por hora
    em eixo duplo. A hora e derivada de time_col (segundos). Renderiza inline.
    """
    tmp = df[[time_col, target_col]].copy()
    tmp["hora"] = (tmp[time_col] // 3600).astype(int) % 24
    agg = tmp.groupby("hora").agg(volume=(target_col, "size"),
                                  fraudes=(target_col, "sum"))
    agg["taxa"] = agg["fraudes"] / agg["volume"] * 100
    agg = agg.reindex(range(24), fill_value=0)

    _fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(agg.index, agg["volume"], color="#1f77b4", alpha=0.55, label="Volume")
    ax1.set_xlabel("Hora do Dia")
    ax1.set_ylabel("Volume de Transacoes", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agg.index, agg["taxa"], color="#d62728", marker="o", lw=2.5, label="Taxa de Fraude")
    ax2.set_ylabel("Taxa de Fraude (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    peak_hour = int(agg["taxa"].idxmax())
    ax2.annotate(
        f"Pico {agg.loc[peak_hour, 'taxa']:.2f}% (h={peak_hour})",
        xy=(peak_hour, agg.loc[peak_hour, "taxa"]),
        xytext=(peak_hour + 2, agg["taxa"].max() * 0.9),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontweight="bold", fontsize=9,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.title("Volume de Transacoes e Taxa de Fraude por Hora do Dia", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str = "Class",
    cols: list[str] | None = None,
    figsize: tuple[float, float] = (17, 14)
) -> None:
    """
    Heatmap da matriz de correlacao ordenada pela correlacao com o alvo
    (decrescente em modulo). Renderiza inline.
    """
    if cols is None:
        cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    ordered = cols + [target_col]
    corr = df[ordered].corr(numeric_only=True)

    order = corr[target_col].drop(target_col).abs().sort_values(ascending=False).index.tolist()
    ordered = order + [target_col]
    corr = corr.loc[ordered, ordered]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    _fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0.0, vmin=-1.0, vmax=1.0,
        square=True, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Matriz de Correlacao Ordenada pela Correlacao com o Alvo", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_target_correlation_ranking(
    df: pd.DataFrame,
    target_col: str = "Class",
    top_n: int = 15,
    figsize: tuple[float, float] = (10, 7)
) -> None:
    """
    Ranking das top_n variaveis por |correlacao| com o alvo, em barras
    coloridas pelo sinal (positivo/negativo). Renderiza inline.
    """
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)
    corr = corr.sort_values()
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in corr.values]

    _fig, ax = plt.subplots(figsize=figsize)
    ax.barh(corr.index, corr.values, color=colors, alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, fontweight="bold")
    ax.set_title(f"Top {top_n} Variaveis por Correlacao com o Alvo ({target_col})",
                 fontweight="bold")
    ax.set_xlabel("Coeficiente de Correlacao de Pearson")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_amount_distribution(
    df: pd.DataFrame,
    target_col: str = "Class",
    amount_col: str = "Amount",
    figsize: tuple[float, float] = (14, 5.5)
) -> None:
    """
    Histograma do log1p do Amount sobreposto por classe e boxplot comparativo.
    Renderiza inline.
    """
    tmp = df[[amount_col, target_col]].copy()
    tmp["log_amount"] = np.log1p(tmp[amount_col])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1 = axes[0]
    sns.histplot(
        data=tmp, x="log_amount", hue=target_col, bins=60, stat="density",
        common_norm=False, palette={0: "#1f77b4", 1: "#d62728"},
        alpha=0.45, ax=ax1, legend=True,
    )
    ax1.set_title("Distribuicao do Valor da Transacao (log1p, por classe)", fontweight="bold")
    ax1.set_xlabel("log(1 + Amount)")
    ax1.set_ylabel("Densidade")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    data0 = tmp[tmp[target_col] == 0]["log_amount"].dropna()
    data1 = tmp[tmp[target_col] == 1]["log_amount"].dropna()
    bp = ax2.boxplot([data0, data1], patch_artist=True,
                     tick_labels=["Legitima (0)", "Fraude (1)"])
    bp["boxes"][0].set_facecolor("#aec7e8")
    bp["boxes"][1].set_facecolor("#ff9896")
    ax2.set_title("Comparacao de Valor por Classe", fontweight="bold")
    ax2.set_ylabel("log(1 + Amount)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("COMPORTAMENTO DE VALOR (AMOUNT) POR CLASSE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_pca_separation(
    df: pd.DataFrame,
    target_col: str = "Class",
    features: tuple[str, ...] = ("V14", "V4", "V12", "V10", "V3"),
    figsize: tuple[float, float] = (15, 8)
) -> None:
    """
    Pequenos multiplos (2x3) de violin plots das componentes PCA lideres
    por classe. Renderiza inline.
    """
    feats = list(features)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.ravel()

    for i, feat in enumerate(feats):
        sns.violinplot(
            data=df, x=target_col, y=feat, hue=target_col, ax=axes_flat[i],
            palette={0: "#1f77b4", 1: "#d62728"}, legend=False, cut=0,
        )
        axes_flat[i].set_title(f"Separacao por classe: {feat}", fontweight="bold")
        axes_flat[i].set_xlabel("Classe")
        axes_flat[i].set_ylabel(feat)
        axes_flat[i].grid(True, alpha=0.3)

    for j in range(len(feats), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("SEPARABILIDADE DAS COMPONENTES PCA LIDERES POR CLASSE",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_feature_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    target_col: str = "Class",
    figsize: tuple[float, float] = (9, 6.5)
) -> None:
    """
    Dispersao bivariada de duas features colorida por classe, com legenda.
    Renderiza inline.
    """
    _fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df, x=x, y=y, hue=target_col, style=target_col,
        palette={0: "#1f77b4", 1: "#d62728"}, alpha=0.45, s=18,
        edgecolor=None, ax=ax,
    )
    ax.set_title(f"Dispersao {x} x {y} por Classe", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Classe", loc="best")
    plt.tight_layout()
    plt.show()


def plot_capture_curve(
    df: pd.DataFrame,
    target_col: str = "Class",
    feature_col: str = "V14",
    q: int = 10,
    ascending: bool = True,
    figsize: tuple[float, float] = (10, 6)
) -> None:
    """
    Curva de captura acumulada de fraudes por decil do preditor lider.
    ascending=True ordena do menor valor (decil 1 de maior risco) para o maior.
    Destaca o lift do primeiro decil. Renderiza inline.
    """
    tmp = df[[feature_col, target_col]].dropna().copy()
    tmp["decil"] = pd.qcut(tmp[feature_col], q=q, labels=False, duplicates="drop")
    if not ascending:
        tmp["decil"] = (q - 1) - tmp["decil"]

    total_fraudes = int(tmp[target_col].sum())
    agg = tmp.groupby("decil", observed=False).agg(
        n=(target_col, "size"), fraudes=(target_col, "sum")
    )
    agg["taxa"] = agg["fraudes"] / agg["n"] * 100
    agg["captura_acum"] = agg["fraudes"].cumsum()
    agg["captura_pct"] = agg["captura_acum"] / total_fraudes * 100
    agg["decil_label"] = agg.index + 1
    base_rate = total_fraudes / len(tmp)
    agg["lift"] = (agg["taxa"] / 100.0) / base_rate

    _fig, ax = plt.subplots(figsize=figsize)
    ax.bar(agg["decil_label"], agg["captura_pct"], color="#1f77b4", alpha=0.7,
           label="Captura acumulada de fraudes (%)")
    ax.plot(agg["decil_label"], agg["captura_pct"], color="#2ca02c",
            marker="o", lw=2.5, label="Curva de captura")
    ax.axhline(100.0, color="gray", linestyle="--", lw=1)

    lift1 = agg["lift"].iloc[0]
    cap1 = agg["captura_pct"].iloc[0]
    ax.annotate(
        f"Decil 1: {cap1:.1f}% das fraudes\nLift = {lift1:.1f}x",
        xy=(agg["decil_label"].iloc[0], cap1),
        xytext=(agg["decil_label"].iloc[1] + 0.5, min(cap1 + 5, 92)),
        arrowprops={"arrowstyle": "->", "color": "black"}, fontweight="bold", fontsize=9,
    )
    ax.set_title(f"Curva de Captura Acumulada de Fraudes por Decil de {feature_col}",
                 fontweight="bold")
    ax.set_xlabel(f"Decil de {feature_col} (1 = maior risco)")
    ax.set_ylabel("Percentual de Fraudes Capturadas")
    ax.set_xticks(range(1, q + 1))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fraud_rate_by_decile(
    df: pd.DataFrame,
    target_col: str = "Class",
    col: str = "Amount",
    q: int = 10,
    figsize: tuple[float, float] = (11, 5.5)
) -> None:
    """
    Taxa de fraude por decil de uma variavel numerica, em barras.
    Renderiza inline.
    """
    tmp = df[[col, target_col]].dropna().copy()
    tmp["decil"] = pd.qcut(tmp[col], q=q, labels=False, duplicates="drop")
    agg = tmp.groupby("decil", observed=False).agg(
        n=(target_col, "size"), fraudes=(target_col, "sum")
    )
    agg["taxa"] = agg["fraudes"] / agg["n"] * 100
    agg["decil_label"] = agg.index + 1

    _fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(agg["decil_label"], agg["taxa"], color="#9467bd", alpha=0.85)
    base_rate = tmp[target_col].mean() * 100
    ax.axhline(base_rate, color="#d62728", linestyle="--", lw=1.5,
               label=f"Taxa basal = {base_rate:.3f}%")
    for b, v in zip(bars, agg["taxa"]):
        ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_title(f"Taxa de Fraude por Decil de {col}", fontweight="bold")
    ax.set_xlabel(f"Decil de {col}")
    ax.set_ylabel("Taxa de Fraude (%)")
    ax.set_xticks(range(1, q + 1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_temporal_stability(
    partitions: dict[str, pd.DataFrame],
    target_col: str = "Class",
    figsize: tuple[float, float] = (13, 5.5)
) -> None:
    """
    Compara volume e taxa basal de fraude entre particoes temporais
    (Treino/Teste/OOT) em barras lado a lado. Renderiza inline.
    """
    names = list(partitions.keys())
    volumes = [len(partitions[k]) for k in names]
    fraudes = [int(partitions[k][target_col].sum()) for k in names]
    taxas = [partitions[k][target_col].mean() * 100 for k in names]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1 = axes[0]
    bars = ax1.bar(names, volumes, color="#1f77b4", alpha=0.85)
    for b, v, f in zip(bars, volumes, fraudes):
        ax1.text(b.get_x() + b.get_width() / 2.0, v * 1.01,
                 f"{v:,}".replace(",", ".") + f"\n({f} fraudes)",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_title("Volume de Transacoes por Particao", fontweight="bold")
    ax1.set_ylabel("Numero de Transacoes")
    ax1.set_ylim(0, max(volumes) * 1.18)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    bars2 = ax2.bar(names, taxas, color="#d62728", alpha=0.85)
    for b, v in zip(bars2, taxas):
        ax2.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.4f}%", ha="center",
                 va="bottom", fontsize=8, fontweight="bold")
    ax2.set_title("Taxa Basal de Fraude por Particao", fontweight="bold")
    ax2.set_ylabel("Taxa de Fraude (%)")
    ax2.set_ylim(0, max(taxas) * 1.3)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("ESTABILIDADE TEMPORAL DO FENOMENO (TREINO / TESTE / OOT)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()