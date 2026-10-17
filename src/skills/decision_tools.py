"""
Módulo de ferramentas determinísticas para decisão financeira, otimização de
threshold (c*) sob Unit Economics, curvas de lucro líquido, simulação de
capacidade operacional (Top-K) e tabelas de lift por decis.

Convenção de Unit Economics:
    value_tp: ganho financeiro (R$) por Verdadeiro Positivo (contato correto).
    cost_fp:  custo operacional (R$) por Falso Positivo (contato indevido).
    cost_fn:  perda financeira (R$) por Falso Negativo (oportunidade ignorada).
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def c_star_amount_dependent(amount, cost_fp: float = 5.0):
    """
    Calcula o corte economico otimo (c*) por transacao sob Unit Economics
    dependente do valor.

    Com value_tp = Amount (ganho por bloquear fraude) e cost_fn = Amount (perda
    integral ao nao bloquear), o valor esperado de bloquear supera o de nao
    bloquear quando p >= cost_fp / (2 * Amount + cost_fp).

    Parameters
    ----------
    amount : float | array-like
        Valor da transacao (R$).
    cost_fp : float
        Custo fixo por Falso Positivo (bloqueio indevido), em R$.

    Returns
    -------
    float | np.ndarray
        Corte otimo por transacao no intervalo (0, 1].
    """
    amount = np.asarray(amount, dtype=float)
    return cost_fp / (2.0 * amount + cost_fp)


def _profit_amount_rule(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amount: np.ndarray,
    cost_fp: float = 5.0
) -> dict[str, float]:
    """Matriz de confusao financeira e lucro liquido dado um vetor de decisoes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amount = np.asarray(amount, dtype=float)

    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    tp = float(tp_mask.sum())
    fp = float(fp_mask.sum())
    fn = float(fn_mask.sum())
    tn = float(tn_mask.sum())
    amount_tp = float(amount[tp_mask].sum())
    amount_fn = float(amount[fn_mask].sum())

    net_profit = amount_tp - (cost_fp * fp) - amount_fn

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "amount_tp": amount_tp,
        "amount_fn": amount_fn,
        "net_profit": float(net_profit)
    }


def net_profit_amount_dependent(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    cost_fp: float = 5.0,
    threshold=None
) -> dict[str, float]:
    """
    Lucro liquido com decisao dependente do valor da transacao.

    Se `threshold` for None, aplica a regra Bayes-otima por transacao
    (bloqueio quando y_score >= c*(Amount)). Se `threshold` for um escalar,
    aplica um corte global constante a toda a carteira.

    Returns
    -------
    Dict[str, float]
        tp, fp, fn, tn, amount_tp, amount_fn, net_profit e modo do corte.
    """
    y_score = np.asarray(y_score)
    amount = np.asarray(amount, dtype=float)

    if threshold is None:
        thr = c_star_amount_dependent(amount, cost_fp)
        mode = "c_star_por_transacao"
    else:
        thr = np.full(amount.shape, float(threshold), dtype=float)
        mode = "corte_global"

    y_pred = (y_score >= thr).astype(int)
    result = _profit_amount_rule(y_true, y_pred, amount, cost_fp)
    result["threshold_mode"] = mode
    result["threshold"] = float(threshold) if threshold is not None else np.nan
    return result


def optimal_threshold_amount_dependent(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    cost_fp: float = 5.0,
    n_grid: int = 200
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Busca o corte global escalar que maximiza o lucro liquido com economia
    dependente do valor da transacao.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        best: melhor threshold global e lucro associado.
        curve: curva (threshold x net_profit) para o dashboard executivo.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    records = [
        net_profit_amount_dependent(y_true, y_score, amount, cost_fp, threshold=t)
        for t in grid
    ]
    curve = pd.DataFrame(records)
    best_idx = curve["net_profit"].idxmax()
    best = curve.loc[best_idx].to_dict()
    return best, curve


def baseline_no_action_profit(y_true: np.ndarray, amount: np.ndarray) -> float:
    """
    Lucro liquido da politica "nao bloquear nada": perda integral das fraudes.

    Returns
    -------
    float
        Valor negativo igual a -sum(Amount das transacoes fraudulentas).
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    return float(-amount[y_true == 1].sum())


def capacity_simulation_amount_dependent(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    cost_fp: float = 5.0,
    capacities: list[float] | None = None,
    n_total: int | None = None
) -> pd.DataFrame:
    """
    Simula lucro liquido sob restricao de capacidade (Top-K) com economia
    dependente do valor da transacao.

    Ordena a base por score decrescente, bloqueia estritamente os Top-K e
    contabiliza TP/FP/FN em valor. `n_total` permite converter capacidades
    absolutas em fracoes da base original.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    amount = np.asarray(amount, dtype=float)
    n = len(y_true)

    if capacities is None:
        capacities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    if n_total is not None:
        capacities = [k / n_total for k in capacities]

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    amount_sorted = amount[order]

    baseline_profit = baseline_no_action_profit(y_true, amount)
    total_events = float((y_true == 1).sum())

    records = []
    for frac in capacities:
        k = max(1, round(frac * n))
        top_true = y_sorted[:k]
        top_amount = amount_sorted[:k]
        tp_mask = top_true == 1
        tp = float(tp_mask.sum())
        fp = float(k - tp)
        fn = float(total_events - tp)
        amount_tp = float(top_amount[tp_mask].sum())
        amount_fn = float(amount[y_true == 1].sum() - amount_tp)
        net_profit = amount_tp - (cost_fp * fp) - amount_fn

        records.append({
            "frac_capacidade": round(frac, 4),
            "k": k,
            "taxa_contatada": round(frac * 100, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "net_profit": float(net_profit),
            "roi_relativo": (
                float(net_profit / baseline_profit) if baseline_profit else np.nan
            )
        })

    return pd.DataFrame(records)


def theoretical_threshold(
    value_tp: float,
    cost_fp: float,
    cost_fn: float
) -> float:
    """
    Calcula o threshold financeiro teórico (c*).

    O corte c* é a probabilidade em que o valor esperado de contatar iguala o
    valor esperado de não contatar:
        c* = cost_fp / (value_tp + cost_fp + cost_fn)

    Returns
    -------
    float
        Probabilidade de corte ótima no intervalo [0, 1].
    """
    denominator = value_tp + cost_fp + cost_fn
    if denominator <= 0:
        raise ValueError("value_tp + cost_fp + cost_fn deve ser positivo.")
    return float(cost_fp / denominator)


def net_profit_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    value_tp: float,
    cost_fp: float,
    cost_fn: float
) -> dict[str, float]:
    """
    Calcula o lucro líquido (R$) e a matriz de confusão para um threshold fixo.

    Regra de decisão: intervenção quando y_score >= threshold.

    Returns
    -------
    Dict[str, float]
        Dicionário com tp, fp, fn, tn, net_profit e threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))

    net_profit = (tp * value_tp) - (fp * cost_fp) - (fn * cost_fn)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "net_profit": float(net_profit)
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    value_tp: float,
    cost_fp: float,
    cost_fn: float,
    n_grid: int = 200
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Otimiza o threshold por busca em grade e retorna a curva de lucro líquido.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        best: melhor resultado (threshold e lucro) encontrado.
        curve: DataFrame com a curva completa (threshold x net_profit).
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    records = [
        net_profit_at_threshold(y_true, y_score, t, value_tp, cost_fp, cost_fn)
        for t in grid
    ]
    curve = pd.DataFrame(records)

    best_idx = curve["net_profit"].idxmax()
    best = curve.loc[best_idx].to_dict()
    return best, curve


def capacity_simulation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    value_tp: float,
    cost_fp: float,
    cost_fn: float,
    capacities: list[float] | None = None,
    n_total: int | None = None
) -> pd.DataFrame:
    """
    Simula o resultado financeiro sob restrição de capacidade operacional (Top-K).

    Ordena a base por score decrescente e contata estritamente os Top-K registros.
    A avaliação pode ser feita na base completa (n_total) ou por frações de
    capacidade (ex: [0.05, 0.10, 0.20, 0.30]).

    Returns
    -------
    pd.DataFrame
        Tabela com k, taxa_contatada, tp, fp, fn, net_profit e roi_relativo.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    if capacities is None:
        capacities = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    if n_total is not None:
        capacities = [k / n_total for k in capacities]

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    baseline_profit = net_profit_at_threshold(
        y_true, y_score, 0.0, value_tp, cost_fp, cost_fn
    )["net_profit"]

    records = []
    for frac in capacities:
        k = max(1, round(frac * n))
        top = y_sorted[:k]
        tp = float(np.sum(top == 1))
        fp = float(k - tp)
        fn = float(np.sum(y_true == 1) - tp)
        net_profit = (tp * value_tp) - (fp * cost_fp) - (fn * cost_fn)

        records.append({
            "frac_capacidade": round(frac, 4),
            "k": k,
            "taxa_contatada": round(frac * 100, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "net_profit": float(net_profit),
            "roi_relativo": float(net_profit / baseline_profit) if baseline_profit else np.nan
        })

    return pd.DataFrame(records)


def decile_lift_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    q: int = 10
) -> pd.DataFrame:
    """
    Constrói a tabela de lift por decil de risco/score (decis ordenda por score).

    Returns
    -------
    pd.DataFrame
        Tabela com decil, n, eventos, taxa_evento, lift e eventos_acumulados_pct.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_score": np.asarray(y_score)})
    df["decil"] = pd.qcut(df["y_score"].rank(method="first", ascending=False), q, labels=False) + 1

    grouped = df.groupby("decil", observed=False).agg(
        n=("y_true", "count"),
        eventos=("y_true", "sum")
    ).reset_index()
    grouped["taxa_evento"] = grouped["eventos"] / grouped["n"]
    base_rate = df["y_true"].mean()
    grouped["lift"] = grouped["taxa_evento"] / base_rate if base_rate else np.nan

    total_eventos = grouped["eventos"].sum()
    grouped["eventos_acumulados_pct"] = (
        grouped["eventos"].cumsum() / total_eventos * 100 if total_eventos else 0.0
    )

    grouped["taxa_evento"] = grouped["taxa_evento"].round(4)
    grouped["lift"] = grouped["lift"].round(3)
    grouped["eventos_acumulados_pct"] = grouped["eventos_acumulados_pct"].round(2)
    return grouped


def plot_profit_and_capacity(
    profit_curve: pd.DataFrame,
    capacity_table: pd.DataFrame,
    theoretical_c_star: float | None = None,
    empirical_c_star: float | None = None
) -> None:
    """
    Gera o dashboard 1x2 de decisão financeira: curva de lucro por threshold e
    lucro líquido por capacidade operacional (Top-K). Renderização inline.
    """
    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.25)

    ax1 = axes[0]
    ax1.plot(profit_curve["threshold"], profit_curve["net_profit"], color="#1f77b4", lw=2.5)
    ax1.axhline(0, color="gray", linestyle="--", lw=1)
    if theoretical_c_star is not None:
        ax1.axvline(
            theoretical_c_star, color="#d62728", linestyle=":",
            lw=2, label=f"c* teórico = {theoretical_c_star:.3f}"
        )
    if empirical_c_star is not None:
        ax1.axvline(
            empirical_c_star, color="#2ca02c", linestyle="--",
            lw=2, label=f"c* empírico = {empirical_c_star:.3f}"
        )
    ax1.set_title("1. Curva de Lucro Líquido por Threshold", fontweight="bold")
    ax1.set_xlabel("Threshold de Probabilidade")
    ax1.set_ylabel("Lucro Líquido (R$)")
    ax1.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(
        capacity_table["taxa_contatada"], capacity_table["net_profit"],
        marker="o", color="#9467bd", lw=2.5
    )
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_title("2. Lucro Líquido sob Capacidade Operacional (Top-K)", fontweight="bold")
    ax2.set_xlabel("% da Base Contatada")
    ax2.set_ylabel("Lucro Líquido (R$)")

    plt.suptitle(
        "EXECUTIVE DASHBOARD: FINANCIAL DECISION & CAPACITY",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.show()
