"""
recommender_tools.py

Módulo de funções determinísticas e reutilizáveis para Sistemas de Recomendação.
Implementa a arquitetura em 3 camadas (Collector, Ranker, Server), métricas de ranqueamento
(MAP@K, NDCG@K, Hit Rate@K, Cobertura, Gini) e estratégias de pós-processamento e exploração.
"""

import math
import random
from typing import Dict, List, Set, Tuple, Union, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix


# =============================================================================
# 1. CAMADA 1: COLLECTOR & MATRIZES DE SIMILARIDADE
# =============================================================================

def compute_similarity_matrix(
    interaction_matrix: csr_matrix,
    kind: str = 'item',
    method: str = 'cosine'
) -> csr_matrix:
    """
    Calcula a matriz de similaridade entre itens ou usuários a partir de uma
    matriz de interação esparsa.

    Parameters
    ----------
    interaction_matrix : csr_matrix
        Matriz esparsa de dimensões (n_users, n_items).
    kind : str, default='item'
        Tipo de similaridade: 'item' para (n_items, n_items) ou 'user' para (n_users, n_users).
    method : str, default='cosine'
        Método de similaridade: 'cosine' ou 'pearson'.

    Returns
    -------
    csr_matrix
        Matriz de similaridade esparsa calculada.
    """
    matrix = interaction_matrix.copy().astype(np.float64)
    if kind == 'user':
        pass
    elif kind == 'item':
        matrix = matrix.T.tocsr()
    else:
        raise ValueError("O parâmetro 'kind' deve ser 'item' ou 'user'.")

    if method == 'pearson':
        # Subtrai a média de cada linha (usuário/item) para centralizar
        row_means = np.array(matrix.sum(axis=1)).flatten() / np.maximum(matrix.getnnz(axis=1), 1)
        # Aplica a centralização apenas nos elementos não-nulos
        matrix_data = matrix.data
        rows, cols = matrix.nonzero()
        for i in range(len(matrix_data)):
            matrix_data[i] -= row_means[rows[i]]
        matrix = csr_matrix((matrix_data, (rows, cols)), shape=matrix.shape)

    # Cálculo da similaridade por produto escalar normalizado (Cosseno / Pearson)
    norms = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
    norms[norms == 0] = 1.0  # Evita divisão por zero
    
    # Normaliza as linhas
    inv_norms = 1.0 / norms
    diag_matrix = sp.diags(inv_norms)
    normalized_matrix = diag_matrix.dot(matrix)

    # Produto matricial para calcular a matriz de similaridade
    similarity = normalized_matrix.dot(normalized_matrix.T)
    
    # Zera a diagonal principal para evitar auto-recomendação
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()

    return similarity.tocsr()


def most_popular_recommender(
    train_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: Optional[str] = None,
    top_k: int = 10
) -> List[Union[int, str]]:
    """
    Baseline Most-Popular Item Recommender (MPIR).
    Retorna os itens mais populares com base na contagem ou soma de interações.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame de treino contendo os logs de interação.
    user_col : str
        Nome da coluna do ID do usuário.
    item_col : str
        Nome da coluna do ID do item.
    rating_col : str, optional
        Nome da coluna de rating/frequência para ponderação.
    top_k : int, default=10
        Número de itens populares a serem retornados.

    Returns
    -------
    List[Union[int, str]]
        Lista com os IDs dos Top-K itens mais populares.
    """
    if rating_col and rating_col in train_df.columns:
        popularity = train_df.groupby(item_col)[rating_col].sum()
    else:
        popularity = train_df.groupby(item_col)[user_col].nunique()

    top_items = popularity.sort_values(ascending=False).head(top_k).index.tolist()
    return top_items


# =============================================================================
# 2. CAMADA 2: RANKER & FATORAÇÃO MATRICIAL (iALS)
# =============================================================================

def implicit_als_factorization(
    R: csr_matrix,
    n_factors: int = 32,
    alpha: float = 40.0,
    reg_lambda: float = 0.01,
    n_iterations: int = 15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fatoração de Matriz via Implicit Alternating Least Squares (iALS)
    para dados de Implicit Feedback.

    Parameters
    ----------
    R : csr_matrix
        Matriz de interações esparsa (n_users, n_items) com contagens/frequências d_{u,i}.
    n_factors : int, default=32
        Número de fatores latentes D.
    alpha : float, default=40.0
        Taxa de inclinação da confiança c_{u,i} = 1 + alpha * d_{u,i}.
    reg_lambda : float, default=0.01
        Termo de regularização L2.
    n_iterations : int, default=15
        Número de iterações do ALS.
    random_state : int, default=42
        Semente aleatória para reprodutibilidade.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: Matriz de fatores latentes dos usuários (n_users, n_factors)
        Y: Matriz de fatores latentes dos itens (n_items, n_factors)
    """
    np.random.seed(random_state)
    n_users, n_items = R.shape

    # Inicialização aleatória dos vetores latentes
    X = np.random.normal(scale=0.01, size=(n_users, n_factors))
    Y = np.random.normal(scale=0.01, size=(n_items, n_factors))

    # Matriz de preferência binária P e confiança C
    R_csc = R.tocsc()
    R_csr = R.tocsr()

    for it in range(n_iterations):
        # 1. Atualização dos fatores dos usuários X
        YTY = Y.T.dot(Y)
        I_f = np.eye(n_factors)
        
        for u in range(n_users):
            row = R_csr[u]
            if row.nnz == 0:
                continue
            indices = row.indices
            data = row.data
            
            # Confiança c_u - 1 = alpha * d_{u,i}
            c_u = 1.0 + alpha * data
            c_minus_1 = c_u - 1.0
            
            # Submatriz de itens que o usuário interagiu
            Y_u = Y[indices, :]
            
            # A_u = Y^T Y + Y_u^T diag(c - 1) Y_u + lambda * I
            A_u = YTY + (Y_u.T * c_minus_1).dot(Y_u) + reg_lambda * I_f
            
            # b_u = Y_u^T (c * p) onde p = 1 para itens interagidos
            b_u = Y_u.T.dot(c_u)
            
            X[u, :] = np.linalg.solve(A_u, b_u)

        # 2. Atualização dos fatores dos itens Y
        XTX = X.T.dot(X)
        for i in range(n_items):
            col = R_csc[:, i]
            if col.nnz == 0:
                continue
            indices = col.indices
            data = col.data
            
            c_i = 1.0 + alpha * data
            c_minus_1 = c_i - 1.0
            
            X_i = X[indices, :]
            
            A_i = XTX + (X_i.T * c_minus_1).dot(X_i) + reg_lambda * I_f
            b_i = X_i.T.dot(c_i)
            
            Y[i, :] = np.linalg.solve(A_i, b_i)

    return X, Y


# =============================================================================
# 3. MÉTRICAS DE AVALIAÇÃO DE RANQUEAMENTO
# =============================================================================

def calculate_map_at_k(
    actual_dict: Dict[Union[int, str], List[Union[int, str]]],
    predicted_dict: Dict[Union[int, str], List[Union[int, str]]],
    k: int = 10
) -> float:
    """
    Calcula o Mean Average Precision at K (MAP@K).

    Parameters
    ----------
    actual_dict : Dict[user, List[relevant_items]]
        Dicionário com o conjunto verdadeiro de itens relevantes por usuário.
    predicted_dict : Dict[user, List[recommended_items]]
        Dicionário com a lista ordenada de itens recomendados por usuário.
    k : int, default=10
        Limite de posições no ranking.

    Returns
    -------
    float
        Valor médio de MAP@K entre todos os usuários.
    """
    average_precisions = []

    for user, actual_items in actual_dict.items():
        if not actual_items:
            continue

        predicted_items = predicted_dict.get(user, [])[:k]
        score = 0.0
        n_hits = 0.0

        for i, p in enumerate(predicted_items):
            if p in actual_items and p not in predicted_items[:i]:
                n_hits += 1.0
                score += n_hits / (i + 1.0)

        ap = score / min(len(actual_items), k)
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def calculate_ndcg_at_k(
    actual_dict: Dict[Union[int, str], List[Union[int, str]]],
    predicted_dict: Dict[Union[int, str], List[Union[int, str]]],
    k: int = 10
) -> float:
    """
    Calcula o Normalized Discounted Cumulative Gain at K (NDCG@K).

    Parameters
    ----------
    actual_dict : Dict[user, List[relevant_items]]
        Dicionário com o conjunto verdadeiro de itens relevantes por usuário.
    predicted_dict : Dict[user, List[recommended_items]]
        Dicionário com a lista ordenada de itens recomendados por usuário.
    k : int, default=10
        Limite de posições no ranking.

    Returns
    -------
    float
        Valor médio de NDCG@K entre todos os usuários.
    """
    ndcg_scores = []

    for user, actual_items in actual_dict.items():
        if not actual_items:
            continue

        actual_set = set(actual_items)
        predicted_items = predicted_dict.get(user, [])[:k]

        dcg = 0.0
        for i, p in enumerate(predicted_items):
            if p in actual_set:
                dcg += 1.0 / math.log2(i + 2)

        # Ideal DCG (IDCG)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(actual_set), k)))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def calculate_hit_rate_at_k(
    actual_dict: Dict[Union[int, str], List[Union[int, str]]],
    predicted_dict: Dict[Union[int, str], List[Union[int, str]]],
    k: int = 10
) -> float:
    """
    Calcula a taxa de acerto (Hit Rate at K).

    Returns
    -------
    float
        Proporção de usuários com pelo menos um item recomendado correto nos Top-K.
    """
    hits = 0
    total_users = 0

    for user, actual_items in actual_dict.items():
        if not actual_items:
            continue

        total_users += 1
        actual_set = set(actual_items)
        predicted_items = predicted_dict.get(user, [])[:k]

        if any(p in actual_set for p in predicted_items):
            hits += 1

    return hits / total_users if total_users > 0 else 0.0


def calculate_catalog_coverage(
    predicted_dict: Dict[Union[int, str], List[Union[int, str]]],
    total_catalog_items: int,
    k: int = 10
) -> float:
    """
    Calcula a Cobertura de Catálogo (Percentual de itens distintos recomendados no Top-K).

    Parameters
    ----------
    predicted_dict : Dict[user, List[recommended_items]]
        Recomendações geradas por usuário.
    total_catalog_items : int
        Contagem total de itens disponíveis no catálogo.
    k : int, default=10
        Limite de corte.

    Returns
    -------
    float
        Percentual de cobertura do catálogo em intervalo [0, 100].
    """
    recommended_items_set = set()
    for recs in predicted_dict.values():
        recommended_items_set.update(recs[:k])

    coverage_pct = (len(recommended_items_set) / total_catalog_items) * 100.0
    return float(coverage_pct)


def calculate_gini_index(
    predicted_dict: Dict[Union[int, str], List[Union[int, str]]],
    total_catalog_items: int,
    k: int = 10
) -> float:
    """
    Calcula o Índice de Gini de Recomendação para medir a desigualdade/concentração
    de recomendações (Viés de Popularidade).

    Returns
    -------
    float
        Índice de Gini entre [0.0, 1.0]. 0 = perfeitamente igualitário, 1 = item único recomendado.
    """
    item_counts: Dict[Union[int, str], int] = {}
    for recs in predicted_dict.values():
        for item in recs[:k]:
            item_counts[item] = item_counts.get(item, 0) + 1

    # Adiciona contagem zero para itens do catálogo não recomendados
    counts = np.array(list(item_counts.values()) + [0] * (total_catalog_items - len(item_counts)))
    counts.sort()

    n = len(counts)
    if n == 0 or np.sum(counts) == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = ((2 * np.sum(index * counts)) / (n * np.sum(counts))) - ((n + 1) / n)
    return float(gini)


# =============================================================================
# 4. CAMADA 3: SERVER, BLOOM FILTERS & EXPLORAÇÃO EPSILON-GREEDY
# =============================================================================

def bloom_filter_deduplication(
    recommendations: List[Union[int, str]],
    consumed_items: Set[Union[int, str]]
) -> List[Union[int, str]]:
    """
    Filtra da lista de recomendações os itens já consumidos pelo usuário.

    Parameters
    ----------
    recommendations : List[Union[int, str]]
        Lista inicial de itens recomendados.
    consumed_items : Set[Union[int, str]]
        Conjunto de itens consumidos historicamente.

    Returns
    -------
    List[Union[int, str]]
        Lista filtrada contendo apenas itens não consumidos.
    """
    return [item for item in recommendations if item not in consumed_items]


def epsilon_greedy_rerank(
    ranker_recs: List[Union[int, str]],
    candidate_pool: List[Union[int, str]],
    epsilon: float = 0.1,
    top_k: int = 10,
    random_seed: Optional[int] = 42
) -> List[Union[int, str]]:
    """
    Aplica o algoritmo Epsilon-Greedy para reordenar e injetar itens de exploração
    na lista do Ranker.

    Parameters
    ----------
    ranker_recs : List[Union[int, str]]
        Lista de recomendações ordenadas do Ranker.
    candidate_pool : List[Union[int, str]]
        Pool estendido de candidatos do Collector para amostragem exploratória.
    epsilon : float, default=0.1
        Probabilidade de selecionar um item exploratório aleatório.
    top_k : int, default=10
        Tamanho da lista final de recomendações.
    random_seed : int, optional
        Semente aleatória para reprodutibilidade.

    Returns
    -------
    List[Union[int, str]]
        Lista final reordenada com itens de exploração injetados.
    """
    if random_seed is not None:
        random.seed(random_seed)

    final_feed: List[Union[int, str]] = []
    seen = set()

    ranker_idx = 0
    exploratory_pool = [i for i in candidate_pool if i not in set(ranker_recs)]

    while len(final_feed) < top_k:
        # Decisão Epsilon-Greedy
        if random.random() < epsilon and exploratory_pool:
            # Seleção de item exploratório
            candidate = random.choice(exploratory_pool)
            exploratory_pool.remove(candidate)
        else:
            # Seleção do próximo item do Ranker
            if ranker_idx < len(ranker_recs):
                candidate = ranker_recs[ranker_idx]
                ranker_idx += 1
            elif exploratory_pool:
                candidate = random.choice(exploratory_pool)
                exploratory_pool.remove(candidate)
            else:
                break

        if candidate not in seen:
            seen.add(candidate)
            final_feed.append(candidate)

    return final_feed
