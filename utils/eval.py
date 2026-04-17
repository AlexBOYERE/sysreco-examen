"""Évaluation Recall@K en tranches d'utilisateurs.

On ne matérialise jamais la matrice de scores complète (n_users × n_items).
À chaque tranche on calcule scores_batch = U_hat[batch] @ V_hat.T
(~30 MB pour batch=500 × 13 584 items en float32), puis on extrait les top-K.
"""
import numpy as np


def _excludes_for_batch(exclude_csr, row_start: int, row_end: int):
    """Retourne (rows_local, cols) pour les interactions à exclure dans la
    tranche utilisateurs [row_start, row_end[. rows_local est recalé sur 0.
    """
    if exclude_csr is None:
        return None, None
    sub = exclude_csr[row_start:row_end].tocoo()
    return sub.row, sub.col


def recall_at_k(
    U_hat: np.ndarray,
    V_hat: np.ndarray,
    target,              # csr (n_users, n_items)
    exclude=None,        # csr ou None
    k: int = 100,
    batch_users: int = 500,
) -> float:
    """Recall@K moyenné sur les utilisateurs ayant >= 1 cible."""
    n_u, d = U_hat.shape
    n_i = V_hat.shape[0]
    k_use = min(k, n_i)

    tgt = target.tocsr()
    recalls = []

    for start in range(0, n_u, batch_users):
        end = min(start + batch_users, n_u)
        scores = U_hat[start:end] @ V_hat.T      # (bs, n_i)
        # Appliquer les exclusions pour cette tranche
        ex_rows, ex_cols = _excludes_for_batch(exclude, start, end)
        if ex_rows is not None:
            scores[ex_rows, ex_cols] = -np.inf

        top_idx = np.argpartition(-scores, kth=k_use - 1, axis=1)[:, :k_use]
        del scores

        for j, u in enumerate(range(start, end)):
            a, b = tgt.indptr[u], tgt.indptr[u + 1]
            pos = tgt.indices[a:b]
            if len(pos) == 0:
                continue
            hits = np.isin(pos, top_idx[j]).sum()
            recalls.append(hits / len(pos))

    return float(np.mean(recalls)) if recalls else 0.0


def recall_at_k_multi(
    U_hat: np.ndarray,
    V_hat: np.ndarray,
    target,
    exclude=None,
    ks=(50, 100, 200, 300, 500),
    batch_users: int = 500,
) -> dict:
    """Recall@k pour plusieurs k, en tranches d'utilisateurs."""
    n_u = U_hat.shape[0]
    n_i = V_hat.shape[0]
    ks_valid = sorted(set(int(k) for k in ks if k <= n_i))
    if not ks_valid:
        return {}
    max_k = ks_valid[-1]

    tgt = target.tocsr()
    out = {k: [] for k in ks_valid}

    for start in range(0, n_u, batch_users):
        end = min(start + batch_users, n_u)
        scores = U_hat[start:end] @ V_hat.T
        ex_rows, ex_cols = _excludes_for_batch(exclude, start, end)
        if ex_rows is not None:
            scores[ex_rows, ex_cols] = -np.inf

        part = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        row_idx = np.arange(end - start)[:, None]
        part_scores = scores[row_idx, part]
        sorted_within = np.argsort(-part_scores, axis=1)
        top_sorted = np.take_along_axis(part, sorted_within, axis=1)
        del scores, part, part_scores, sorted_within

        for j, u in enumerate(range(start, end)):
            a, b = tgt.indptr[u], tgt.indptr[u + 1]
            pos = tgt.indices[a:b]
            if len(pos) == 0:
                continue
            pos_set = set(pos.tolist())
            ranked = top_sorted[j]
            hits_cum = 0
            ki = 0
            for rank, item in enumerate(ranked, start=1):
                if item in pos_set:
                    hits_cum += 1
                while ki < len(ks_valid) and rank == ks_valid[ki]:
                    out[ks_valid[ki]].append(hits_cum / len(pos))
                    ki += 1
                if ki >= len(ks_valid):
                    break

    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}
