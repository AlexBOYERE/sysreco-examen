"""Weighted Matrix Factorization via ALS (Hu, Koren, Volinsky 2008).

Modèle implicite :
  min_{U,V}  sum_{u,v} c_uv (p_uv - U_u V_v^T)^2  +  λ (||U||² + ||V||²)
avec  p_uv = 1 si interaction, 0 sinon
     c_uv = 1 + α * p_uv  (confiance)

ALS résout chaque U_u (V_v) à sous-problème convexe fermé, en alternant.

Implémentation efficace (trick du papier HKV08) :
  Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y
  Pour R binaire : (C^u - I)_vv = α si v est positif, 0 sinon
  => Y^T (C^u - I) Y = α * Y_pos^T Y_pos   (petite matrice rank x rank)
  => Y^T C^u p(u)    = (1 + α) * sum_{v ∈ pos(u)} Y_v

Le terme Y^T Y est calculé UNE FOIS par demi-itération, puis réutilisé
pour tous les utilisateurs. Les Y_pos sont juste quelques lignes.
"""
import numpy as np


def wmf_als(
    R,
    rank: int = 200,
    reg: float = 0.01,
    alpha: float = 40.0,
    n_iter: int = 15,
    seed: int = 0,
    verbose: bool = True,
):
    """Entraîne WMF par ALS. R doit être csr binaire.

    Returns:
        U : (n_users, rank) float32
        V : (n_items, rank) float32
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    R = R.tocsr()
    Rt = R.T.tocsr()

    U = (rng.standard_normal((n_u, rank)) * 0.01).astype(np.float32)
    V = (rng.standard_normal((n_i, rank)) * 0.01).astype(np.float32)

    reg_eye = (reg * np.eye(rank)).astype(np.float32)

    def als_step(X, Y, R_src):
        """Met à jour X étant donnés Y fixé et R_src (csr).
        Pour chaque i : R_src.indices[indptr[i]:indptr[i+1]] = items positifs.
        """
        YtY = (Y.T @ Y).astype(np.float32)
        X_new = np.empty_like(X)
        for i in range(X.shape[0]):
            s, e = R_src.indptr[i], R_src.indptr[i + 1]
            pos = R_src.indices[s:e]
            if len(pos) == 0:
                X_new[i] = 0.0
                continue
            Y_pos = Y[pos]
            A = YtY + alpha * (Y_pos.T @ Y_pos) + reg_eye
            b = (1.0 + alpha) * Y_pos.sum(axis=0)
            X_new[i] = np.linalg.solve(A, b).astype(np.float32)
        return X_new

    for it in range(n_iter):
        U = als_step(U, V, R)
        V = als_step(V, U, Rt)
        if verbose:
            loss = _sampled_loss(U, V, R, alpha)
            print(f"  [WMF] iter {it+1}/{n_iter}  sampled_loss≈{loss:.4f}")

    return U, V


def _sampled_loss(U, V, R, alpha, n_sample=500, seed=42):
    """Pseudo-loss échantillonnée : moyenne de c_uv*(p_uv - U·V)^2 sur un
    échantillon d'utilisateurs (positifs uniquement). Indicatif, pas comparable
    entre hyper-paramètres.
    """
    rng = np.random.default_rng(seed)
    sample_u = rng.integers(0, U.shape[0], size=n_sample)
    total = 0.0
    count = 0
    for u in sample_u:
        s, e = R.indptr[u], R.indptr[u + 1]
        pos = R.indices[s:e]
        if len(pos) == 0:
            continue
        pred = U[u] @ V[pos].T
        total += ((1.0 + alpha) * (pred - 1.0) ** 2).sum()
        count += len(pos)
    return total / max(count, 1)
