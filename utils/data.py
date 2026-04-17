"""Chargement des données CiteULike + split warm hold-out.

Phi_V est ~0.1 % dense (TF-IDF) et est gardée en scipy CSR (~1 MB au lieu
de 441 MB en dense).

Note méthodologique : dans le prep fourni, la partie warm de R_test est
identique à R_train (aucune interaction warm tenue à l'écart). On crée donc
un hold-out warm en retirant x % des interactions par utilisateur de R_train
pour obtenir une vraie évaluation warm-start.
"""
from pathlib import Path
import gc

import numpy as np
from scipy.sparse import load_npz, csr_matrix


def load_citeulike(data_dir: str = "data"):
    """Charge les fichiers préparés.

    Returns un dict avec :
        R_train_warm    : csr (n_users, n_warm)
        R_test_cold     : csr (n_users, n_cold)
        Phi_V_warm      : csr (n_warm, n_feat)
        Phi_V_cold      : csr (n_cold, n_feat)
        warm_items      : int array (n_warm,)  — indices full-space
        cold_items      : int array (n_cold,)
    """
    data_dir = Path(data_dir)
    R_train_warm = load_npz(data_dir / "R_train.npz").tocsr().astype(np.float32)
    R_test_full = load_npz(data_dir / "R_test.npz").tocsr().astype(np.float32)
    warm_items = np.load(data_dir / "warm_start_items.npy").astype(np.int64)
    cold_items = np.load(data_dir / "cold_start_items.npy").astype(np.int64)

    # Phi_V : chargement dense temporaire pour construire les CSR puis libération
    Phi_dense = np.load(data_dir / "Phi_V.npy").astype(np.float32, copy=False)
    Phi_V_warm = csr_matrix(Phi_dense[warm_items])
    Phi_V_cold = csr_matrix(Phi_dense[cold_items])
    del Phi_dense
    gc.collect()

    R_test_cold = R_test_full[:, cold_items].tocsr().astype(np.float32)

    return {
        "R_train_warm": R_train_warm,
        "R_test_cold": R_test_cold,
        "Phi_V_warm": Phi_V_warm,
        "Phi_V_cold": Phi_V_cold,
        "warm_items": warm_items,
        "cold_items": cold_items,
    }


def make_warm_holdout(R_train_warm, holdout_frac: float = 0.2, seed: int = 42):
    """Carve un hold-out warm-start par utilisateur.

    Au moins 1 interaction reste en train (utilisateurs avec 1 seule
    interaction gardent tout en train, rien en test).
    """
    rng = np.random.default_rng(seed)
    R = R_train_warm.tolil()
    n_u, n_i = R.shape

    tr_rows, tr_cols, te_rows, te_cols = [], [], [], []
    for u in range(n_u):
        items = list(R.rows[u])
        if not items:
            continue
        n = len(items)
        if n == 1:
            tr_rows.append(u); tr_cols.append(items[0])
            continue
        n_te = max(1, int(round(n * holdout_frac)))
        n_te = min(n_te, n - 1)
        perm = rng.permutation(n)
        for ti in perm[n_te:]:
            tr_rows.append(u); tr_cols.append(items[ti])
        for ti in perm[:n_te]:
            te_rows.append(u); te_cols.append(items[ti])

    def _build(rows, cols):
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(n_u, n_i))

    return _build(tr_rows, tr_cols), _build(te_rows, te_cols)


def summary_dict(data: dict) -> str:
    phi = data["Phi_V_warm"]
    density = phi.nnz / (phi.shape[0] * phi.shape[1]) * 100
    return (
        f"  Utilisateurs        : {data['R_train_warm'].shape[0]}\n"
        f"  Items warm          : {data['R_train_warm'].shape[1]}\n"
        f"  Items cold          : {data['R_test_cold'].shape[1]}\n"
        f"  Features contenu    : {phi.shape[1]} (sparse, densité {density:.3f} %)\n"
        f"  Interactions train  : {data['R_train_warm'].nnz}\n"
        f"  Interactions cold T : {data['R_test_cold'].nnz}"
    )
