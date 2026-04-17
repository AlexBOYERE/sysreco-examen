"""DropoutNet — modèle de cold start pour systèmes de recommandation.

Référence : Volkovs, Yu, Poutanen, "DropoutNet: Addressing Cold Start in
Recommender Systems", NIPS 2017.

Architecture (cas CiteULike : seulement contenu item disponible)
----------------------------------------------------------------
  Branche utilisateur  : U_u           -> f_U  -> Û_u  (rang D)
  Branche item         : [V_v, Φ_v]    -> f_V  -> V̂_v  (rang D)
  Score                : ŝ_uv = Û_u · V̂_v^T

Objectif
--------
  L = E_{u,v} (U_u V_v^T - Û_u V̂_v^T)^2

avec, par mini-batch et avec probabilité τ, les items passent par l'une de :
  (a) dropout          : V_v -> 0
  (b) transform inférence : V_v -> mean_{u ∈ U(v)} U_u   (éq. 4 du papier)
On alterne (a) et (b) à chaque itération (comme dans Algorithm 1).

Implémentation
--------------
Phi_V est scipy CSR (densité ~0.1 %). À chaque mini-batch on slice les
lignes correspondantes aux items tirés et on densifie ce petit bloc avant
de le passer à PyTorch. Empreinte mémoire : ~30 MB par batch au lieu de
353 MB si Phi_V était conservée en dense.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, csc_matrix


class DropoutNet(nn.Module):
    """Deux branches (user, item), 1 couche cachée chacune, tanh + linéaire."""

    def __init__(
        self,
        d_latent: int = 200,
        n_item_content: int = 6493,
        d_hidden: int = 500,
        d_out: int | None = None,
    ):
        super().__init__()
        d_out = d_out or d_latent
        self.user_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )
        self.item_net = nn.Sequential(
            nn.Linear(d_latent + n_item_content, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )

    def encode_user(self, U: torch.Tensor) -> torch.Tensor:
        return self.user_net(U)

    def encode_item(self, V: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        return self.item_net(torch.cat([V, Phi], dim=-1))


def _mean_user_per_item(U: np.ndarray, R_csc: csc_matrix) -> np.ndarray:
    """Pour chaque item v, mean_{u ∈ U(v)} U_u. Items sans users -> 0."""
    n_i = R_csc.shape[1]
    d = U.shape[1]
    out = np.zeros((n_i, d), dtype=np.float32)
    for v in range(n_i):
        s, e = R_csc.indptr[v], R_csc.indptr[v + 1]
        users = R_csc.indices[s:e]
        if len(users) > 0:
            out[v] = U[users].mean(axis=0)
    return out


def _slice_dense(Phi_csr: csr_matrix, idx: np.ndarray) -> np.ndarray:
    """Slice et densifie un petit bloc de lignes. Retourne float32 dense."""
    return Phi_csr[idx].toarray().astype(np.float32, copy=False)


def train_dropoutnet(
    model: DropoutNet,
    U: np.ndarray,            # (n_users, D) float32
    V: np.ndarray,            # (n_warm_items, D) float32
    Phi_V: csr_matrix,        # (n_warm_items, n_content) sparse
    R_train,                  # csr (n_users, n_warm_items)
    tau: float = 0.5,
    n_iters: int = 2500,
    batch_user: int = 1000,
    batch_item: int = 1000,
    lr: float = 5e-3,
    momentum: float = 0.9,
    log_every: int = 200,
    seed: int = 0,
    device: str = "cpu",
):
    """Entraîne DropoutNet en alternant dropout et approximation d'inférence
    sur la branche item. Retourne l'historique des pertes (par itération).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model.to(device).train()
    # Adam est beaucoup plus robuste pour cet objectif MSE que SGD — les
    # magnitudes de U et V issues de WMF peuvent être très asymétriques
    # (||U||>>||V||) et SGD converge mal dans ce cas.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    U_t = torch.from_numpy(U).float().to(device)
    V_t = torch.from_numpy(V).float().to(device)

    # Approximation d'inférence : mean des U des utilisateurs ayant interagi
    R_csc = R_train.tocsc()
    mean_U_per_item = _mean_user_per_item(U, R_csc)
    mean_U_t = torch.from_numpy(mean_U_per_item).to(device)

    n_u, n_i = R_train.shape
    losses = []

    for it in range(n_iters):
        u_idx = rng.integers(0, n_u, size=batch_user)
        v_idx = rng.integers(0, n_i, size=batch_item)

        U_b = U_t[u_idx]
        V_b = V_t[v_idx]
        Phi_b = torch.from_numpy(_slice_dense(Phi_V, v_idx)).to(device)

        # Cible : scores WMF (détachés)
        with torch.no_grad():
            S_tgt = U_b @ V_b.T

        # Alternance dropout / inférence transform sur les items
        p = torch.rand(batch_item, device=device)
        mask = (p < tau).float().unsqueeze(1)
        if (it % 2) == 0:
            V_in = V_b * (1.0 - mask)
        else:
            V_in = V_b * (1.0 - mask) + mean_U_t[v_idx] * mask

        U_hat = model.encode_user(U_b)
        V_hat = model.encode_item(V_in, Phi_b)
        S_pred = U_hat @ V_hat.T

        loss = ((S_tgt - S_pred) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if log_every and ((it + 1) % log_every == 0):
            recent = float(np.mean(losses[-log_every:]))
            print(f"  [DN τ={tau:.2f}] iter {it+1}/{n_iters}  loss={recent:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Inférence : calcule Û et V̂ item par item (en batches) sans matérialiser
# la Phi_V dense en entier.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_user_embeddings(
    model: DropoutNet,
    U: np.ndarray,
    batch: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    model.eval().to(device)
    U_t = torch.from_numpy(U).float().to(device)
    out = []
    for i in range(0, U_t.shape[0], batch):
        out.append(model.encode_user(U_t[i : i + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def compute_item_embeddings(
    model: DropoutNet,
    V: np.ndarray,          # (n_items, D) préférences ; pour cold, passer des zéros
    Phi_V: csr_matrix,      # (n_items, n_content) sparse
    batch: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Calcule V̂ pour tous les items, en petits batches densifiés."""
    model.eval().to(device)
    n = V.shape[0]
    outs = []
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        V_b = torch.from_numpy(V[idx]).float().to(device)
        Phi_b = torch.from_numpy(_slice_dense(Phi_V, idx)).to(device)
        outs.append(model.encode_item(V_b, Phi_b).cpu().numpy())
    return np.concatenate(outs, axis=0)
