"""Entraîne DropoutNet pour UN seul τ, en rechargeant U,V WMF déjà
sauvegardés par run_experiment.py. Ajoute le résultat à results.json.

Permet d'enchaîner les τ un par un quand on est limité en temps
(ex: environnement avec timeout court).

Usage :
  python3 scripts/train_one_tau.py --tau 0.3 [--dn-iters 700]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import load_citeulike, make_warm_holdout
from utils.eval import recall_at_k_multi
from models.dropoutnet import (
    DropoutNet,
    train_dropoutnet,
    compute_user_embeddings,
    compute_item_embeddings,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dn-iters", type=int, default=700)
    ap.add_argument("--dn-hidden", type=int, default=500)
    ap.add_argument("--dn-lr", type=float, default=1e-3)
    ap.add_argument("--batch-user", type=int, default=1000)
    ap.add_argument("--batch-item", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    # Charge les résultats existants (WMF + autres τ déjà faits)
    results_path = results_dir / "results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    U = np.load(results_dir / "U_wmf.npy")
    V = np.load(results_dir / "V_wmf.npy")
    D = U.shape[1]

    data = load_citeulike(args.data_dir)
    R_train_reduced, R_warm_test = make_warm_holdout(
        data["R_train_warm"], holdout_frac=0.2, seed=args.seed
    )
    R_test_cold = data["R_test_cold"]

    print(f"Entraînement DropoutNet τ={args.tau:.2f}  (rang={D}, iters={args.dn_iters})")
    model = DropoutNet(
        d_latent=D,
        n_item_content=data["Phi_V_warm"].shape[1],
        d_hidden=args.dn_hidden,
        d_out=D,
    )

    t0 = time.time()
    losses = train_dropoutnet(
        model, U=U, V=V,
        Phi_V=data["Phi_V_warm"],
        R_train=R_train_reduced,
        tau=args.tau,
        n_iters=args.dn_iters,
        batch_user=args.batch_user, batch_item=args.batch_item,
        lr=args.dn_lr, momentum=0.9,
        log_every=max(args.dn_iters // 7, 50),
        seed=args.seed,
    )
    dn_time = time.time() - t0
    print(f"  Entraîné en {dn_time:.1f}s   final_loss={np.mean(losses[-50:]):.4f}")

    U_hat = compute_user_embeddings(model, U)
    V_hat_warm = compute_item_embeddings(model, V, data["Phi_V_warm"])
    V_cold_zeros = np.zeros((data["Phi_V_cold"].shape[0], D), dtype=np.float32)
    V_hat_cold = compute_item_embeddings(model, V_cold_zeros, data["Phi_V_cold"])
    del model; gc.collect()

    ks = (50, 100, 200, 300, 500)
    warm = recall_at_k_multi(U_hat, V_hat_warm, R_warm_test, exclude=R_train_reduced, ks=ks)
    cold = recall_at_k_multi(U_hat, V_hat_cold, R_test_cold, exclude=None, ks=ks)
    print(f"  warm  " + "  ".join(f"R@{k}:{v:.4f}" for k, v in sorted(warm.items())))
    print(f"  cold  " + "  ".join(f"R@{k}:{v:.4f}" for k, v in sorted(cold.items())))

    all_results.setdefault("dropoutnet", {})[f"tau_{args.tau:.2f}"] = {
        "tau": args.tau,
        "time_s": dn_time,
        "n_iters": args.dn_iters,
        "final_loss": float(np.mean(losses[-50:])),
        "loss_history": [float(x) for x in losses[:: max(1, args.dn_iters // 200)]],
        "warm": warm, "cold": cold,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Résultats mis à jour : {results_path}")


if __name__ == "__main__":
    main()
