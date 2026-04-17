"""Orchestrateur des expériences DropoutNet sur CiteULike.

Pipeline :
  1. Charge les données et crée un hold-out warm-start (20 %)
  2. Entraîne WMF (ALS) sur le train réduit -> U, V
  3. Évalue WMF (warm seulement — WMF ne gère pas le cold)
  4. Entraîne DropoutNet pour différents τ (taux de dropout)
  5. Évalue chaque DropoutNet en warm et cold start
  6. Sauvegarde les résultats dans results/results.json

Usage :
  python3 scripts/run_experiment.py [--quick]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import load_citeulike, make_warm_holdout, summary_dict
from utils.eval import recall_at_k_multi
from models.wmf import wmf_als
from models.dropoutnet import (
    DropoutNet,
    train_dropoutnet,
    compute_user_embeddings,
    compute_item_embeddings,
)


def eval_warm_cold(
    U_hat, V_hat_warm, V_hat_cold,
    R_train_reduced, R_warm_test, R_test_cold,
    ks=(50, 100, 200, 300, 500),
) -> dict:
    warm = recall_at_k_multi(
        U_hat, V_hat_warm, R_warm_test, exclude=R_train_reduced, ks=ks
    )
    cold = recall_at_k_multi(
        U_hat, V_hat_cold, R_test_cold, exclude=None, ks=ks
    )
    return {"warm": warm, "cold": cold}


def format_eval(name: str, ev: dict) -> str:
    lines = [f"  {name}"]
    for split in ("warm", "cold"):
        parts = [f"R@{k}:{ev[split][k]:.4f}" for k in sorted(ev[split].keys())]
        lines.append(f"    {split:4s}  " + "  ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wmf-rank", type=int, default=200)
    ap.add_argument("--wmf-iter", type=int, default=8)
    ap.add_argument("--wmf-alpha", type=float, default=40.0)
    ap.add_argument("--wmf-reg", type=float, default=0.01)
    ap.add_argument("--dn-iters", type=int, default=1500)
    ap.add_argument("--dn-hidden", type=int, default=500)
    ap.add_argument("--dn-lr", type=float, default=1e-3)
    ap.add_argument("--dn-batch-user", type=int, default=1000)
    ap.add_argument("--dn-batch-item", type=int, default=1000)
    ap.add_argument(
        "--taus", type=str, default="0.0,0.3,0.5,0.7",
        help="taux de dropout à tester (séparés par virgules)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        args.wmf_rank = 64
        args.wmf_iter = 3
        args.dn_iters = 400
        args.taus = "0.0,0.5"

    taus = [float(x) for x in args.taus.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Données -----
    print("=" * 70)
    print("1. Chargement des données")
    print("=" * 70)
    data = load_citeulike(args.data_dir)
    print(summary_dict(data))

    R_train_reduced, R_warm_test = make_warm_holdout(
        data["R_train_warm"], holdout_frac=0.2, seed=args.seed
    )
    print(
        f"\nHold-out warm 20 % :"
        f"\n  R_train réduit : {R_train_reduced.nnz}"
        f"\n  R_warm_test    : {R_warm_test.nnz}"
        f"\n  R_test_cold    : {data['R_test_cold'].nnz}"
    )

    # ----- 2. WMF -----
    print("\n" + "=" * 70)
    print(f"2. WMF-ALS  (rang={args.wmf_rank}, α={args.wmf_alpha}, "
          f"λ={args.wmf_reg}, iter={args.wmf_iter})")
    print("=" * 70)
    t0 = time.time()
    U, V = wmf_als(
        R_train_reduced,
        rank=args.wmf_rank,
        reg=args.wmf_reg,
        alpha=args.wmf_alpha,
        n_iter=args.wmf_iter,
        seed=args.seed,
    )
    wmf_time = time.time() - t0
    print(f"  WMF entraîné en {wmf_time:.1f}s")

    np.save(results_dir / "U_wmf.npy", U)
    np.save(results_dir / "V_wmf.npy", V)

    # ----- 3. Évaluation WMF -----
    print("\n3. Évaluation WMF (warm seulement)")
    wmf_warm = recall_at_k_multi(
        U, V, R_warm_test, exclude=R_train_reduced, ks=(50, 100, 200, 300, 500)
    )
    print("  WMF warm : " + "  ".join(
        f"R@{k}:{v:.4f}" for k, v in sorted(wmf_warm.items())))
    gc.collect()

    # ----- 4. DropoutNet -----
    all_results = {
        "config": vars(args),
        "data_stats": {
            "n_users": int(data["R_train_warm"].shape[0]),
            "n_warm": int(data["R_train_warm"].shape[1]),
            "n_cold": int(data["R_test_cold"].shape[1]),
            "n_features": int(data["Phi_V_warm"].shape[1]),
            "phi_density_pct": float(
                data["Phi_V_warm"].nnz
                / (data["Phi_V_warm"].shape[0] * data["Phi_V_warm"].shape[1])
                * 100
            ),
            "n_train_interactions": int(R_train_reduced.nnz),
            "n_warm_test_interactions": int(R_warm_test.nnz),
            "n_cold_test_interactions": int(data["R_test_cold"].nnz),
        },
        "wmf": {"time_s": wmf_time, "warm": wmf_warm},
        "dropoutnet": {},
    }

    n_feat = data["Phi_V_warm"].shape[1]
    V_cold_zeros = np.zeros(
        (data["Phi_V_cold"].shape[0], args.wmf_rank), dtype=np.float32
    )

    for tau in taus:
        print("\n" + "=" * 70)
        print(f"4. DropoutNet  τ={tau:.2f}  "
              f"(iters={args.dn_iters}, hidden={args.dn_hidden})")
        print("=" * 70)
        model = DropoutNet(
            d_latent=args.wmf_rank,
            n_item_content=n_feat,
            d_hidden=args.dn_hidden,
            d_out=args.wmf_rank,
        )
        t0 = time.time()
        losses = train_dropoutnet(
            model,
            U=U, V=V,
            Phi_V=data["Phi_V_warm"],
            R_train=R_train_reduced,
            tau=tau,
            n_iters=args.dn_iters,
            batch_user=args.dn_batch_user,
            batch_item=args.dn_batch_item,
            lr=args.dn_lr,
            momentum=0.9,
            log_every=max(args.dn_iters // 10, 50),
            seed=args.seed,
        )
        dn_time = time.time() - t0
        print(f"  Entraîné en {dn_time:.1f}s  final_loss={np.mean(losses[-50:]):.4f}")

        U_hat = compute_user_embeddings(model, U)
        V_hat_warm = compute_item_embeddings(model, V, data["Phi_V_warm"])
        V_hat_cold = compute_item_embeddings(model, V_cold_zeros, data["Phi_V_cold"])

        # Libérer le modèle tant qu'on n'en a plus besoin pour l'éval
        del model
        gc.collect()

        ev = eval_warm_cold(
            U_hat, V_hat_warm, V_hat_cold,
            R_train_reduced, R_warm_test, data["R_test_cold"],
        )
        print(format_eval(f"DropoutNet τ={tau:.2f}", ev))

        all_results["dropoutnet"][f"tau_{tau:.2f}"] = {
            "tau": tau,
            "time_s": dn_time,
            "final_loss": float(np.mean(losses[-50:])),
            "loss_history": [float(x) for x in losses[:: max(1, args.dn_iters // 200)]],
            "warm": ev["warm"],
            "cold": ev["cold"],
        }

        with open(results_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        del U_hat, V_hat_warm, V_hat_cold
        gc.collect()

    # ----- 5. Récap -----
    print("\n" + "=" * 70)
    print("5. Récapitulatif")
    print("=" * 70)
    print(f"  WMF warm recall@100 : {wmf_warm[100]:.4f}")
    for tau_key, dn in all_results["dropoutnet"].items():
        print(
            f"  DN {tau_key}   warm@100 {dn['warm'][100]:.4f}   "
            f"cold@100 {dn['cold'][100]:.4f}"
        )
    print(f"\nRésultats : {results_dir}/results.json")


if __name__ == "__main__":
    main()
