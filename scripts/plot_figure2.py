"""Génère le plot warm & cold Recall@100 vs taux de dropout τ
(équivalent Figure 2 du papier DropoutNet)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    results_path = ROOT / "results" / "results.json"
    with open(results_path) as f:
        r = json.load(f)

    dn = r["dropoutnet"]
    taus, warm, cold = [], [], []
    for key in sorted(dn.keys()):
        taus.append(dn[key]["tau"])
        warm.append(dn[key]["warm"]["100"])
        cold.append(dn[key]["cold"]["100"])

    wmf_warm100 = r["wmf"]["warm"]["100"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(taus, warm, marker="o", linewidth=2, label="DropoutNet — warm")
    ax.plot(taus, cold, marker="s", linewidth=2, label="DropoutNet — cold")
    ax.axhline(
        wmf_warm100, color="gray", linestyle="--", alpha=0.7,
        label=f"WMF warm (baseline) = {wmf_warm100:.3f}",
    )

    ax.set_xlabel("Taux de dropout τ", fontsize=11)
    ax.set_ylabel("Recall@100", fontsize=11)
    ax.set_title(
        f"DropoutNet sur CiteULike — rang {r['config']['wmf_rank']}, "
        f"{r['config']['dn_iters']} itérations",
        fontsize=11,
    )
    ax.set_ylim(0.0, max(max(warm), max(cold), wmf_warm100) * 1.15)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10)

    fig.tight_layout()
    out = ROOT / "results" / "figure2_reproduction.png"
    fig.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")

    # Affichage des chiffres
    print("\nRécap :")
    print(f"  WMF warm@100         : {wmf_warm100:.4f}")
    print(f"  {'τ':<6}{'warm@100':<12}{'cold@100':<12}")
    for t, w, c in zip(taus, warm, cold):
        print(f"  {t:<6.2f}{w:<12.4f}{c:<12.4f}")


if __name__ == "__main__":
    main()
