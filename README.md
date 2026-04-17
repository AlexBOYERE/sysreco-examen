# DropoutNet — Implémentation PyTorch sur CiteULike

Reproduction du modèle **DropoutNet** (Volkovs, Yu & Poutanen, NIPS 2017) pour
le cold start en recommandation, appliqué au dataset **CiteULike-a**.

## Contenu du projet

```
dropoutnet/
├── data/                            # données préparées (pas dans le repo git si >100 MB)
│   ├── R_train.npz                  # (n_users × n_warm) interactions warm
│   ├── R_test.npz                   # (n_users × n_total) interactions complètes
│   ├── Phi_V.npy                    # (n_total × n_feat) features TF-IDF
│   ├── warm_start_items.npy         # indices items warm
│   └── cold_start_items.npy         # indices items cold (tenus à l'écart)
├── models/
│   ├── wmf.py                       # WMF-ALS (Hu-Koren-Volinsky 2008)
│   └── dropoutnet.py                # DropoutNet PyTorch + boucle d'entraînement
├── utils/
│   ├── data.py                      # chargement + hold-out warm 20 %
│   └── eval.py                      # Recall@K batché (warm / cold)
├── scripts/
│   ├── run_experiment.py            # orchestrateur complet (WMF + tous les τ)
│   ├── train_one_tau.py             # entraîne UN τ (utile si timeout court)
│   └── plot_figure2.py              # reproduit Figure 2 du papier
├── results/                         # sorties (U_wmf.npy, V_wmf.npy, results.json, figure2_reproduction.png)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Le projet tourne en CPU (PyTorch CPU suffit). GPU non requis.

## Exécution

### Pipeline complet en une commande

```bash
python3 scripts/run_experiment.py --wmf-rank 200 --wmf-iter 8 \
                                  --dn-iters 1500 --dn-lr 1e-3 \
                                  --taus "0.0,0.1,0.3,0.5,0.7"
```

Ordre des étapes (automatique) :
1. Chargement des données + création d'un hold-out warm-start de 20 %
2. Entraînement WMF (ALS implicite)
3. Évaluation WMF (warm uniquement — WMF ne gère pas le cold)
4. Entraînement DropoutNet pour chaque τ (avec Adam, LR=1e-3)
5. Évaluation de chaque modèle en warm et cold start

Durée approximative sur CPU 2 cœurs : ~30 min avec les hyper-paramètres ci-dessus.

### Exécution pas à pas (utile en environnement contraint)

Si votre environnement a une limite de durée par job, lancez la partie WMF
une seule fois puis enchaînez les τ un par un :

```bash
# 1) WMF + τ=0 (l'orchestrateur sauvegarde U_wmf.npy, V_wmf.npy, results.json)
python3 scripts/run_experiment.py --taus "0.0" --dn-iters 700

# 2) Les autres τ (rechargent les latents WMF déjà sauvegardés)
python3 scripts/train_one_tau.py --tau 0.1 --dn-iters 700
python3 scripts/train_one_tau.py --tau 0.3 --dn-iters 700
python3 scripts/train_one_tau.py --tau 0.5 --dn-iters 700
python3 scripts/train_one_tau.py --tau 0.7 --dn-iters 700

# 3) Tracer la Figure 2
python3 scripts/plot_figure2.py
```

### Mode rapide (test de bout en bout en ~5 min)

```bash
python3 scripts/run_experiment.py --quick
```

## Choix d'implémentation

### Pourquoi Adam plutôt que SGD ?

Les latents WMF présentent une asymétrie d'amplitude (‖U‖≈8, ‖V‖≈0.1) qui
rend la convergence SGD très lente : avec lr=5e-3 sur 400 itérations,
DropoutNet à τ=0 plafonne à R@100=0.01 (effondrement vers 0). Avec Adam
lr=1e-3 sur 700 itérations, DN à τ=0 reproduit WMF à ±0.001.

### Pourquoi Phi_V en sparse CSR ?

La matrice TF-IDF fournie a une densité de 0.086 % (5-6 mots non-nuls par
document en moyenne). Stockée en dense float32 elle occupe 441 MB ; en CSR
elle tient dans moins de 1 MB. Les mini-batches sont densifiés à la volée
(batch_item × 6493 float32 ≈ 25 MB par batch).

### Pourquoi créer un hold-out warm-start ?

Dans les données fournies, la partie warm de `R_test` est strictement
identique à `R_train` (aucune interaction warm tenue à l'écart). Évaluer
warm-start sur ces données reviendrait à mesurer la capacité du modèle à
reconstruire son training set — ce qui n'est pas ce que mesure le papier.

On retire donc 20 % des interactions par utilisateur de `R_train` pour
former un test warm-start proprement tenu à l'écart. Les 3 396 items cold
restent inchangés.

### Formulation DropoutNet

Objectif (équation 2 du papier, MSE sur les scores) :
```
L = E_{u,v} ( U_u · V_v^T  -  f_U(U_u) · f_V(V_v, Φ_v)^T )^2
```

Avec probabilité τ par item dans le mini-batch, le vecteur de préférence V_v
passe par l'une des deux transformations :
- **dropout** : `V_v → 0`
- **approximation d'inférence** (équation 4) : `V_v → mean_{u ∈ U(v)} U_u`

On alterne dropout (itérations paires) et approximation (impaires), comme
décrit dans l'Algorithm 1 du papier.

## Résultats obtenus

Avec rang 128, 6 itérations WMF, 700 itérations DropoutNet et batch
1000×1000 :

| Modèle        | warm R@100 | cold R@100 |
|---------------|-----------:|-----------:|
| WMF           | 0.5262     | —          |
| DN τ=0.00     | 0.5264     | 0.0760     |
| DN τ=0.10     | 0.4388     | 0.4322     |
| DN τ=0.30     | 0.4382     | 0.4360     |
| DN τ=0.50     | 0.4379     | 0.4393     |
| DN τ=0.70     | 0.4368     | 0.4426     |

Voir `results/figure2_reproduction.png` pour le tracé visuel.

Observations :
- À τ=0, DropoutNet reproduit exactement WMF (**0.5264 ≈ 0.5262**) — le
  réseau apprend bien la fonction identité sur la branche préférence.
- Dès τ=0.1, le cold start passe de 0.076 à **0.432** (×5.7) pendant que
  le warm start ne chute que de 0.526 à 0.439 (-17 %).
- Pour τ ∈ [0.1, 0.7], warm et cold sont quasi stables : le réseau a trouvé
  un équilibre entre reconstruction et généralisation au cold.

Ce comportement reproduit qualitativement la Figure 2 du papier
(dropout améliore massivement le cold start avec un coût limité sur le warm).
