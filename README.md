# DropoutNet — Implémentation sur CiteULike

Reproduction du modèle **DropoutNet** (Volkovs, Yu & Poutanen, NIPS 2017)
appliqué au jeu de données **CiteULike-a** pour l'étude du cold start en
recommandation.

---

## Architecture du projet

```
sysreco-examen/
├── data/                                   Données préparées (fournies)
│   ├── R_train.npz                         Interactions warm (5551 × 13584, sparse)
│   ├── R_test.npz                          Interactions complètes (5551 × 16980)
│   ├── Phi_V.npy                           Features TF-IDF (16980 × 6493, float32)
│   ├── warm_start_items.npy                13584 indices items warm
│   ├── cold_start_items.npy                3396 indices items cold
│   └── preparation_donnees_citeulike.ipynb Notebook de préparation (référence)
│
├── models/
│   ├── wmf.py                              WMF-ALS (Hu, Koren, Volinsky 2008)
│   └── dropoutnet.py                       Architecture PyTorch + boucle d'entraînement
│
├── utils/
│   ├── data.py                             Chargement + hold-out warm 20 %
│   └── eval.py                             Recall@K batché (warm et cold)
│
├── scripts/
│   ├── run_experiment.py                   Orchestrateur complet (WMF + tous les τ)
│   ├── train_one_tau.py                    Entraîne UN τ isolé
│   └── plot_figure2.py                     Reproduit la Figure 2 du papier
│
├── results/                                Sorties générées par les scripts
│   ├── U_wmf.npy, V_wmf.npy                Latents WMF sauvegardés
│   ├── results.json                        Toutes les métriques en JSON
│   └── figure2_reproduction.png            Courbe warm/cold vs τ
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prérequis

- Python 3.10 ou plus récent (testé sur 3.12)
- `python3-venv` et `python3-full` sur Ubuntu/Debian :
  ```bash
  sudo apt install python3-full python3-venv
  ```
- Environ 3 GB de RAM libre pendant l'exécution

### Installation des dépendances

Depuis la racine du projet, en quatre commandes :

```bash
cd sysreco-examen                        # racine du projet
python3 -m venv .venv                    # crée un environnement isolé
source .venv/bin/activate                # l'active (prompt préfixé par (.venv))
pip install -r requirements.txt          # installe numpy, scipy, torch, matplotlib
```

L'installation prend ~2-3 min (PyTorch CPU fait ~200 MB). Le GPU n'est pas
requis.

### Vérification

```bash
python3 -c "import torch, numpy, scipy; print('OK')"
```

---

## Exécution

Toutes les commandes sont à lancer **depuis la racine du projet**, avec le
venv activé (`source .venv/bin/activate`).

### 1) Test rapide (~3 min)

Valide le pipeline de bout en bout avec des hyper-paramètres réduits :

```bash
python3 scripts/run_experiment.py --quick
```

Attendez-vous à :
- WMF rang 64, 3 itérations → warm@100 ≈ 0.46
- DropoutNet τ=0 → reproduit WMF
- DropoutNet τ=0.5 → cold@100 ≈ 0.40

### 2) Expérience complète (~10 min)

Reproduit la Figure 2 du papier avec 5 points de dropout :

```bash
python3 scripts/run_experiment.py \
    --wmf-rank 128 --wmf-iter 6 \
    --dn-iters 700 --dn-lr 1e-3 \
    --taus 0.0,0.1,0.3,0.5,0.7
```

Temps observés sur CPU 8 cœurs (Yoga Slim 7 14IMH9) :
- WMF (6 itérations) : ~3-4 min
- DropoutNet (700 itérations) : ~70 s × 5 τ = ~6 min
- **Total : ~9-10 min**

### 3) Générer la figure

Après un run, trace la courbe Recall@100 warm/cold en fonction de τ :

```bash
python3 scripts/plot_figure2.py
```

Le fichier `results/figure2_reproduction.png` est écrit.

### 4) Entraîner un seul τ supplémentaire

Si vous voulez ajouter un point de dropout sans refaire tout le pipeline
(par exemple τ=0.9 pour tester la dégradation à fort dropout) :

```bash
python3 scripts/train_one_tau.py --tau 0.9 --dn-iters 700
python3 scripts/plot_figure2.py           # regénère la figure
```

Le script recharge les latents WMF depuis `results/U_wmf.npy` et
`results/V_wmf.npy`, donc ne relance pas WMF.

---

## Paramètres disponibles

`run_experiment.py` accepte les options suivantes :

| Option | Défaut | Rôle |
|---|---|---|
| `--wmf-rank` | 200 | Dimension latente D |
| `--wmf-iter` | 8 | Nombre d'itérations ALS |
| `--wmf-alpha` | 40.0 | Paramètre de confiance WMF |
| `--wmf-reg` | 0.01 | Régularisation L2 pour WMF |
| `--dn-iters` | 1500 | Itérations d'entraînement DropoutNet |
| `--dn-hidden` | 500 | Taille de la couche cachée |
| `--dn-lr` | 1e-3 | Learning rate d'Adam |
| `--dn-batch-user` | 1000 | Taille du batch utilisateur |
| `--dn-batch-item` | 1000 | Taille du batch item |
| `--taus` | "0.0,0.3,0.5,0.7" | Taux de dropout à tester (séparés par virgules) |
| `--seed` | 42 | Seed pour la reproductibilité |
| `--quick` | off | Mode rapide (ignore les autres paramètres) |

---

## Résultats obtenus

Configuration : rang 128, 6 itérations WMF, 700 itérations DropoutNet,
batch 1000×1000, Adam lr=1e-3.

| Modèle | warm R@100 | cold R@100 |
|---|---:|---:|
| WMF (baseline) | **0.5262** | — |
| DN τ=0.00 | 0.5262 | 0.0759 |
| DN τ=0.10 | 0.4393 | 0.4317 |
| DN τ=0.30 | 0.4378 | 0.4360 |
| DN τ=0.50 | 0.4377 | 0.4405 |
| DN τ=0.70 | 0.4381 | 0.4424 |

Observations :
- DN τ=0 reproduit WMF à 4 décimales près → validation du pipeline.
- Entre τ=0 et τ=0.1, le cold start passe de 0.076 à 0.432 (×5.7) tandis
  que le warm ne perd que 17 %.
- Pour τ ∈ [0.1, 0.7], warm et cold sont quasi stables → le modèle a
  trouvé son équilibre.
- À τ=0.7, le cold dépasse le warm : le contenu porte plus d'information
  utile pour le ranking que les préférences dans ce régime.

---

## Choix d'implémentation importants

### Phi_V en sparse CSR

La matrice TF-IDF a une densité de 0.086 % (5-6 mots non-nuls par document).
Stockée en dense float32 elle occupe 441 MB ; en CSR elle tient dans moins
de 1 MB. Les mini-batches sont densifiés à la volée (~25 MB par batch).

### Adam plutôt que SGD

Les latents WMF présentent une forte asymétrie d'amplitude (‖U‖ ≈ 8,
‖V‖ ≈ 0.1). SGD converge très mal dans ce régime. Adam avec lr=1e-3 est
robuste et permet à DN τ=0 de reproduire WMF à 4 décimales en 700
itérations.

### Hold-out warm 20 %

Dans les données fournies, la partie warm de `R_test` est strictement
identique à `R_train`. Pour obtenir une évaluation warm-start
méthodologiquement correcte, 20 % des interactions par utilisateur sont
retirées de `R_train` pour former un test warm-start distinct. Les items
cold restent inchangés.

### Objectif (équation 2 du papier)

```
L = E_{u,v} ( U_u · V_v^T  −  f_U(U_u) · f_V(V_v, Φ_v)^T )²
```

Avec probabilité τ par item, le vecteur V_v passe par l'une des
transformations :
- **dropout** : `V_v → 0`
- **approximation d'inférence** : `V_v → mean_{u ∈ U(v)} U_u` (éq. 4)

On alterne les deux transformations à chaque mini-batch (Algorithm 1 du
papier).

---

## Dépannage

### `ModuleNotFoundError: No module named 'torch'`

Le venv n'est pas activé. Lancez `source .venv/bin/activate` depuis la
racine du projet.

### `FileNotFoundError: data/R_train.npz`

Vous lancez le script depuis un mauvais dossier. Les chemins sont relatifs
à la racine du projet :

```bash
cd sysreco-examen                        # toujours démarrer ici
python3 scripts/run_experiment.py ...
```

### `error: externally-managed-environment` lors du `pip install`

Ubuntu 24+ interdit l'installation dans le Python système (PEP 668). Passez
toujours par un venv :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### MemoryError ou OOM pendant l'entraînement

Si la machine a moins de 4 GB de RAM libre, réduisez les batchs :

```bash
python3 scripts/run_experiment.py \
    --dn-batch-user 500 --dn-batch-item 500 \
    --wmf-rank 64 --dn-iters 500
```

### Le script ne trouve pas `Phi_V.npy`

Le fichier fait 441 MB et peut être manquant si l'archive n'a pas été
extraite au bon endroit. Vérifiez :

```bash
ls -lh data/Phi_V.npy
```

Si absent, ré-extrayez-le depuis `citeulike_data_prepared.zip`.

---

## Lancement via PyCharm (optionnel)

Si vous préférez l'IDE :

1. **File → Settings → Project → Python Interpreter → Add → Existing →
   `.venv/bin/python`**
2. **Run → Edit Configurations → +** → **Python** :
   - Script : `scripts/run_experiment.py`
   - Parameters : `--wmf-rank 128 --wmf-iter 6 --dn-iters 700 --taus 0.0,0.1,0.3,0.5,0.7`
   - Working directory : racine du projet (`sysreco-examen`)
   - Environment variables : `PYTHONUNBUFFERED=1;PYTHONIOENCODING=utf-8`
3. Bouton Play.

---

## Référence

Volkovs M., Yu G., Poutanen T. (2017). *DropoutNet: Addressing Cold Start
in Recommender Systems*. 31st Conference on Neural Information Processing
Systems (NIPS 2017), Long Beach, CA, USA.