# Jalon D — Gain de Kalman appris (KalmanNet)

[← README](../README.md)

> **État : code et tests prêts, entraînement des 3 graines à lancer.** Les résultats de KalmanNet manquent encore ;
> les chiffres ci-dessous sont ceux des **références** à battre, mesurés sur le jeu de test.

## Hypothèse (ROADMAP, option D1)

*KalmanNet ≥ IMM en virage avec un modèle CV seul, à coût de calcul raisonnable.* Un résultat négatif est valable :
« l'IMM reste meilleur sauf dans tel régime ».

## Montage

KalmanNet (Revach et al., IEEE TSP 2022) garde la prédiction et la mise à jour de Kalman (modèle CV seul) et remplace uniquement le
calcul du gain K par un petit GRU. Code : `src/learned/` (`dataset.py`, `kalmannet.py`, `train.py`, `evaluate.py`).

- **Réseau** : projection linéaire (7 → 32), GRU de 64 unités, tête linéaire (→ gain 4×2) : **19 592 paramètres** (< 50 000).
- **Entrées** : innovation normalisée (2), correction précédente dans le repère de ligne de visée (4), distance prédite (1).
- **Données** : une cible à la fois, 150 balayages, mesures polaires, 4 000 trajectoires d'entraînement, 300 de validation, 300 de test,
  flux aléatoires indépendants ; 50 % des trajectoires comportent un virage (0,015–0,03 rad/s).
- **Perte** : erreur d'état x̂_{k|k} normalisée (50 m, 1 m/s), rétropropagée à travers les pas de filtre.

Écarts assumés par rapport à l'article, pour économiser des données avec un modèle minuscule :

- **Repère de ligne de visée** : le gain est exprimé le long du rayon et en travers, puis ramené en cartésien par la rotation du gisement.
  Le réseau est équivariant par rotation autour du radar (test exact en double précision).
- **Gain borné et a priori alpha-bêta** (a = 0,2, b = 0,03, choisis sur la validation) : K̃ = bas + (haut − bas)·sigmoïde(sortie + logit(a priori)).
  Sans borne, une perturbation de 0,03 sur les poids de la tête suffit à faire diverger l'état (10¹⁴ m), donc à produire des NaN.
- **Pas de covariance** : KalmanNet n'en donne pas, donc ni gating de Mahalanobis ni NEES avec lui.

## Protocole de comparaison

`evaluate.py` compare, sur les mêmes mesures et la même initialisation (première mesure) :

| méthode | rôle |
|---|---|
| EKF-CV | référence classique, Q et R nominaux (jalon A) |
| IMM (CV + CT±) | référence à battre (jalon C) |
| gain fixe (non appris) | KalmanNet sans apprentissage : isole ce que l'apprentissage apporte |
| KalmanNet | 3 graines d'entraînement, IC de Student sur les graines |

Quatre conditions de test ; tous les filtres gardent leurs réglages nominaux dans les trois dernières :
**nominal** (distribution d'entraînement), **virages 3 à 4 fois plus forts** (0,06–0,10 rad/s), **bruit de mesure ×2**,
**séquences 4 fois plus longues**. C'est la section « quand ça ne marche pas ».

Biais de la condition « séquences ×4 » : la zone de 20 km est la même, et le tirage par rejet (le navire doit y rester toute
la séquence) écarte surtout les navires rapides. Vitesse initiale moyenne 8,9 m/s au lieu de 10,0, part au-dessus de 12 m/s
divisée par 2, distance moyenne au radar 6,4 km au lieu de 7,2. Les méthodes restent comparables entre elles sur ce jeu, mais
sa RMSE plus faible qu'en nominal ne vient pas seulement de la durée : les cibles y sont plus lentes et plus proches (bruit
transverse r·σ_θ plus faible). Agrandir la zone ne corrige pas le biais de vitesse (9,8 m/s même à ×4) et éloigne les cibles.

## Références à battre (jeu de test, 300 trajectoires, IC 95 % par bootstrap)

RMSE de position [m] :

| méthode | nominal : globale | nominal : en virage | virages forts | bruit ×2 | séquences ×4 |
|---|---|---|---|---|---|
| EKF-CV | 59,5 ± 5,7 | 99,4 ± 9,1 | 106,3 ± 5,9 | 77,2 ± 5,7 | 33,3 ± 2,2 |
| **IMM (CV + CT±)** | **28,2 ± 0,9** | **33,5 ± 2,5** | 64,3 ± 4,1 | **55,7 ± 1,8** | **23,0 ± 0,5** |
| gain fixe (non appris) | 35,1 ± 1,1 | 40,9 ± 2,5 | **41,7 ± 1,1** | 65,9 ± 2,3 | 30,6 ± 0,8 |

Constats déjà utiles, avant tout entraînement :

- Sur la distribution d'entraînement, l'IMM est la référence à battre (28,2 m) ; un gain constant fait déjà 35,1 m.
- Avec des virages 3 à 4 fois plus forts, le **gain fixe bat l'IMM** (41,7 m contre 64,3 m) : les modèles CT de l'IMM (0,02 rad/s) sont désaccordés.
- Coût : un pas de filtre sous torch en batch de 1 coûte ≈ 676 µs (mesuré sur le gain fixe, même structure), contre ≈ 528 µs pour l'IMM
  et ≈ 78 µs pour l'EKF-CV en numpy. L'inférence d'un réseau aussi petit devra passer par du numpy ou du C++ (jalon G).

## Résultats de KalmanNet

À compléter après l'entraînement et l'évaluation (tableaux, figure `figures/D_comparaison.png`, section « quand ça ne marche pas »).

## Reproduire

Estimer la durée sans entraîner (4 lots chronométrés, aucun point de contrôle écrit) :

```bash
uv run python -m learned.train --estimate --device cpu
```

Entraîner les 3 graines en parallèle sur CPU (≈ 29 à 33 s par époque, 60 époques au plus, ≤ 30 min par graine ; le CPU est 2,7 fois plus
rapide que CUDA pour ce petit GRU) :

```bash
mkdir -p checkpoints && for s in 0 1 2; do OMP_NUM_THREADS=1 uv run python -m learned.train --seed $s --device cpu --out-dir checkpoints > checkpoints/seed$s.log 2>&1 & done; wait
```

Évaluer (≈ 5 min) :

```bash
uv run python -m learned.evaluate --checkpoints checkpoints/kalmannet_seed{0,1,2}_best.pt --out results/D_evaluation.json --figure figures/D_comparaison.png
```

Sans `--checkpoints`, seules les références sont évaluées (c'est ainsi que le tableau ci-dessus a été obtenu).

Tests : `uv run pytest tests/test_dataset.py tests/test_kalmannet.py tests/test_train.py tests/test_evaluate.py`.
