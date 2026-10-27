# Jalon D — Gain de Kalman appris (KalmanNet)

[← README](../README.md)

> **Verdict : hypothèse rejetée.** Un petit KalmanNet sur modèle CV égale l'IMM en ligne droite et sur séquences longues,
> mais perd nettement en virage (43,6 m contre 33,5 m) et coûte plus cher. Son point fort est la robustesse aux manœuvres
> non prévues (virages forts : 44,7 m contre 64,3 m pour l'IMM), mais un gain fixe non appris y fait aussi bien.

## Hypothèse

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

Entraînement : 3 graines, 60 époques chacune, meilleure RMSE de validation 30,5 / 30,8 / 30,6 m.

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

## Résultats (jeu de test, 300 trajectoires par condition)

RMSE de position [m]. IC 95 % : bootstrap des trajectoires pour les références, Student sur les 3 graines pour KalmanNet.

| méthode | nominal : globale | nominal : en virage | nominal : ligne droite | virages forts | bruit ×2 | séquences ×4 |
|---|---|---|---|---|---|---|
| EKF-CV | 59,5 ± 5,7 | 99,4 ± 9,1 | 52,4 ± 5,3 | 106,3 ± 5,8 | 77,2 ± 5,8 | 33,3 ± 2,3 |
| IMM (CV + CT±) | **28,2 ± 0,9** | **33,5 ± 2,5** | **27,4 ± 0,9** | 64,3 ± 4,0 | **55,7 ± 1,8** | **23,0 ± 0,5** |
| gain fixe (non appris) | 35,1 ± 1,1 | 40,9 ± 2,5 | 34,3 ± 1,1 | **41,7 ± 1,1** | 65,9 ± 2,1 | 30,6 ± 0,9 |
| KalmanNet | 29,7 ± 0,2 | 43,6 ± 0,9 | **27,5 ± 0,2** | 44,7 ± 2,6 | 59,0 ± 1,7 | **23,1 ± 0,1** |

Détail virage / ligne droite pour chaque condition : [results/D_evaluation.md](../results/D_evaluation.md).

![RMSE globale par condition et par méthode](../figures/D_comparaison.png)

- **En virage, l'hypothèse ne tient pas** : 43,6 m contre 33,5 m pour l'IMM, et même moins bien que l'a priori non appris (40,9 m).
- **L'apprentissage paie en ligne droite** : 34,3 → 27,5 m par rapport au gain fixe, au niveau de l'IMM (27,4 m). Tout le gain
  global (35,1 → 29,7 m, −15 %) vient de là.
- Explication probable (non isolée par une expérience dédiée) : les virages ne représentent qu'environ 12 % des pas
  d'entraînement (une trajectoire sur deux, 20 à 50 pas sur 150), donc la perte est dominée par les lignes droites et le réseau
  apprend à lisser davantage, ce qui se paie en virage. Piste : pondérer les pas en virage dans la perte, ou sur-échantillonner
  les trajectoires qui virent.
- **Virages forts** : KalmanNet (44,7 m) bat largement l'IMM (64,3 m), dont les modèles CT (0,02 rad/s) sont désaccordés, mais
  reste derrière le gain fixe (41,7 m) : l'apprentissage a un peu dégradé la tenue en manœuvre par rapport à son a priori.
- **Bruit ×2** : 59,0 m, entre l'IMM (55,7 m) et le gain fixe (65,9 m).
- **Séquences ×4** : égalité avec l'IMM (23,1 contre 23,0 m), donc pas de dérive du GRU sur des séquences 4 fois plus longues
  que celles de l'entraînement (avec le biais de tirage décrit plus haut).
- **Graines** : les 3 graines sont très proches, d'où l'IC étroit de KalmanNet. Cet IC ne mesure que la variance d'entraînement,
  pas celle des trajectoires de test : il n'est pas comparable à celui des références. L'écart nominal avec l'IMM (1,5 m) dépasse
  l'IC de l'IMM (0,9 m), mais le test propre serait une comparaison appariée sur les mêmes trajectoires.

### Coût

| méthode | paramètres appris | temps par mise à jour [µs] |
|---|---|---|
| EKF-CV (numpy) | 0 | 40 |
| IMM (CV + CT±, numpy) | 0 | 324 |
| gain fixe (torch) | 0 | 433 |
| KalmanNet (torch) | 19 592 | 433 |

Le gain fixe, même structure sous torch sans apprentissage, coûte autant que KalmanNet : le surcoût vient de torch en batch de 1,
pas du réseau. Les temps absolus dépendent de la machine et de sa charge ; seuls les rapports sont à retenir. L'inférence d'un
réseau aussi petit devrait passer par du numpy ou du C++ (jalon G).

## Reproduire

Estimer la durée sans entraîner (4 lots chronométrés, aucun point de contrôle écrit) :

```bash
uv run python -m learned.train --estimate --device cpu
```

Entraîner les 3 graines en parallèle sur CPU (mesuré : ≈ 60 s par époque avec les 3 graines en parallèle, 60 époques, ≈ 1 h ;
le CPU est 2,7 fois plus rapide que CUDA pour ce petit GRU). Les poids entraînés ne sont pas versionnés :

```bash
mkdir -p checkpoints && for s in 0 1 2; do OMP_NUM_THREADS=1 uv run python -m learned.train --seed $s --device cpu --out-dir checkpoints > checkpoints/seed$s.log 2>&1 & done; wait
```

Évaluer (≈ 5 min) :

```bash
uv run python -m learned.evaluate --checkpoints checkpoints/kalmannet_seed{0,1,2}_best.pt --out results/D_evaluation.json --figure figures/D_comparaison.png
```

Sans `--checkpoints`, seules les références sont évaluées.

Tests : `uv run pytest tests/test_dataset.py tests/test_kalmannet.py tests/test_train.py tests/test_evaluate.py`.
