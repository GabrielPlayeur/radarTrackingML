# Jalon A — Kalman / EKF, une cible

[← README](../README.md) · Notebook : [A_ekf.ipynb](../notebooks/A_ekf.ipynb)

## Objectif

Coder à la main un filtre de Kalman linéaire puis un EKF à mesures polaires (r, θ) pour une cible, vérifier qu'ils sont
**honnêtes** sur leur incertitude (NEES / NIS) et montrer quantitativement comment un modèle de mouvement faux les fait décrocher.

## Montage

- État `[x, y, vx, vy]`, pas de temps dt = 2 s (rotation d'antenne d'un radar marin), capteur à l'origine.
- Modèle CV, bruit d'accélération σ_a = 0,05 m/s² : Q = σ_a²·G·Gᵀ (accélération blanche discrète, matrice de rang 2).
- Mesures polaires : σ_r = 25 m, σ_θ = 0,01 rad ; innovation d'angle normalisée dans [−π, π].
- Mise à jour sous forme de Joseph. Code : `src/tracking/kf.py`, `ekf.py`, `src/sim/trajectories.py`, `src/metrics/nees.py`, `rmse.py`.
- Scénario des figures : navire à 15 m/s, ligne droite, virage à 0,02 rad/s (rayon 750 m, accélération latérale 0,3 m/s², 6 fois le σ_a nominal), ligne droite.

## Validations (tests dans `tests/`)

| Vérification | Résultat |
|---|---|
| KF vs `filterpy.kalman.KalmanFilter` (x, P, innovation, S) | identiques à 1e-9 |
| RMSE du filtre vs RMSE des mesures (CV, Q et R corrects) | 13 m contre 42 m |
| Jacobienne analytique de h vs différences finies | accord à 1e-6 (rel.) |
| Cible traversant θ = ±π | innovation max 0,03 rad avec normalisation (6,28 rad sans) ; saut d'estimation de 2 955 m sans normalisation |
| NEES moyen, 100 Monte-Carlo, CV, Q correct | EKF : 3,96 pour 4 attendu, IC χ² 95 % : [3,46 ; 4,57] ; le KF est aussi dans l'IC (test) |
| NIS moyen, même scénario | 1,95 pour 2 attendu, IC : [1,63 ; 2,41] |

Critère d'intervalle : les instants successifs d'un filtre sont corrélés, donc on utilise l'IC χ² à R = 100 tirages **par instant**
(et au moins 85 % des instants dedans), pas l'intervalle étroit à R·K degrés de liberté.

## Figure 1 — l'EKF décroche en virage avec un modèle CV

Un Q 100 fois plus grand l'en empêche, mais bruite la ligne droite.

![Trajectoire et erreur de position](../figures/A_trajectoire.png)

## Figure 2 — cohérence (NEES et NIS)

| cas | NEES moyen (attendu 4) | NIS moyen (attendu 2) |
|---|---|---|
| CV, Q correct | 3,96 | 1,95 |
| virage, Q nominal | 262,4 | 10,9 |
| CV, Q 100 fois trop petit | 68,0 | 2,22 |

Dans la bande χ² quand le modèle est juste. Un Q trop petit sur une trajectoire droite se voit à peine dans le NIS, très bien dans
le NEES (qui demande la vérité terrain, donc n'est utilisable qu'en simulation).

![NEES et NIS](../figures/A_nees.png)

## Figure 3 — Q faux : RMSE en fonction du facteur d'erreur sur Q

Q du filtre = f × Q de la vérité. 50 tirages par point.

| f | RMSE trajectoire CV [m] | RMSE trajectoire avec virage [m] |
|---|---|---|
| 0,01 | 25,2 | 330,9 |
| 0,1 | 22,6 | 263,3 |
| **1** | **21,0** | 130,3 |
| 10 | 23,2 | 51,2 |
| **100** | 28,7 | **30,3** |
| 1000 | 36,7 | 34,3 |

![RMSE en fonction du facteur d'erreur sur Q](../figures/A_rmse_vs_q.png)

## Ce qu'on retient

- L'EKF polaire → cartésien est honnête (NEES ≈ 4, NIS ≈ 2) tant que le modèle de mouvement est juste.
- Avec un modèle faux (virage), il devient trop sûr de lui : l'erreur augmente *et* le NEES explose.
- Un Q fixe est un **compromis** : f = 100 sauve le virage (30 m) mais coûte en ligne droite (29 m au lieu de 21 m, +37 %).
  Cette tension motive l'IMM (jalon C) et les approches apprises (jalon D).

## Reproduire

```bash
uv run pytest tests/test_kf.py tests/test_ekf.py tests/test_nees.py tests/test_trajectories.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/A_ekf.ipynb   # ~20 s
```
