### test (distribution d'entraînement)

| méthode | RMSE globale [m] | RMSE en virage [m] | RMSE en ligne droite [m] |
|---|---|---|---|
| EKF-CV | 59.5 ± 5.7 | 99.4 ± 9.1 | 52.4 ± 5.3 |
| IMM (CV + CT±) | 28.2 ± 0.9 | 33.5 ± 2.5 | 27.4 ± 0.9 |
| gain fixe (non appris) | 35.1 ± 1.1 | 40.9 ± 2.5 | 34.3 ± 1.1 |
| KalmanNet | 29.7 ± 0.2 | 43.6 ± 0.9 | 27.5 ± 0.2 |

### virages 3 à 4 fois plus forts

| méthode | RMSE globale [m] | RMSE en virage [m] | RMSE en ligne droite [m] |
|---|---|---|---|
| EKF-CV | 106.3 ± 5.8 | 103.2 ± 5.3 | 106.7 ± 6.1 |
| IMM (CV + CT±) | 64.3 ± 4.0 | 69.1 ± 4.0 | 63.7 ± 4.1 |
| gain fixe (non appris) | 41.7 ± 1.1 | 60.0 ± 2.4 | 38.9 ± 1.1 |
| KalmanNet | 44.7 ± 2.6 | 72.6 ± 3.3 | 40.1 ± 2.6 |

### bruit de mesure ×2

| méthode | RMSE globale [m] | RMSE en virage [m] | RMSE en ligne droite [m] |
|---|---|---|---|
| EKF-CV | 77.2 ± 5.8 | 104.2 ± 9.4 | 71.9 ± 5.6 |
| IMM (CV + CT±) | 55.7 ± 1.8 | 57.1 ± 3.2 | 55.5 ± 1.8 |
| gain fixe (non appris) | 65.9 ± 2.1 | 69.3 ± 3.8 | 65.3 ± 2.3 |
| KalmanNet | 59.0 ± 1.7 | 67.7 ± 1.1 | 57.5 ± 2.2 |

### séquences 4 fois plus longues

| méthode | RMSE globale [m] | RMSE en virage [m] | RMSE en ligne droite [m] |
|---|---|---|---|
| EKF-CV | 33.3 ± 2.3 | 90.4 ± 8.4 | 29.4 ± 1.7 |
| IMM (CV + CT±) | 23.0 ± 0.5 | 28.5 ± 1.8 | 22.8 ± 0.5 |
| gain fixe (non appris) | 30.6 ± 0.9 | 39.1 ± 2.0 | 30.3 ± 0.9 |
| KalmanNet | 23.1 ± 0.1 | 39.8 ± 1.4 | 22.3 ± 0.1 |

IC à 95 % : rééchantillonnage (bootstrap) des trajectoires de test pour les références ; Student sur les 3 graines d'entraînement pour KalmanNet.

### Coût

| méthode | paramètres appris | temps par mise à jour [µs] |
|---|---|---|
| EKF-CV | 0 | 40 |
| IMM (CV + CT±) | 0 | 324 |
| gain fixe (non appris) | 0 | 433 |
| KalmanNet | 19592 | 433 |

### Quand ça ne marche pas

- **test (distribution d'entraînement)** : KalmanNet 29.7 m, moins bon que IMM (CV + CT±) (28.2 m).
- **virages 3 à 4 fois plus forts** : KalmanNet 44.7 m, moins bon que gain fixe (non appris) (41.7 m).
- **bruit de mesure ×2** : KalmanNet 59.0 m, moins bon que IMM (CV + CT±) (55.7 m).
- **séquences 4 fois plus longues** : KalmanNet 23.1 m, moins bon que IMM (CV + CT±) (23.0 m).
