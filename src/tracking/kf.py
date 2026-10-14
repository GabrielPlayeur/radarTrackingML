"""Filtre de Kalman linéaire écrit à la main (prédiction / mise à jour), sans bibliothèque.

Modèle : x_k = F·x_{k-1} + w, w ~ N(0, Q)   et   z_k = H·x_k + v, v ~ N(0, R).
Le filtre de Kalman étendu (ekf.py) hérite de cette classe et ne change que la mesure.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class KalmanFilter:
    """Filtre de Kalman : l'état x (moyenne) et sa covariance P résument tout le passé."""

    def __init__(self, F, Q, H, R, x0, P0):
        """Copie les matrices du modèle (F, Q, H, R) et l'état initial (x0, P0)."""
        self.F = np.asarray(F, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.innovation = None  # y = z - h(x⁻) de la dernière mise à jour
        self.S = None  # covariance de l'innovation de la dernière mise à jour

    def predict(self) -> None:
        """Prédiction : on propage la moyenne et on agrandit l'incertitude du bruit de modèle.

        x⁻ = F·x   et   P⁻ = F·P·Fᵀ + Q.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _measurement_matrix(self) -> np.ndarray:
        """Matrice d'observation H (constante ici ; l'EKF la remplace par une jacobienne)."""
        return self.H

    def _innovation(self, z: np.ndarray) -> np.ndarray:
        """Innovation y = z - H·x⁻ : la part de la mesure que la prédiction n'explique pas."""
        return z - self.H @ self.x

    def innovation_and_covariance(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Innovation y et covariance S qu'aurait une mise à jour avec z, sans modifier le filtre.

        y = z - h(x⁻) et S = H·P⁻·Hᵀ + R. Sert au gating : la distance de Mahalanobis yᵀ·S⁻¹·y
        mesure à quel point z est plausible compte tenu de la prédiction.
        """
        H = self._measurement_matrix()
        return self._innovation(np.asarray(z, dtype=float)), H @ self.P @ H.T + self.R

    def update(self, z: np.ndarray) -> None:
        """Mise à jour : on corrige la prédiction par l'innovation, pondérée par le gain K.

        y = z - H·x⁻,  S = H·P⁻·Hᵀ + R,  K = P⁻·Hᵀ·S⁻¹,  x = x⁻ + K·y,
        P = (I - K·H)·P⁻·(I - K·H)ᵀ + K·R·Kᵀ (forme de Joseph, qui garde P symétrique définie positive).
        """
        y, S = self.innovation_and_covariance(z)
        H = self._measurement_matrix()
        K = np.linalg.solve(S, H @ self.P).T  # P·Hᵀ·S⁻¹ (S et P symétriques)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.innovation, self.S = y, S


@dataclass
class FilterResult:
    """Historique d'un passage de filtre sur n instants (l'indice 0 est l'a priori, sans mesure)."""

    x: np.ndarray  # (n, dim_x) estimations a posteriori
    P: np.ndarray  # (n, dim_x, dim_x) covariances a posteriori
    innovations: np.ndarray  # (n, dim_z) innovations (NaN à k = 0)
    S: np.ndarray  # (n, dim_z, dim_z) covariances d'innovation (NaN à k = 0)


def run_filter(kf: KalmanFilter, measurements: np.ndarray) -> FilterResult:
    """Fait tourner le filtre : pour k = 1 … n-1, predict() puis update(z_k).

    L'instant 0 est l'a priori (x0, P0) : measurements[0] n'est pas utilisée, ce qui aligne les
    indices de l'estimation et de la vérité terrain.
    """
    n, dim_z = len(measurements), len(measurements[0])
    dim_x = len(kf.x)
    result = FilterResult(
        x=np.zeros((n, dim_x)),
        P=np.zeros((n, dim_x, dim_x)),
        innovations=np.full((n, dim_z), np.nan),
        S=np.full((n, dim_z, dim_z), np.nan),
    )
    result.x[0], result.P[0] = kf.x, kf.P
    for k in range(1, n):
        kf.predict()
        kf.update(measurements[k])
        result.x[k], result.P[k] = kf.x, kf.P
        result.innovations[k], result.S[k] = kf.innovation, kf.S
    return result


@dataclass
class MonteCarloResult:
    """Résultats empilés de n_runs passages de filtre (axe 0 = tirage)."""

    x_true: np.ndarray  # (R, n, dim_x)
    x: np.ndarray  # (R, n, dim_x)
    P: np.ndarray  # (R, n, dim_x, dim_x)
    innovations: np.ndarray  # (R, n, dim_z)
    S: np.ndarray  # (R, n, dim_z, dim_z)


def run_monte_carlo(
    n_runs: int,
    simulate_run: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    make_filter: Callable[[np.random.Generator, np.ndarray], KalmanFilter],
    seed: int = 0,
) -> MonteCarloResult:
    """Répète « générer une vérité + des mesures, filtrer » n_runs fois avec des graines distinctes.

    simulate_run(rng) → (x_true, z) ; make_filter(rng, x_true) → filtre initialisé. Chaque tirage
    a son propre générateur dérivé de `seed`, donc l'expérience est entièrement reproductible.
    """
    runs = []
    for child in np.random.SeedSequence(seed).spawn(n_runs):
        rng = np.random.default_rng(child)
        x_true, z = simulate_run(rng)
        runs.append((x_true, run_filter(make_filter(rng, x_true), z)))
    return MonteCarloResult(
        x_true=np.array([t for t, _ in runs]),
        x=np.array([r.x for _, r in runs]),
        P=np.array([r.P for _, r in runs]),
        innovations=np.array([r.innovations for _, r in runs]),
        S=np.array([r.S for _, r in runs]),
    )
