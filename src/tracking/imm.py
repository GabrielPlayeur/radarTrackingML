"""IMM (Interacting Multiple Model) : plusieurs filtres de modèles de mouvement différents qui se mélangent.

Idée : un navire est tantôt en ligne droite (CV), tantôt en virage (CT). Plutôt que de choisir un
modèle, on fait tourner un filtre par modèle et on pondère par la probabilité μ_j que le navire
suive le modèle j. Cette probabilité évolue : une chaîne de Markov (matrice π) propose les
changements de modèle, et la vraisemblance des mesures confirme ou infirme.
"""

from collections.abc import Callable

import numpy as np

from sim.trajectories import process_noise_cv, transition_ct, transition_cv
from tracking.ekf import ExtendedKalmanFilter
from tracking.track_manager import initial_state_from_detection


class IMM:
    """IMM sur une liste de filtres (KF ou EKF) qui partagent le même espace d'état et la même mesure."""

    def __init__(self, filters: list, mu0, transition):
        """filters : un filtre par modèle ; mu0 : probabilités initiales ; transition[i, j] = π_ij = P(i → j)."""
        self.filters = filters
        self.mu = np.asarray(mu0, dtype=float)
        self.pi = np.asarray(transition, dtype=float)
        self.mu_pred = self.mu.copy()
        self.innovation = None  # innovation du mélange à la dernière mise à jour (voir innovation_and_covariance)
        self.S = None
        self._combine(self.mu)

    def _combine(self, weights: np.ndarray) -> None:
        """Estimation combinée : x = Σ_j w_j·x_j et P = Σ_j w_j·(P_j + (x_j − x)(x_j − x)ᵀ)."""
        xs = np.array([f.x for f in self.filters])
        self.x = weights @ xs
        dx = xs - self.x
        self.P = sum(w * (f.P + np.outer(d, d)) for w, f, d in zip(weights, self.filters, dx))

    def predict(self) -> None:
        """Mélange des états puis prédiction de chaque filtre.

        c̄_j = Σ_i π_ij·μ_i ; μ_{i|j} = π_ij·μ_i / c̄_j ; x⁰_j = Σ_i μ_{i|j}·x_i ;
        P⁰_j = Σ_i μ_{i|j}·(P_i + (x_i − x⁰_j)(x_i − x⁰_j)ᵀ). Chaque filtre j prédit ensuite depuis (x⁰_j, P⁰_j).
        """
        c_bar = self.pi.T @ self.mu
        mixing = self.pi * self.mu[:, None] / c_bar[None, :]  # mixing[i, j] = μ_{i|j}
        xs = np.array([f.x for f in self.filters])
        Ps = np.array([f.P for f in self.filters])
        mixed = []
        for j in range(len(self.filters)):
            x0 = mixing[:, j] @ xs
            dx = xs - x0
            P0 = sum(mixing[i, j] * (Ps[i] + np.outer(dx[i], dx[i])) for i in range(len(xs)))
            mixed.append((x0, P0))
        for f, (x0, P0) in zip(self.filters, mixed):
            f.x, f.P = x0, P0
            f.predict()
        self.mu = self.mu_pred = c_bar  # sans mesure, les probabilités sont celles de la chaîne de Markov
        self._combine(c_bar)

    def innovation_and_covariance(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Innovation et covariance du mélange, pour le gating : y = Σ μ_j·y_j, S = Σ μ_j·(S_j + (y_j − y)(y_j − y)ᵀ)."""
        parts = [f.innovation_and_covariance(z) for f in self.filters]
        y = sum(w * yj for w, (yj, _) in zip(self.mu_pred, parts))
        S = sum(w * (Sj + np.outer(yj - y, yj - y)) for w, (yj, Sj) in zip(self.mu_pred, parts))
        return y, S

    def update(self, z: np.ndarray) -> None:
        """Mise à jour de chaque filtre, puis des probabilités : μ_j ∝ Λ_j·c̄_j avec Λ_j = N(y_j ; 0, S_j)."""
        log_lik = np.empty(len(self.filters))
        for j, f in enumerate(self.filters):
            f.update(z)
            _, logdet = np.linalg.slogdet(2 * np.pi * f.S)
            log_lik[j] = -0.5 * (f.innovation @ np.linalg.solve(f.S, f.innovation) + logdet)
        ys = [f.innovation for f in self.filters]
        self.innovation = sum(w * y for w, y in zip(self.mu_pred, ys))
        self.S = sum(
            w * (f.S + np.outer(y - self.innovation, y - self.innovation))
            for w, f, y in zip(self.mu_pred, self.filters, ys)
        )
        log_mu = np.log(self.mu_pred) + log_lik
        self.mu = np.exp(log_mu - log_mu.max())
        self.mu /= self.mu.sum()
        self._combine(self.mu)


def transition_matrix(n_models: int, p_stay: float) -> np.ndarray:
    """Matrice de transition π : on reste dans son modèle avec la probabilité p_stay, sinon on change vers l'un des autres."""
    pi = np.full((n_models, n_models), (1 - p_stay) / (n_models - 1))
    np.fill_diagonal(pi, p_stay)
    return pi


def build_imm(
    x0: np.ndarray,
    P0: np.ndarray,
    dt: float,
    sigma_a: float,
    R: np.ndarray,
    turn_rates: tuple = (0.02,),
    p_stay: float = 0.98,
    mu_cv: float = 0.9,
) -> IMM:
    """IMM à mesures polaires : un modèle CV puis un modèle CT par taux de virage de `turn_rates` (rad/s).

    Tous les modèles ont le même bruit d'accélération σ_a et partent du même (x0, P0). Avec
    turn_rates = (ω,) c'est l'IMM « CV + CT » ; avec (ω, −ω) il couvre les virages dans les deux sens.
    """
    Q = process_noise_cv(dt, sigma_a)
    Fs = [transition_cv(dt)] + [transition_ct(dt, w) for w in turn_rates]
    filters = [ExtendedKalmanFilter(F, Q, R, x0, P0) for F in Fs]
    n = len(filters)
    mu0 = np.array([mu_cv] + [(1 - mu_cv) / (n - 1)] * (n - 1))
    return IMM(filters, mu0, transition_matrix(n, p_stay))


def make_imm_factory(
    dt: float,
    sigma_a: float,
    sigma_r: float,
    sigma_theta: float,
    sigma_v: float = 10.0,
    turn_rates: tuple = (0.02,),
    p_stay: float = 0.98,
) -> Callable[[np.ndarray], IMM]:
    """Fabrique d'IMM pour le gestionnaire de pistes (initialisation depuis une détection polaire)."""
    R = np.diag([sigma_r**2, sigma_theta**2])

    def factory(z: np.ndarray) -> IMM:
        x0, P0 = initial_state_from_detection(z, sigma_r, sigma_theta, sigma_v)
        return build_imm(x0, P0, dt, sigma_a, R, turn_rates, p_stay)

    return factory
