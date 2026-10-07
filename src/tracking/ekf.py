"""Filtre de Kalman étendu (EKF) : mouvement linéaire, mesures radar polaires (r, θ).

Le radar mesure une distance et un angle, mais l'état est cartésien [x, y, vx, vy] : la mesure
h(x) = [√(x²+y²), atan2(y, x)] est non linéaire. L'EKF la linéarise autour de la prédiction
avec la jacobienne H = ∂h/∂x, puis applique les équations du filtre de Kalman.
"""

import numpy as np

from tracking.kf import KalmanFilter


def wrap_angle(angle):
    """Ramène un angle dans [−π, π[ : a − 2π·floor((a + π) / 2π).

    Piège classique : θ = 179° et θ = −179° ne sont distants que de 2°, pas de 358°.
    """
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def polar_h(x: np.ndarray) -> np.ndarray:
    """Fonction de mesure polaire : h(x) = [√(x² + y²), atan2(y, x)] (capteur à l'origine)."""
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def polar_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobienne analytique de h : H = [[x/r, y/r, 0, 0], [−y/r², x/r², 0, 0]], r² = x² + y².

    Les dérivées par rapport à vx et vy sont nulles : la mesure ne dépend que de la position.
    """
    px, py = x[0], x[1]
    r2 = px**2 + py**2
    if r2 < 1e-12:
        raise ValueError("Jacobienne polaire indéfinie à l'origine (r = 0)")
    r = np.sqrt(r2)
    return np.array([[px / r, py / r, 0.0, 0.0], [-py / r2, px / r2, 0.0, 0.0]])


def polar_to_cartesian(r: float, theta: float) -> np.ndarray:
    """Position cartésienne d'une mesure polaire : [r·cos θ, r·sin θ]."""
    return np.array([r * np.cos(theta), r * np.sin(theta)])


class ExtendedKalmanFilter(KalmanFilter):
    """EKF à mesures polaires : seul le calcul de l'innovation et de H change par rapport au KF."""

    def __init__(self, F, Q, R, x0, P0):
        """R est la covariance de bruit polaire diag(σ_r², σ_θ²) ; H est recalculée à chaque mise à jour."""
        super().__init__(F, Q, polar_jacobian(np.asarray(x0, dtype=float)), R, x0, P0)

    def _measurement_matrix(self) -> np.ndarray:
        """Jacobienne de h évaluée à la prédiction : H = ∂h/∂x |_{x⁻}."""
        return polar_jacobian(self.x)

    def _innovation(self, z: np.ndarray) -> np.ndarray:
        """Innovation y = z − h(x⁻), avec la composante d'angle normalisée dans [−π, π].

        Sans cette normalisation, z_θ = 179° et θ̂ = −179° donneraient une innovation de 358° :
        le filtre sauterait à l'opposé du plan à chaque passage de la coupure ±π.
        """
        y = z - polar_h(self.x)
        y[1] = wrap_angle(y[1])
        return y
