"""Trajectoires de référence (vérité terrain) et mesures bruitées, capteur radar à l'origine.

État : [x, y, vx, vy] (mètres, m/s). Pas de temps par défaut dt = 2 s (rotation d'antenne marine).
Le modèle de mouvement sert à la fois à générer la vérité et à construire les filtres, ce qui
permet de tester la cohérence (NEES) quand « Q correct » signifie vraiment « Q du générateur ».
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryParams:
    """Paramètres de génération d'une trajectoire."""

    dt: float = 2.0  # pas de temps [s]
    n_steps: int = 100  # nombre d'instants (états retournés, x0 inclus)
    sigma_a: float = 0.0  # écart-type de l'accélération aléatoire [m/s²] (0 = trajectoire lisse)
    seed: int = 0  # graine du générateur


def transition_cv(dt: float) -> np.ndarray:
    """Matrice de transition à vitesse constante : x' = x + vx·dt, y' = y + vy·dt, v' = v."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def transition_ct(dt: float, omega: float) -> np.ndarray:
    """Matrice de transition à taux de virage constant ω (rad/s, > 0 = sens trigonométrique).

    La vitesse tourne de ω·dt : v' = R(ω·dt)·v, et la position intègre cette rotation :
    x' = x + (sin(ωdt)/ω)·vx − ((1−cos(ωdt))/ω)·vy (et symétriquement pour y). Limite ω→0 : CV.
    """
    if abs(omega) < 1e-9:
        return transition_cv(dt)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    a, b = s / omega, (1 - c) / omega
    return np.array(
        [
            [1.0, 0.0, a, -b],
            [0.0, 1.0, b, a],
            [0.0, 0.0, c, -s],
            [0.0, 0.0, s, c],
        ]
    )


def noise_gain(dt: float) -> np.ndarray:
    """Matrice G (4×2) qui injecte une accélération aléatoire constante sur un pas dt.

    Δx = a·dt²/2 et Δv = a·dt, donc G = [[dt²/2, 0], [0, dt²/2], [dt, 0], [0, dt]].
    """
    G = np.zeros((4, 2))
    G[0, 0] = G[1, 1] = dt**2 / 2
    G[2, 0] = G[3, 1] = dt
    return G


def process_noise_cv(dt: float, sigma_a: float) -> np.ndarray:
    """Covariance du bruit de modèle (accélération blanche discrète) : Q = σ_a² · G · Gᵀ.

    Par axe, Q = σ_a²·[[dt⁴/4, dt³/2], [dt³/2, dt²]]. Q est de rang 2 (une accélération par axe).
    """
    G = noise_gain(dt)
    return sigma_a**2 * G @ G.T


def maneuver_trajectory(
    x0: np.ndarray,
    segments: list[tuple[int, float]],
    params: TrajectoryParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Trajectoire par segments (n_pas, ω) : ω = 0 donne un segment CV, ω ≠ 0 un virage CT.

    x_{k+1} = F(ω)·x_k + G·a_k avec a_k ~ N(0, σ_a²·I₂). Retourne 1 + Σ n_pas états (x0 inclus).
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)
    G = noise_gain(params.dt)
    states = [np.asarray(x0, dtype=float)]
    for n_pas, omega in segments:
        F = transition_ct(params.dt, omega)
        for _ in range(n_pas):
            accel = params.sigma_a * rng.standard_normal(2)
            states.append(F @ states[-1] + G @ accel)
    return np.array(states)


def cv_trajectory(
    x0: np.ndarray, params: TrajectoryParams, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Trajectoire à vitesse constante (perturbée par σ_a si non nul) : params.n_steps états."""
    return maneuver_trajectory(x0, [(params.n_steps - 1, 0.0)], params, rng)


def ct_trajectory(
    x0: np.ndarray, omega: float, params: TrajectoryParams, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Trajectoire en virage à taux constant ω : cercle de rayon |v|/|ω| si σ_a = 0."""
    return maneuver_trajectory(x0, [(params.n_steps - 1, omega)], params, rng)


def cartesian_measurements(
    states: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Mesures de position cartésiennes bruitées : z = [x, y] + N(0, σ²·I₂), forme (n, 2)."""
    return states[:, :2] + sigma * rng.standard_normal((len(states), 2))


def polar_measurements(
    states: np.ndarray, sigma_r: float, sigma_theta: float, rng: np.random.Generator
) -> np.ndarray:
    """Mesures polaires bruitées : z = [√(x²+y²), atan2(y, x)] + bruit, forme (n, 2).

    L'angle bruité est ramené dans [−π, π] (comme le ferait un capteur réel).
    """
    r = np.hypot(states[:, 0], states[:, 1]) + sigma_r * rng.standard_normal(len(states))
    theta = np.arctan2(states[:, 1], states[:, 0]) + sigma_theta * rng.standard_normal(len(states))
    return np.column_stack([r, np.arctan2(np.sin(theta), np.cos(theta))])
