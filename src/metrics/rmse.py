"""Erreurs quadratiques moyennes de position."""

import numpy as np


def position_error_squared(x_true: np.ndarray, x_est: np.ndarray) -> np.ndarray:
    """Erreur de position au carré : (x−x̂)² + (y−ŷ)², sur les deux premières composantes d'état.

    Fonctionne pour un tirage (n, 4) ou plusieurs (R, n, 4) : `x_true` est diffusé sur `x_est`.
    """
    return np.sum((x_true[..., :2] - x_est[..., :2]) ** 2, axis=-1)


def rmse_per_step(x_true: np.ndarray, x_est: np.ndarray) -> np.ndarray:
    """RMSE de position à chaque instant, moyenne quadratique sur les tirages : √E[‖p − p̂‖²].

    Pour un seul tirage (n, 4), c'est simplement la norme de l'erreur de position à chaque instant.
    """
    err2 = position_error_squared(x_true, x_est)
    return np.sqrt(err2.mean(axis=0)) if err2.ndim > 1 else np.sqrt(err2)


def position_rmse(x_true: np.ndarray, x_est: np.ndarray, skip: int = 0) -> float:
    """RMSE de position globale √(moyenne de ‖p − p̂‖² sur les tirages et les instants k ≥ skip).

    `skip` écarte le transitoire d'initialisation du filtre.
    """
    err2 = position_error_squared(x_true, x_est)
    return float(np.sqrt(err2[..., skip:].mean()))
