"""Erreurs quadratiques moyennes de position et statistiques de pistes (pertes, fausses pistes, identités)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


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


@dataclass
class TrackStats:
    """Qualité d'un passage du pisteur par rapport à la vérité terrain."""

    rmse: float  # RMSE de position des paires (cible, piste) associées [m]
    rmse_masked: float  # idem, restreinte aux instants du masque (ex. : en virage) ; NaN sans masque
    n_losses: int  # une cible pistée à k−1 et non pistée à k : perte de piste
    n_id_switches: int  # une cible pistée à k−1 et à k par deux pistes différentes
    n_false_tracks: int  # pistes confirmées associées à une cible dans moins de la moitié de leur vie
    n_never_tracked: int  # cibles qui n'ont jamais été pistées
    tracked_fraction: float  # part des couples (cible, balayage) où la cible est pistée


def assign_truth_to_tracks(truth_xy: np.ndarray, track_xy: np.ndarray, max_dist: float) -> np.ndarray:
    """Pour chaque cible, l'indice de la piste associée (−1 si aucune) : hongrois sur les distances ≤ max_dist."""
    assigned = -np.ones(len(truth_xy), dtype=int)
    if len(truth_xy) == 0 or len(track_xy) == 0:
        return assigned
    dist = np.linalg.norm(truth_xy[:, None, :] - track_xy[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(dist <= max_dist, dist, 1e9))
    for i, j in zip(rows, cols):
        if dist[i, j] <= max_dist:
            assigned[i] = j
    return assigned


def track_statistics(
    truth: np.ndarray, ids: list, states: list, max_dist: float = 300.0, mask: np.ndarray | None = None
) -> TrackStats:
    """Compare les pistes confirmées à la vérité : RMSE, pertes, changements d'identité, fausses pistes.

    truth : (n_cibles, n_balayages, 4) ; ids[k] et states[k] : identifiants et états (n, 4) des pistes
    confirmées au balayage k ; mask : (n_cibles, n_balayages) booléen pour un RMSE restreint (virages).
    À chaque balayage, cibles et pistes sont appariées (hongrois, distance ≤ max_dist).
    """
    n_ships, n_steps = truth.shape[:2]
    owner = -np.ones((n_ships, n_steps), dtype=int)  # identifiant de la piste associée
    err2 = np.full((n_ships, n_steps), np.nan)
    life, matched_life = {}, {}
    for k in range(n_steps):
        track_xy = states[k][:, :2]
        assigned = assign_truth_to_tracks(truth[:, k, :2], track_xy, max_dist)
        for s, j in enumerate(assigned):
            if j >= 0:
                owner[s, k] = ids[k][j]
                err2[s, k] = np.sum((truth[s, k, :2] - track_xy[j]) ** 2)
        used = {int(j) for j in assigned if j >= 0}
        for j, track_id in enumerate(ids[k]):
            life[int(track_id)] = life.get(int(track_id), 0) + 1
            matched_life[int(track_id)] = matched_life.get(int(track_id), 0) + (j in used)

    tracked = owner >= 0
    before, after = tracked[:, :-1], tracked[:, 1:]
    return TrackStats(
        rmse=float(np.sqrt(np.nanmean(err2))) if tracked.any() else float("nan"),
        rmse_masked=float(np.sqrt(np.nanmean(np.where(mask, err2, np.nan))))
        if mask is not None and (tracked & mask).any()
        else float("nan"),
        n_losses=int(np.sum(before & ~after)),
        n_id_switches=int(np.sum(before & after & (owner[:, :-1] != owner[:, 1:]))),
        n_false_tracks=sum(1 for t in life if matched_life[t] / life[t] < 0.5),
        n_never_tracked=int(np.sum(~tracked.any(axis=1))),
        tracked_fraction=float(tracked.mean()),
    )
