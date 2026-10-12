"""Gestion de pistes : initialisation, confirmation M/N, suppression, et exécution sur un scénario.

Cycle par balayage : prédire toutes les pistes → associer (GNN) → mettre à jour les pistes associées →
confirmer / supprimer → créer une piste provisoire par détection restée seule.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from sim.trajectories import process_noise_cv, transition_cv
from tracking.ekf import ExtendedKalmanFilter
from tracking.gnn import associate


@dataclass(frozen=True)
class TrackManagerParams:
    """Règles de gestion de pistes."""

    gate_probability: float = 0.999  # probabilité de la porte de Mahalanobis
    confirm_m: int = 3  # M détections ...
    confirm_n: int = 4  # ... parmi les N derniers balayages pour confirmer une piste provisoire
    max_misses: int = 5  # une piste confirmée est supprimée après K balayages consécutifs sans détection


class Track:
    """Une piste : un estimateur (KF, EKF ou IMM) plus son historique de détections."""

    def __init__(self, track_id: int, estimator):
        self.id = track_id
        self.estimator = estimator
        self.confirmed = False
        self.history = [True]  # détectée à sa création
        self.consecutive_misses = 0

    def record(self, hit: bool) -> None:
        """Enregistre une détection (True) ou un raté (False) pour ce balayage."""
        self.history.append(hit)
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1


class TrackManager:
    """Gère l'ensemble des pistes : provisoires (M/N pas encore atteint) et confirmées."""

    def __init__(
        self,
        make_estimator: Callable[[np.ndarray], object],
        params: TrackManagerParams = TrackManagerParams(),
    ):
        """`make_estimator(z)` construit l'estimateur d'une nouvelle piste à partir d'une détection polaire z."""
        self.make_estimator = make_estimator
        self.params = params
        self.tracks: list[Track] = []
        self._next_id = 0

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Pistes confirmées (celles que l'on présente à l'opérateur)."""
        return [t for t in self.tracks if t.confirmed]

    def step(self, detections: np.ndarray) -> None:
        """Traite un balayage : prédiction, association GNN, mise à jour, confirmation, suppression, création.

        Confirmation M/N : une piste provisoire devient confirmée si elle a ≥ M détections dans les N
        derniers balayages ; elle est supprimée dès que M devient inatteignable dans la fenêtre.
        """
        p = self.params
        for track in self.tracks:
            track.estimator.predict()

        result = associate([t.estimator for t in self.tracks], detections, p.gate_probability)
        for i, j in result.matches:
            self.tracks[i].estimator.update(detections[j])
            self.tracks[i].record(True)
        for i in result.unmatched_tracks:
            self.tracks[i].record(False)

        survivors = []
        for track in self.tracks:
            if track.confirmed:
                keep = track.consecutive_misses < p.max_misses
            else:
                window = track.history[-p.confirm_n :]
                hits = sum(window)
                if hits >= p.confirm_m:
                    track.confirmed = True
                keep = track.confirmed or hits + (p.confirm_n - len(window)) >= p.confirm_m
            if keep:
                survivors.append(track)
        self.tracks = survivors

        for j in result.unmatched_detections:
            self.tracks.append(Track(self._next_id, self.make_estimator(detections[j])))
            self._next_id += 1


def initial_state_from_detection(
    z: np.ndarray, sigma_r: float, sigma_theta: float, sigma_v: float
) -> tuple[np.ndarray, np.ndarray]:
    """État et covariance initiaux d'une piste à partir d'une seule détection polaire (r, θ).

    Position : [r·cos θ, r·sin θ], covariance J·diag(σ_r², σ_θ²)·Jᵀ avec J = ∂(x, y)/∂(r, θ).
    Vitesse inconnue : nulle, d'écart-type σ_v (de l'ordre de la vitesse maximale d'un navire).
    """
    r, theta = z
    c, s = np.cos(theta), np.sin(theta)
    J = np.array([[c, -r * s], [s, r * c]])
    P0 = np.zeros((4, 4))
    P0[:2, :2] = J @ np.diag([sigma_r**2, sigma_theta**2]) @ J.T
    P0[2:, 2:] = sigma_v**2 * np.eye(2)
    return np.array([r * c, r * s, 0.0, 0.0]), P0


def make_ekf_factory(
    dt: float, sigma_a: float, sigma_r: float, sigma_theta: float, sigma_v: float = 10.0
) -> Callable[[np.ndarray], ExtendedKalmanFilter]:
    """Fabrique d'EKF à modèle CV pour le gestionnaire de pistes (mesures polaires)."""
    F, Q = transition_cv(dt), process_noise_cv(dt, sigma_a)
    R = np.diag([sigma_r**2, sigma_theta**2])

    def factory(z: np.ndarray) -> ExtendedKalmanFilter:
        x0, P0 = initial_state_from_detection(z, sigma_r, sigma_theta, sigma_v)
        return ExtendedKalmanFilter(F, Q, R, x0, P0)

    return factory


@dataclass
class TrackingRun:
    """Sortie d'un passage du pisteur sur un scénario."""

    ids: list = field(default_factory=list)  # par balayage : identifiants des pistes confirmées
    states: list = field(default_factory=list)  # par balayage : états (n, 4) des pistes confirmées
    step_times: list = field(default_factory=list)  # durée de chaque balayage [s]
    n_active: list = field(default_factory=list)  # pistes (provisoires + confirmées) traitées par balayage


def run_tracking(manager: TrackManager, detections_per_scan: list) -> TrackingRun:
    """Fait tourner le gestionnaire sur toutes les détections d'un scénario, en chronométrant chaque balayage.

    Temps par mise à jour = Σ durées / Σ pistes traitées (prédiction + association + mise à jour).
    """
    run = TrackingRun()
    for detections in detections_per_scan:
        n_before = len(manager.tracks)
        start = time.perf_counter()
        manager.step(detections)
        run.step_times.append(time.perf_counter() - start)
        run.n_active.append(n_before)
        confirmed = manager.confirmed_tracks
        run.ids.append(np.array([t.id for t in confirmed], dtype=int))
        run.states.append(np.array([t.estimator.x for t in confirmed]).reshape(-1, 4))
    return run
