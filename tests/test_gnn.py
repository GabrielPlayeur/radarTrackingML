"""Validations du gating de Mahalanobis et de l'association hongroise (jalon C)."""

import itertools

import numpy as np
import pytest

from sim.trajectories import process_noise_cv, transition_cv
from tracking.ekf import ExtendedKalmanFilter
from tracking.gnn import associate, distance_matrix, gate_threshold, mahalanobis_squared


class PointTrack:
    """Piste factice : position prédite p, covariance d'innovation σ²·I, mesure = position."""

    def __init__(self, position, sigma=1.0):
        self.p = np.asarray(position, dtype=float)
        self.S = sigma**2 * np.eye(2)

    def innovation_and_covariance(self, z):
        return np.asarray(z, dtype=float) - self.p, self.S


def test_gate_threshold_matches_chi2_table():
    """χ²(2) à 99,9 % = 13,8155 ; à 95 % = 5,9915 (valeurs de table)."""
    assert gate_threshold(0.999, 2) == pytest.approx(13.8155, abs=1e-3)
    assert gate_threshold(0.95, 2) == pytest.approx(5.9915, abs=1e-3)


def test_gate_keeps_the_requested_fraction_of_true_detections():
    """Si l'innovation suit N(0, S), la porte à 95 % garde ≈ 95 % des vraies détections."""
    rng = np.random.default_rng(0)
    S = np.array([[4.0, 1.5], [1.5, 2.0]])
    y = rng.multivariate_normal(np.zeros(2), S, size=20_000)
    d2 = np.array([mahalanobis_squared(v, S) for v in y])
    assert np.mean(d2 < gate_threshold(0.95)) == pytest.approx(0.95, abs=0.01)


def test_mahalanobis_accounts_for_covariance_shape():
    """Même distance euclidienne, mais plus plausible le long de l'axe où l'incertitude est grande."""
    S = np.diag([100.0, 1.0])
    assert mahalanobis_squared(np.array([10.0, 0.0]), S) == pytest.approx(1.0)
    assert mahalanobis_squared(np.array([0.0, 10.0]), S) == pytest.approx(100.0)


def test_hungarian_matches_brute_force_optimum():
    """Sur 200 petits problèmes aléatoires, l'affectation égale l'optimum trouvé par force brute."""
    rng = np.random.default_rng(1)
    gamma = gate_threshold(0.999)
    for _ in range(200):
        n_t, n_d = rng.integers(1, 5), rng.integers(1, 5)
        tracks = [PointTrack(rng.uniform(0, 10, 2)) for _ in range(n_t)]
        dets = rng.uniform(0, 10, (n_d, 2))
        d2 = distance_matrix(tracks, dets)

        best = (0, 0.0)  # (nombre d'associations valides, −coût) à maximiser
        for choice in itertools.product([None, *range(n_d)], repeat=n_t):
            used = [c for c in choice if c is not None]
            if len(used) != len(set(used)):
                continue
            pairs = [(i, c) for i, c in enumerate(choice) if c is not None and d2[i, c] < gamma]
            if len(pairs) != len(used):
                continue
            cand = (len(pairs), -sum(d2[i, c] for i, c in pairs))
            best = max(best, cand)

        result = associate(tracks, dets)
        got = (len(result.matches), -sum(d2[i, j] for i, j in result.matches))
        assert got[0] == best[0]
        assert got[1] == pytest.approx(best[1], abs=1e-9)


def test_global_assignment_beats_greedy_nearest_neighbour():
    """Cas où le plus proche voisin glouton est piégé : le GNN trouve un coût total bien plus bas."""
    tracks = [PointTrack([0.0, 0.0], 2.0), PointTrack([3.0, 0.0], 2.0)]
    dets = np.array([[1.0, 0.0], [-1.2, 0.0]])
    d2 = distance_matrix(tracks, dets)  # [[0,25 ; 0,36], [1,00 ; 4,41]]

    # glouton : on prend la plus petite paire, puis la meilleure paire restante
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    greedy_cost = d2[i, j] + d2[1 - i, 1 - j]
    result = associate(tracks, dets)
    gnn_cost = sum(d2[a, b] for a, b in result.matches)

    assert sorted(result.matches) == [(0, 1), (1, 0)]  # croisement : optimum global
    assert gnn_cost == pytest.approx(0.36 + 1.00)
    assert greedy_cost == pytest.approx(0.25 + 4.41)
    assert gnn_cost < greedy_cost / 3


def test_detection_outside_the_gate_is_left_unmatched():
    """Une détection très éloignée n'est associée à aucune piste, ni la piste à aucune détection."""
    result = associate([PointTrack([0.0, 0.0])], np.array([[50.0, 50.0]]))
    assert result.matches == []
    assert result.unmatched_tracks == [0] and result.unmatched_detections == [0]


def test_unmatched_bookkeeping_is_a_partition():
    """Pistes et détections se répartissent exactement entre associées et non associées."""
    tracks = [PointTrack([0, 0]), PointTrack([10, 0]), PointTrack([100, 0])]
    dets = np.array([[0.5, 0.0], [10.2, 0.0], [-80.0, 40.0], [300.0, 0.0]])
    result = associate(tracks, dets)
    assert sorted(result.matches) == [(0, 0), (1, 1)]
    assert result.unmatched_tracks == [2]
    assert result.unmatched_detections == [2, 3]


def test_empty_inputs():
    """Sans piste ou sans détection, tout reste non associé."""
    assert associate([], np.zeros((3, 2))).unmatched_detections == [0, 1, 2]
    assert associate([PointTrack([0, 0])], np.zeros((0, 2))).unmatched_tracks == [0]


def test_association_works_across_the_theta_pi_cut():
    """Piste prédite à θ = π − 0,002 et détection à θ = −π + 0,002 : associées (innovation normalisée)."""
    P0 = np.diag([100.0**2, 100.0**2, 5.0**2, 5.0**2])
    ekf = ExtendedKalmanFilter(
        transition_cv(2.0), process_noise_cv(2.0, 0.05), np.diag([25.0**2, 0.01**2]),
        np.array([-5_000.0, 10.0, 0.0, 0.0]), P0,
    )
    ekf.predict()
    theta_pred = np.arctan2(ekf.x[1], ekf.x[0])
    assert theta_pred > 3.0  # côté +π de la coupure
    detection = np.array([[np.hypot(ekf.x[0], ekf.x[1]), -np.pi + (np.pi - theta_pred)]])
    result = associate([ekf], detection)
    assert result.matches == [(0, 0)]
