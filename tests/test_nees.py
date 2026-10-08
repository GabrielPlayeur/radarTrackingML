"""Validations de la cohérence statistique (NEES / NIS) sur 100 Monte-Carlo (jalon A).

Critère : à chaque instant k, la moyenne sur les R = 100 tirages du NEES suit χ²(R·n)/R. On vérifie
que (1) la moyenne temporelle est dans l'intervalle à 95 % et (2) au moins 85 % des instants y sont
(≈ 95 % attendus). On n'utilise pas l'intervalle étroit à R·K degrés de liberté : les instants
successifs d'un même filtre sont corrélés, les K valeurs ne sont pas indépendantes.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from metrics.nees import chi2_interval, nees, nis
from sim.trajectories import (
    TrajectoryParams,
    cartesian_measurements,
    cv_trajectory,
    maneuver_trajectory,
    polar_measurements,
    process_noise_cv,
    transition_cv,
)
from tracking.ekf import ExtendedKalmanFilter
from tracking.kf import KalmanFilter, run_monte_carlo

DT, SIGMA_A, SIGMA_Z = 2.0, 0.05, 30.0
SIGMA_R, SIGMA_THETA = 25.0, 0.01
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
P0 = np.diag([100.0**2, 100.0**2, 5.0**2, 5.0**2])
X0 = np.array([-5_000.0, 1_000.0, 0.0, -10.0])
N_RUNS = 100
TURN = [(40, 0.0), (40, 0.02), (40, 0.0)]  # CV, virage de 0,02 rad/s (1,1 °/s), CV


def simulate(rng, polar, segments=None):
    """Vérité (CV bruitée par σ_a, ou manœuvre par segments) et mesures cartésiennes ou polaires."""
    params = TrajectoryParams(dt=DT, n_steps=100, sigma_a=SIGMA_A)
    if segments is None:
        truth = cv_trajectory(X0, params, rng)
    else:
        truth = maneuver_trajectory(X0, segments, params, rng)
    if polar:
        return truth, polar_measurements(truth, SIGMA_R, SIGMA_THETA, rng)
    return truth, cartesian_measurements(truth, SIGMA_Z, rng)


def make_filter(polar, sigma_a):
    """Fabrique de filtres : a priori tiré selon N(x_vrai, P0) pour que P0 soit honnête."""

    def factory(rng, truth):
        x0 = truth[0] + np.linalg.cholesky(P0) @ rng.standard_normal(4)
        Q = process_noise_cv(DT, sigma_a)
        if polar:
            return ExtendedKalmanFilter(
                transition_cv(DT), Q, np.diag([SIGMA_R**2, SIGMA_THETA**2]), x0, P0
            )
        return KalmanFilter(transition_cv(DT), Q, H, SIGMA_Z**2 * np.eye(2), x0, P0)

    return factory


def monte_carlo(polar, sigma_a=SIGMA_A, segments=None, seed=1):
    """100 tirages ; retourne les moyennes sur les tirages du NEES (k = 0…) et du NIS (k ≥ 1)."""
    mc = run_monte_carlo(
        N_RUNS, lambda rng: simulate(rng, polar, segments), make_filter(polar, sigma_a), seed
    )
    return (
        nees(mc.x_true, mc.x, mc.P).mean(axis=0),
        nis(mc.innovations[:, 1:], mc.S[:, 1:]).mean(axis=0),
    )


def assert_consistent(mean_values, dof):
    """La moyenne temporelle est dans l'IC 95 % et ≥ 85 % des instants aussi."""
    lo, hi = chi2_interval(dof, N_RUNS)
    assert lo <= mean_values.mean() <= hi, f"moyenne {mean_values.mean():.2f} hors [{lo:.2f}, {hi:.2f}]"
    assert np.mean((mean_values >= lo) & (mean_values <= hi)) >= 0.85


@pytest.mark.parametrize("polar", [False, True], ids=["KF cartésien", "EKF polaire"])
def test_cv_with_correct_q_is_consistent(polar):
    """Scénario CV avec Q correct : NEES (χ²(4)) et NIS (χ²(2)) cohérents, KF comme EKF."""
    mean_nees, mean_nis = monte_carlo(polar)
    assert_consistent(mean_nees, dof=4)
    assert_consistent(mean_nis, dof=2)


def test_turn_with_cv_model_is_overconfident():
    """En virage avec un filtre CV, le NEES explose : le filtre est trop sûr de lui (P trop petite)."""
    mean_nees, _ = monte_carlo(polar=True, segments=TURN, seed=3)
    _, hi = chi2_interval(4, N_RUNS)
    assert mean_nees.mean() > 5 * hi


def test_q_too_large_is_conservative():
    """Q 100 fois trop grand (σ_a ×10) : le filtre est trop prudent, NEES sous la borne basse."""
    mean_nees, _ = monte_carlo(polar=True, sigma_a=10 * SIGMA_A, seed=4)
    lo, _ = chi2_interval(4, N_RUNS)
    assert mean_nees.mean() < lo


def test_q_too_small_is_detected_by_nees():
    """Q 100 fois trop petit (σ_a/10) : NEES très au-dessus de la borne haute (le NIS, lui, reste faible)."""
    mean_nees, _ = monte_carlo(polar=True, sigma_a=SIGMA_A / 10, seed=5)
    _, hi = chi2_interval(4, N_RUNS)
    assert mean_nees.mean() > 5 * hi


def test_nees_known_value():
    """Erreur [1, 2] avec P = diag(1, 4) : NEES = 1²/1 + 2²/4 = 2."""
    value = nees(np.array([1.0, 2.0]), np.zeros(2), np.diag([1.0, 4.0]))
    assert value == pytest.approx(2.0)


def test_nis_known_value_and_nan_passthrough():
    """Innovation [3, 4] avec S = diag(9, 16) : NIS = 1 + 1 = 2 ; les instants NaN restent NaN."""
    innovations = np.array([[np.nan, np.nan], [3.0, 4.0]])
    S = np.array([np.full((2, 2), np.nan), np.diag([9.0, 16.0])])
    result = nis(innovations, S)
    assert np.isnan(result[0]) and result[1] == pytest.approx(2.0)


def test_nees_of_gaussian_samples_averages_to_dimension():
    """Si l'erreur suit vraiment N(0, P), le NEES moyen vaut n (ici 4) à 2 % près."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    P = A @ A.T + np.eye(4)
    errors = rng.multivariate_normal(np.zeros(4), P, size=50_000)
    values = nees(errors, np.zeros_like(errors), np.broadcast_to(P, (50_000, 4, 4)))
    assert values.mean() == pytest.approx(4.0, rel=0.02)


def test_chi2_interval_shrinks_with_runs_and_contains_dof():
    """L'IC de la moyenne contient la dimension et se resserre en 1/√R."""
    lo1, hi1 = chi2_interval(4, 1)
    lo100, hi100 = chi2_interval(4, 100)
    assert (lo1, hi1) == pytest.approx(tuple(chi2.ppf([0.025, 0.975], 4)))
    assert lo100 < 4 < hi100
    assert (hi100 - lo100) < 0.2 * (hi1 - lo1)
