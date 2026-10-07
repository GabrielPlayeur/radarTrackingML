"""Validations des générateurs de trajectoires CV / CT et des mesures (jalon A)."""

import numpy as np
import pytest
from filterpy.common import Q_discrete_white_noise

from sim.trajectories import (
    TrajectoryParams,
    cartesian_measurements,
    ct_trajectory,
    cv_trajectory,
    maneuver_trajectory,
    polar_measurements,
    process_noise_cv,
    transition_ct,
    transition_cv,
)

X0 = np.array([1_000.0, -2_000.0, 8.0, 6.0])
PARAMS = TrajectoryParams(dt=2.0, n_steps=50)


def test_cv_is_a_straight_line_at_constant_velocity():
    """Sans bruit, CV : position = x0 + v·t et vitesse inchangée."""
    states = cv_trajectory(X0, PARAMS)
    t = PARAMS.dt * np.arange(PARAMS.n_steps)
    assert states.shape == (PARAMS.n_steps, 4)
    np.testing.assert_allclose(states[:, 0], X0[0] + X0[2] * t)
    np.testing.assert_allclose(states[:, 1], X0[1] + X0[3] * t)
    np.testing.assert_allclose(states[:, 2:], np.tile(X0[2:], (PARAMS.n_steps, 1)))


def test_ct_is_a_circle_with_constant_speed_and_turn_rate():
    """Sans bruit, CT : vitesse constante, cap qui tourne de ω·dt par pas, cercle de rayon |v|/ω."""
    omega = 0.02
    states = ct_trajectory(X0, omega, PARAMS)
    speed = np.hypot(states[:, 2], states[:, 3])
    np.testing.assert_allclose(speed, np.hypot(X0[2], X0[3]))

    heading = np.unwrap(np.arctan2(states[:, 3], states[:, 2]))
    np.testing.assert_allclose(np.diff(heading), omega * PARAMS.dt)

    center = X0[:2] + np.array([-X0[3], X0[2]]) / omega
    radius = np.hypot(*(states[:, :2] - center).T)
    np.testing.assert_allclose(radius, np.hypot(X0[2], X0[3]) / omega)


def test_ct_tends_to_cv_when_turn_rate_vanishes():
    """La limite ω → 0 de la matrice CT est la matrice CV."""
    np.testing.assert_allclose(transition_ct(2.0, 1e-12), transition_cv(2.0))
    np.testing.assert_allclose(transition_ct(2.0, 1e-5), transition_cv(2.0), atol=1e-4)


def test_maneuver_chains_segments():
    """CV puis CT puis CV : 1 + Σ pas états, et la vitesse change de cap seulement pendant le virage."""
    states = maneuver_trajectory(X0, [(10, 0.0), (20, 0.03), (10, 0.0)], PARAMS)
    assert states.shape == (41, 4)
    heading = np.arctan2(states[:, 3], states[:, 2])
    np.testing.assert_allclose(heading[:11], heading[0])
    np.testing.assert_allclose(heading[30:], heading[30])
    assert abs(heading[30] - heading[10]) == pytest.approx(0.03 * 20 * PARAMS.dt)


def test_process_noise_matches_filterpy_discrete_white_noise():
    """Q par axe = σ²·[[dt⁴/4, dt³/2], [dt³/2, dt²]], identique à filterpy (état [x, y, vx, vy])."""
    dt, sigma_a = 2.0, 0.3
    block = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_a**2)
    Q = process_noise_cv(dt, sigma_a)
    for axis in range(2):
        idx = [axis, axis + 2]  # (position, vitesse) de l'axe
        np.testing.assert_allclose(Q[np.ix_(idx, idx)], block, atol=1e-12)
    assert Q[0, 1] == Q[0, 3] == Q[2, 1] == 0  # axes x et y indépendants


def test_generated_process_noise_has_covariance_Q():
    """La covariance empirique de F·x + G·a − F·x vaut Q = σ_a²·G·Gᵀ (à 5 % près sur la diagonale)."""
    params = TrajectoryParams(dt=2.0, n_steps=2, sigma_a=0.5)
    rng = np.random.default_rng(0)
    steps = np.array([cv_trajectory(np.zeros(4), params, rng)[1] for _ in range(20_000)])
    Q = process_noise_cv(params.dt, params.sigma_a)
    np.testing.assert_allclose(np.diag(np.cov(steps.T)), np.diag(Q), rtol=0.05)


def test_cartesian_measurement_noise_level():
    """L'écart-type du bruit de mesure cartésien vaut σ."""
    states = np.zeros((20_000, 4))
    z = cartesian_measurements(states, 30.0, np.random.default_rng(1))
    assert z.std(axis=0) == pytest.approx([30.0, 30.0], rel=0.03)


def test_polar_measurement_noise_level_and_angle_range():
    """Bruit polaire d'écart-types (σ_r, σ_θ) autour de la vérité, et θ ∈ [−π, π]."""
    states = np.tile([3_000.0, 4_000.0, 0.0, 0.0], (20_000, 1))
    z = polar_measurements(states, 25.0, 0.01, np.random.default_rng(2))
    assert z[:, 0].mean() == pytest.approx(5_000.0, abs=1.0)
    assert z[:, 0].std() == pytest.approx(25.0, rel=0.03)
    assert z[:, 1].mean() == pytest.approx(np.arctan2(4_000.0, 3_000.0), abs=1e-3)
    assert z[:, 1].std() == pytest.approx(0.01, rel=0.03)
    assert np.all(np.abs(z[:, 1]) <= np.pi)
