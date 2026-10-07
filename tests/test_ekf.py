"""Validations de l'EKF polaire : jacobienne, normalisation d'angle, passage de θ = ±π (jalon A)."""

import numpy as np
import pytest

from sim.trajectories import (
    TrajectoryParams,
    cv_trajectory,
    polar_measurements,
    process_noise_cv,
    transition_cv,
)
from tracking.ekf import (
    ExtendedKalmanFilter,
    polar_h,
    polar_jacobian,
    polar_to_cartesian,
    wrap_angle,
)
from tracking.kf import run_filter

DT, SIGMA_A = 2.0, 0.05
SIGMA_R, SIGMA_THETA = 25.0, 0.01
R_POLAR = np.diag([SIGMA_R**2, SIGMA_THETA**2])
P0 = np.diag([100.0**2, 100.0**2, 5.0**2, 5.0**2])


@pytest.mark.parametrize(
    "state",
    [
        [3_000.0, 4_000.0, 5.0, -2.0],
        [-5_000.0, 1_000.0, 0.0, 0.0],
        [-800.0, -6_000.0, 3.0, 3.0],
        [12.0, -9_000.0, 1.0, 0.0],
    ],
)
def test_jacobian_matches_finite_differences(state):
    """La jacobienne analytique de h égale la différence finie centrale colonne par colonne."""
    x = np.array(state)
    h = 1e-3  # 1 mm : très petit devant les distances de plusieurs km
    numeric = np.zeros((2, 4))
    for j in range(4):
        step = np.zeros(4)
        step[j] = h
        diff = polar_h(x + step) - polar_h(x - step)
        diff[1] = wrap_angle(diff[1])
        numeric[:, j] = diff / (2 * h)
    np.testing.assert_allclose(polar_jacobian(x), numeric, rtol=1e-6, atol=1e-9)


def test_jacobian_undefined_at_origin():
    """À l'origine, l'angle est indéfini : on refuse plutôt que de diviser par zéro."""
    with pytest.raises(ValueError):
        polar_jacobian(np.zeros(4))


def test_wrap_angle_range_and_equivalence():
    """wrap_angle ramène dans [−π, π[ en préservant l'angle modulo 2π."""
    angles = np.linspace(-10, 10, 401)
    wrapped = wrap_angle(angles)
    assert np.all((wrapped >= -np.pi) & (wrapped < np.pi))
    np.testing.assert_allclose(np.cos(wrapped), np.cos(angles), atol=1e-12)
    np.testing.assert_allclose(np.sin(wrapped), np.sin(angles), atol=1e-12)


def test_innovation_angle_is_normalized_across_the_cut():
    """θ mesuré = π − 0,01 et θ prédit = −π + 0,01 : l'innovation vaut −0,02 rad, pas 2π − 0,02."""
    r = 5_000.0
    ekf = ExtendedKalmanFilter(
        transition_cv(DT), np.zeros((4, 4)), R_POLAR,
        np.concatenate([polar_to_cartesian(r, -np.pi + 0.01), [0.0, 0.0]]), P0,
    )
    y = ekf._innovation(np.array([r, np.pi - 0.01]))
    assert y[1] == pytest.approx(-0.02, abs=1e-9)
    assert abs(y[1]) < np.pi


def test_target_crossing_theta_pi_is_tracked_without_jump():
    """Cible qui coupe l'axe x négatif (θ passe de ≈ +π à ≈ −π) : l'estimation reste continue."""
    rng = np.random.default_rng(11)
    params = TrajectoryParams(dt=DT, n_steps=120, sigma_a=SIGMA_A)
    truth = cv_trajectory(np.array([-5_000.0, 1_000.0, 0.0, -10.0]), params, rng)
    z = polar_measurements(truth, SIGMA_R, SIGMA_THETA, rng)
    # la mesure franchit bien la coupure : θ passe d'un côté de ±π à l'autre
    assert z[:, 1].max() > 3.0 and z[:, 1].min() < -3.0

    x0 = truth[0] + np.linalg.cholesky(P0) @ rng.standard_normal(4)
    ekf = ExtendedKalmanFilter(
        transition_cv(DT), process_noise_cv(DT, SIGMA_A), R_POLAR, x0, P0
    )
    result = run_filter(ekf, z)

    # innovations d'angle toujours petites (quelques σ_θ), jamais ≈ 2π (sans normalisation : 6,28)
    assert np.nanmax(np.abs(result.innovations[:, 1])) < 10 * SIGMA_THETA
    # pas de saut d'estimation : la position estimée ne bouge jamais plus que la vérité + marge
    steps = np.linalg.norm(np.diff(result.x[:, :2], axis=0), axis=1)
    assert steps.max() < 150.0  # vitesse 10 m/s · 2 s = 20 m par pas ; 150 m = 6 σ_r de marge
    # et l'erreur de position reste de l'ordre du bruit de mesure
    err = np.linalg.norm(result.x[20:, :2] - truth[20:, :2], axis=1)
    assert err.max() < 250.0
