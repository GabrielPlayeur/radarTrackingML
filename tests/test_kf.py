"""Validations du filtre de Kalman linéaire écrit à la main (jalon A)."""

import numpy as np
import pytest
from filterpy.kalman import KalmanFilter as FilterpyKF

from metrics.rmse import position_rmse
from sim.trajectories import (
    TrajectoryParams,
    cartesian_measurements,
    cv_trajectory,
    process_noise_cv,
    transition_cv,
)
from tracking.kf import KalmanFilter, run_filter

DT, SIGMA_A, SIGMA_Z = 2.0, 0.05, 30.0
X0 = np.array([0.0, 0.0, 8.0, -5.0])
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = SIGMA_Z**2 * np.eye(2)
P0 = np.diag([100.0**2, 100.0**2, 5.0**2, 5.0**2])


def make_scenario(seed=0, n_steps=200):
    """Trajectoire CV bruitée par σ_a, mesures cartésiennes σ_z, état initial a priori décalé."""
    rng = np.random.default_rng(seed)
    params = TrajectoryParams(dt=DT, n_steps=n_steps, sigma_a=SIGMA_A)
    truth = cv_trajectory(X0, params, rng)
    z = cartesian_measurements(truth, SIGMA_Z, rng)
    x0_est = truth[0] + np.linalg.cholesky(P0) @ rng.standard_normal(4)
    return truth, z, x0_est


def make_kf(x0_est):
    """Filtre avec le Q et le R corrects pour le scénario ci-dessus."""
    return KalmanFilter(transition_cv(DT), process_noise_cv(DT, SIGMA_A), H, R, x0_est, P0)


def test_filter_rmse_is_below_measurement_rmse():
    """Avec Q et R corrects sur une trajectoire CV, le filtre est plus précis que les mesures."""
    truth, z, x0_est = make_scenario()
    result = run_filter(make_kf(x0_est), z)
    skip = 20  # on laisse converger l'initialisation
    rmse_filter = position_rmse(truth, result.x, skip)
    rmse_measure = float(np.sqrt(np.mean(np.sum((truth[skip:, :2] - z[skip:]) ** 2, axis=1))))
    assert rmse_measure == pytest.approx(SIGMA_Z * np.sqrt(2), rel=0.15)  # bruit bien réglé
    assert rmse_filter < 0.6 * rmse_measure


def test_matches_filterpy_to_1e_9():
    """Mêmes matrices, mêmes mesures : x et P identiques à ceux de filterpy à 1e-9 près."""
    _, z, x0_est = make_scenario(seed=3, n_steps=60)
    mine = run_filter(make_kf(x0_est), z)

    ref = FilterpyKF(dim_x=4, dim_z=2)
    ref.F, ref.Q = transition_cv(DT), process_noise_cv(DT, SIGMA_A)
    ref.H, ref.R, ref.x, ref.P = H, R, x0_est.copy(), P0.copy()
    for k in range(1, len(z)):
        ref.predict()
        ref.update(z[k])
        np.testing.assert_allclose(mine.x[k], ref.x.ravel(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine.P[k], ref.P, rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine.innovations[k], ref.y.ravel(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine.S[k], ref.S, rtol=0, atol=1e-9)


def test_predict_grows_covariance_and_update_shrinks_it():
    """predict() augmente l'incertitude (P⁻ = F·P·Fᵀ + Q), update() la réduit."""
    _, z, x0_est = make_scenario(n_steps=5)
    kf = make_kf(x0_est)
    trace_before = np.trace(kf.P)
    kf.predict()
    trace_predicted = np.trace(kf.P)
    kf.update(z[1])
    assert trace_predicted > trace_before
    assert np.trace(kf.P) < trace_predicted


def test_covariance_stays_symmetric_positive_definite():
    """La forme de Joseph garde P symétrique et définie positive sur 200 pas."""
    _, z, x0_est = make_scenario()
    result = run_filter(make_kf(x0_est), z)
    for P in result.P:
        np.testing.assert_allclose(P, P.T, atol=1e-9)
        assert np.linalg.eigvalsh(P).min() > 0
