"""Validations de l'IMM : comparaison à filterpy, probabilités de modèle, gain en virage (jalon C)."""

import numpy as np
import pytest
from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter as FilterpyKF

from metrics.nees import chi2_interval, nees
from metrics.rmse import position_rmse
from sim.trajectories import (
    TrajectoryParams,
    cartesian_measurements,
    maneuver_trajectory,
    polar_measurements,
    process_noise_cv,
    transition_ct,
    transition_cv,
)
from tracking.ekf import ExtendedKalmanFilter
from tracking.imm import IMM, build_imm, transition_matrix
from tracking.kf import KalmanFilter, run_monte_carlo

DT, SIGMA_A, SIGMA_R, SIGMA_THETA = 2.0, 0.05, 25.0, 0.01
R_POLAR = np.diag([SIGMA_R**2, SIGMA_THETA**2])
P0 = np.diag([100.0**2, 100.0**2, 5.0**2, 5.0**2])
X0 = np.array([-6_000.0, 3_000.0, 0.0, -15.0])
OMEGA = 0.02
TURN_START, TURN_END = 40, 90  # transitions en virage : pas 41 à 90
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def maneuver(sign, rng):
    """Vérité CV (40 pas), virage de ±ω (50 pas), CV (40 pas) et mesures polaires bruitées."""
    segments = [(40, 0.0), (50, sign * OMEGA), (40, 0.0)]
    truth = maneuver_trajectory(X0, segments, TrajectoryParams(dt=DT, sigma_a=SIGMA_A), rng)
    return truth, polar_measurements(truth, SIGMA_R, SIGMA_THETA, rng)


def a_priori(rng, truth):
    """État initial tiré selon N(x_vrai, P0) pour que P0 soit honnête."""
    return truth[0] + np.linalg.cholesky(P0) @ rng.standard_normal(4)


def make_ekf(rng, truth):
    """EKF à modèle CV seul (référence)."""
    return ExtendedKalmanFilter(
        transition_cv(DT), process_noise_cv(DT, SIGMA_A), R_POLAR, a_priori(rng, truth), P0
    )


def make_imm(turn_rates):
    """Fabrique d'IMM (CV + un CT par taux) pour run_monte_carlo."""
    return lambda rng, truth: build_imm(a_priori(rng, truth), P0, DT, SIGMA_A, R_POLAR, turn_rates)


def monte_carlo(sign, make_filter, n_runs=40, seed=0):
    return run_monte_carlo(n_runs, lambda rng: maneuver(sign, rng), make_filter, seed)


def turn_rmse(mc):
    """RMSE de position pendant le virage (pas 41 à 90), moyenne quadratique sur les tirages."""
    window = slice(TURN_START + 1, TURN_END + 1)
    return position_rmse(mc.x_true[:, window], mc.x[:, window])


def test_matches_filterpy_imm_estimator_on_one_target():
    """Même modèles, mêmes mesures cartésiennes : x, P et μ égaux à ceux de filterpy après chaque mise à jour."""
    rng = np.random.default_rng(4)
    truth = maneuver_trajectory(
        X0, [(30, 0.0), (40, OMEGA), (30, 0.0)], TrajectoryParams(dt=DT, sigma_a=SIGMA_A), rng
    )
    z = cartesian_measurements(truth, 30.0, rng)
    R = 30.0**2 * np.eye(2)
    x0 = truth[0] + np.linalg.cholesky(P0) @ rng.standard_normal(4)
    Fs, Q = [transition_cv(DT), transition_ct(DT, OMEGA)], process_noise_cv(DT, SIGMA_A)
    mu0, pi = np.array([0.9, 0.1]), transition_matrix(2, 0.95)

    mine = IMM([KalmanFilter(F, Q, H, R, x0, P0) for F in Fs], mu0, pi)
    ref_filters = []
    for F in Fs:
        f = FilterpyKF(dim_x=4, dim_z=2)
        f.F, f.Q, f.H, f.R, f.x, f.P = F, Q, H, R, x0.copy(), P0.copy()
        ref_filters.append(f)
    ref = IMMEstimator(ref_filters, mu0.copy(), pi)

    for k in range(1, len(z)):
        mine.predict()
        mine.update(z[k])
        ref.predict()
        ref.update(z[k])
        np.testing.assert_allclose(mine.x, ref.x.ravel(), rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(mine.P, ref.P, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(mine.mu, ref.mu, rtol=1e-9, atol=1e-12)


def test_single_model_imm_is_the_plain_filter():
    """Un IMM à un seul modèle (π = [[1]]) redonne exactement l'EKF correspondant."""
    rng = np.random.default_rng(1)
    truth, z = maneuver(+1, rng)
    x0 = a_priori(rng, truth)
    make = lambda: ExtendedKalmanFilter(
        transition_cv(DT), process_noise_cv(DT, SIGMA_A), R_POLAR, x0, P0
    )
    imm, ekf = IMM([make()], [1.0], [[1.0]]), make()
    for k in range(1, 50):
        imm.predict(), ekf.predict()
        imm.update(z[k]), ekf.update(z[k])
        np.testing.assert_allclose(imm.x, ekf.x, atol=1e-9)
        np.testing.assert_allclose(imm.P, ekf.P, atol=1e-9)


def test_model_probabilities_stay_valid_even_after_an_outlier():
    """Σ μ_j = 1 et μ_j ∈ [0, 1] à chaque pas, sans NaN, même après une mesure aberrante de 3 km.

    L'a posteriori peut sous-dépasser à 0 (vraisemblance ~10⁻²⁰⁰) : c'est légitime. Ce qui doit rester vrai,
    c'est que la probabilité prédite c̄_j = Σ π_ij·μ_i soit > 0 (π_ij > 0), sinon log(μ_pred) = −∞.
    """
    rng = np.random.default_rng(2)
    truth, z = maneuver(+1, rng)
    imm = build_imm(a_priori(rng, truth), P0, DT, SIGMA_A, R_POLAR, (OMEGA, -OMEGA))
    for k in range(1, 130):
        imm.predict()
        assert np.all(imm.mu_pred > 0)
        imm.update(z[k] + (np.array([3_000.0, 0.5]) if k == 60 else 0.0))
        assert not np.isnan(imm.mu).any() and not np.isnan(imm.x).any()
        assert imm.mu.sum() == pytest.approx(1.0)
        assert np.all((imm.mu >= 0) & (imm.mu <= 1))
    # le filtre se remet de l'aberrante : erreur finale de l'ordre du bruit
    assert np.linalg.norm(imm.x[:2] - truth[129, :2]) < 150.0


def test_predict_without_measurement_is_the_mixture_of_predictions():
    """Sans mise à jour (raté de détection), x = Σ c̄_j·x_j et μ = c̄ (probabilités de la chaîne de Markov)."""
    rng = np.random.default_rng(3)
    truth, z = maneuver(+1, rng)
    imm = build_imm(a_priori(rng, truth), P0, DT, SIGMA_A, R_POLAR, (OMEGA,))
    for k in range(1, 10):
        imm.predict()
        imm.update(z[k])
    c_bar = imm.pi.T @ imm.mu
    imm.predict()
    np.testing.assert_allclose(imm.mu, c_bar)
    np.testing.assert_allclose(imm.x, sum(w * f.x for w, f in zip(c_bar, imm.filters)), atol=1e-9)


def test_mode_probability_follows_the_maneuver():
    """μ(CT) est faible en ligne droite et élevé pendant le virage (moyenne sur 40 tirages)."""
    mu_ct = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        truth, z = maneuver(+1, rng)
        imm = build_imm(a_priori(rng, truth), P0, DT, SIGMA_A, R_POLAR, (OMEGA,))
        history = [imm.mu[1]]
        for k in range(1, len(z)):
            imm.predict()
            imm.update(z[k])
            history.append(imm.mu[1])
        mu_ct.append(history)
    mu_ct = np.mean(mu_ct, axis=0)
    assert mu_ct[10:36].mean() < 0.3  # ligne droite avant le virage
    assert mu_ct[50:86].mean() > 0.7  # en plein virage
    assert mu_ct[110:].mean() < 0.3  # ligne droite après le virage


def test_imm_clearly_beats_cv_filter_in_a_turn():
    """Virage de ±ω : l'IMM CV + CT(±ω) divise l'erreur en virage de l'EKF CV par plus de 2 (mesuré : ≈ 6,5)."""
    for sign in (+1, -1):
        rmse_cv = turn_rmse(monte_carlo(sign, make_ekf))
        rmse_imm = turn_rmse(monte_carlo(sign, make_imm((OMEGA, -OMEGA))))
        assert rmse_imm < 0.5 * rmse_cv, f"sens {sign:+d} : IMM {rmse_imm:.0f} m, EKF-CV {rmse_cv:.0f} m"


def test_imm_costs_little_on_a_straight_line():
    """En ligne droite l'IMM ne perd presque rien face à l'EKF CV (au plus +15 % ; mesuré : ≈ +5 %)."""
    straight = lambda rng: (
        lambda t: (t, polar_measurements(t, SIGMA_R, SIGMA_THETA, rng))
    )(maneuver_trajectory(X0, [(130, 0.0)], TrajectoryParams(dt=DT, sigma_a=SIGMA_A), rng))
    rmse = {}
    for name, make in [("cv", make_ekf), ("imm", make_imm((OMEGA, -OMEGA)))]:
        mc = run_monte_carlo(40, straight, make, seed=0)
        rmse[name] = position_rmse(mc.x_true, mc.x, skip=10)
    assert rmse["imm"] < 1.15 * rmse["cv"]


def test_two_model_imm_only_helps_for_the_turn_direction_it_knows():
    """Limite de l'IMM à 2 modèles : son CT(+ω) aide peu pour un virage de −ω (erreur > 2× celle de l'IMM à 3 modèles)."""
    wrong_way = turn_rmse(monte_carlo(-1, make_imm((OMEGA,))))
    both_ways = turn_rmse(monte_carlo(-1, make_imm((OMEGA, -OMEGA))))
    right_way = turn_rmse(monte_carlo(+1, make_imm((OMEGA,))))
    assert wrong_way > 2 * both_ways
    assert right_way < 0.5 * wrong_way


def test_imm_is_consistent_in_a_turn_where_the_cv_filter_is_not():
    """NEES en virage : l'IMM reste dans l'intervalle χ² à 95 %, l'EKF CV est à plus de 50 fois la borne."""
    window = slice(TURN_START + 1, TURN_END + 1)
    lo, hi = chi2_interval(4, 100)
    mean_nees = {}
    for name, make in [("cv", make_ekf), ("imm", make_imm((OMEGA,)))]:
        mc = monte_carlo(+1, make, n_runs=100, seed=5)
        mean_nees[name] = nees(mc.x_true, mc.x, mc.P).mean(axis=0)[window].mean()
    assert lo <= mean_nees["imm"] <= hi * 1.1  # 10 % de marge : 50 instants corrélés, pas indépendants
    assert mean_nees["cv"] > 50 * hi
