"""Validations de KalmanNet : taille, structure de Kalman, équivariance par rotation, différentiabilité (jalon D)."""

import numpy as np
import pytest
import torch

from learned.dataset import DatasetParams, make_splits, to_tensors
from learned.kalmannet import KalmanNet, KalmanNetFilter, KalmanNetParams, wrap_angle
from sim.trajectories import TrajectoryParams, cv_trajectory, polar_measurements, transition_cv

PARAMS = KalmanNetParams()
DATA = DatasetParams(n_train=8, n_val=4, n_test=40, n_steps=80, seed=2)
TEST = make_splits(DATA)["test"]


def random_net(seed=0, dtype=torch.float32, head_noise=0.3):
    """Réseau à poids aléatoires (tête perturbée pour que le gain dépende vraiment des entrées)."""
    torch.manual_seed(seed)
    net = KalmanNet(PARAMS)
    with torch.no_grad():
        net.head.weight.add_(head_noise * torch.randn_like(net.head.weight))
    return net.to(dtype)


def test_network_is_small():
    """Moins de 50 000 paramètres (consigne du jalon D)."""
    assert KalmanNet(PARAMS).n_parameters < 50_000


def test_forward_shape_and_initialisation_from_first_measurement():
    """Sortie (B, T, 4) ; l'instant 0 est la position de la première mesure, vitesse nulle."""
    _, z = to_tensors(TEST)
    est = random_net()(z)
    assert est.shape == (40, 80, 4)
    r, th = z[:, 0, 0], z[:, 0, 1]
    torch.testing.assert_close(est[:, 0, 0], r * torch.cos(th))
    torch.testing.assert_close(est[:, 0, 1], r * torch.sin(th))
    assert torch.all(est[:, 0, 2:] == 0)


def reference_step(x, z, K, dt=PARAMS.dt):
    """Pas de Kalman à gain K̃ (4×2, repère ligne de visée) calculé en numpy, indépendamment du réseau."""
    x_pred = transition_cv(dt) @ x
    r_hat, th = np.hypot(*x_pred[:2]), np.arctan2(x_pred[1], x_pred[0])
    d_r, d_th = z[0] - r_hat, np.angle(np.exp(1j * (z[1] - th)))
    c_los = K @ np.array([d_r, r_hat * d_th])
    c_los[2:] /= dt
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    correction = np.concatenate([R @ c_los[:2], R @ c_los[2:]])
    return x_pred + correction, correction


def run_network_step(net, x, z):
    """Un pas du réseau (double précision) depuis l'état x, sans correction précédente."""
    return net.step(
        torch.tensor(x[None]), torch.zeros(1, 4, dtype=torch.float64),
        torch.zeros(1, PARAMS.hidden_gru, dtype=torch.float64), torch.tensor(z[None]),
    )


X_STEP = np.array([3_000.0, -4_000.0, 6.0, 2.0])
Z_STEP = np.array([5_040.0, np.arctan2(-4_000.0, 3_000.0) + 0.012])


def test_update_equals_explicit_kalman_formula_with_the_prior_gain():
    """Sortie du réseau nulle : x = F·x + T(θ̂)·K̃₀·[δr, r̂·δθ] avec K̃₀ = a priori alpha-bêta, formule en numpy."""
    net = KalmanNet(PARAMS).double()
    with torch.no_grad():
        net.head.weight.zero_()
    a, b = PARAMS.gain_prior_pos, PARAMS.gain_prior_vel
    K0 = np.array([[a, 0], [0, a], [b, 0], [0, b]])
    x_net, correction, _ = run_network_step(net, X_STEP, Z_STEP)
    x_ref, corr_ref = reference_step(X_STEP, Z_STEP, K0)
    np.testing.assert_allclose(correction[0].detach().numpy(), corr_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(x_net[0].detach().numpy(), x_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("saturation", [+60.0, -60.0])
def test_gain_saturates_at_its_bounds(saturation):
    """Sortie très grande ou très négative : le gain vaut exactement la borne haute ou basse, et la formule reste juste."""
    net = KalmanNet(PARAMS).double()
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.fill_(saturation)
    bound = net.gain_hi if saturation > 0 else net.gain_lo
    K = bound.view(4, 2).numpy()
    x_net, _, _ = run_network_step(net, X_STEP, Z_STEP)
    np.testing.assert_allclose(x_net[0].detach().numpy(), reference_step(X_STEP, Z_STEP, K)[0], rtol=1e-8, atol=1e-8)


def test_gain_is_bounded_whatever_the_weights_so_the_filter_cannot_diverge():
    """Poids de tête très perturbés (σ = 1) : gains toujours dans [bas, haut], états bornés, aucun NaN."""
    net = random_net(head_noise=1.0)
    _, z = to_tensors(TEST)
    est = net(z)
    assert torch.isfinite(est).all()
    assert est.abs().max() < 5e4  # de l'ordre de la zone (±10 km) + marge, pas 1e14
    out = torch.randn(1000, 8) * 20
    gain = net.gain_from_output(out).view(-1, 8)
    assert torch.all(gain >= net.gain_lo - 1e-6) and torch.all(gain <= net.gain_hi + 1e-6)


@pytest.mark.parametrize("phi", [0.7, np.pi, -2.5])
def test_equivariance_by_rotation_around_the_radar(phi):
    """Tourner toute la scène de φ autour du radar tourne les estimations de φ (réseau à poids aléatoires)."""
    net = random_net(dtype=torch.float64)
    _, z = to_tensors(TEST, dtype=torch.float64)
    z_rot = z.clone()
    z_rot[..., 1] = wrap_angle(z[..., 1] + phi)
    est, est_rot = net(z), net(z_rot)
    c, s = np.cos(phi), np.sin(phi)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    expected = torch.cat([est[..., :2] @ R.T, est[..., 2:] @ R.T], dim=-1)
    torch.testing.assert_close(est_rot, expected, rtol=1e-8, atol=1e-6)


def test_gradients_reach_every_parameter():
    """Une perte de position rétropropage un gradient fini et non nul vers tous les paramètres."""
    net = random_net()
    truth, z = to_tensors(TEST)
    loss = ((net(z)[..., :2] - truth[..., :2]) ** 2).mean()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_single_target_wrapper_matches_batched_forward():
    """KalmanNetFilter (batch de 1, interface predict/update) donne les mêmes estimations que forward."""
    net = random_net().eval()
    _, z = to_tensors(TEST)
    with torch.no_grad():
        reference = net(z[:1])[0].numpy()
    filt = KalmanNetFilter(net, z[0, 0].numpy())
    for k in range(1, 30):
        filt.predict()
        filt.update(z[0, k].numpy())
        np.testing.assert_allclose(filt.x, reference[k], rtol=1e-4, atol=1e-2)


def test_untrained_network_is_already_a_stable_filter():
    """Avant tout entraînement, l'a priori alpha-bêta donne une erreur de position inférieure à celle des mesures."""
    truth, z = to_tensors(TEST)
    est = KalmanNet(PARAMS)(z)
    err_filter = ((est[:, 10:, :2] - truth[:, 10:, :2]) ** 2).sum(-1).mean().sqrt()
    meas_xy = torch.stack([z[..., 0] * torch.cos(z[..., 1]), z[..., 0] * torch.sin(z[..., 1])], dim=-1)
    err_meas = ((meas_xy[:, 10:] - truth[:, 10:, :2]) ** 2).sum(-1).mean().sqrt()
    assert err_filter < 0.9 * err_meas, f"filtre {err_filter:.1f} m, mesures {err_meas:.1f} m"


def test_estimate_is_continuous_across_theta_pi():
    """Cible qui coupe le demi-axe x négatif (θ : +π → −π) : aucun saut d'estimation (angle normalisé)."""
    rng = np.random.default_rng(11)
    truth = cv_trajectory(np.array([-5_000.0, 1_000.0, 0.0, -10.0]), TrajectoryParams(dt=2.0, n_steps=120), rng)
    z = torch.as_tensor(polar_measurements(truth, 25.0, 0.01, rng), dtype=torch.float32)[None]
    assert z[..., 1].max() > 3.0 and z[..., 1].min() < -3.0  # la mesure franchit bien la coupure
    est = KalmanNet(PARAMS)(z)[0].detach().numpy()
    assert np.linalg.norm(np.diff(est[:, :2], axis=0), axis=1).max() < 150.0
    assert np.linalg.norm(est[20:, :2] - truth[20:, :2], axis=1).max() < 250.0


def test_wrap_angle_range_and_gradient():
    """wrap_angle ∈ [−π, π], égale a modulo 2π, avec un gradient 1 (différentiable)."""
    a = torch.linspace(-10, 10, 201, requires_grad=True)
    w = wrap_angle(a)
    assert torch.all(w.abs() <= np.pi + 1e-6)
    torch.testing.assert_close(torch.cos(w), torch.cos(a))
    w.sum().backward()
    torch.testing.assert_close(a.grad, torch.ones_like(a))
