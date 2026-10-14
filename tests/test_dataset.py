"""Validations du jeu de données du gain appris (jalon D)."""

import dataclasses

import numpy as np
import pytest
import torch

from learned.dataset import (
    STRESS_KINDS,
    DatasetParams,
    generate_batch,
    make_splits,
    make_stress_test_set,
    stress_params,
    to_tensors,
)

PARAMS = DatasetParams(n_train=60, n_val=20, n_test=30, n_steps=80, seed=5)
SPLITS = make_splits(PARAMS)


def test_shapes_of_each_split():
    """Chaque découpage a la bonne taille ; vérité (N, T, 4), mesures (N, T, 2), masque (N, T)."""
    for name, n in [("train", 60), ("val", 20), ("test", 30)]:
        batch = SPLITS[name]
        assert batch.truth.shape == (n, 80, 4)
        assert batch.z.shape == (n, 80, 2)
        assert batch.in_turn.shape == (n, 80)


def test_splits_are_reproducible_and_mutually_different():
    """Même graine = mêmes données ; train, val et test viennent de flux indépendants (aucun recouvrement)."""
    again = make_splits(PARAMS)
    for name in SPLITS:
        np.testing.assert_array_equal(again[name].truth, SPLITS[name].truth)
        np.testing.assert_array_equal(again[name].z, SPLITS[name].z)
    starts = {name: {tuple(np.round(t[0, :2], 3)) for t in b.truth} for name, b in SPLITS.items()}
    assert not (starts["train"] & starts["val"]) and not (starts["train"] & starts["test"])
    assert not (starts["val"] & starts["test"])
    other = make_splits(dataclasses.replace(PARAMS, seed=6))
    assert not np.array_equal(other["train"].truth, SPLITS["train"].truth)


def test_trajectories_stay_in_zone_and_away_from_radar():
    """Toutes les trajectoires restent dans la zone et à plus de min_range du radar."""
    xy = SPLITS["train"].truth[..., :2]
    assert np.abs(xy).max() <= PARAMS.half_size
    assert np.hypot(xy[..., 0], xy[..., 1]).min() >= PARAMS.min_range


def test_turn_probability_and_mask_consistency():
    """≈ la moitié des trajectoires virent (turn_probability = 0,5) et le masque est vide sinon."""
    batch = make_splits(dataclasses.replace(PARAMS, n_train=400))["train"]
    turning = batch.in_turn.any(axis=1)
    assert turning.mean() == pytest.approx(0.5, abs=0.08)
    assert not batch.in_turn[~turning].any()


def test_measurement_noise_matches_sigma():
    """Écart mesure − vérité polaire : σ_r en distance, σ_θ en angle (à 5 % près sur 12 000 points)."""
    batch = make_splits(dataclasses.replace(PARAMS, n_train=150))["train"]
    r_true = np.hypot(batch.truth[..., 0], batch.truth[..., 1])
    theta_true = np.arctan2(batch.truth[..., 1], batch.truth[..., 0])
    dr = batch.z[..., 0] - r_true
    dtheta = np.angle(np.exp(1j * (batch.z[..., 1] - theta_true)))
    assert dr.std() == pytest.approx(PARAMS.sigma_r, rel=0.05)
    assert dtheta.std() == pytest.approx(PARAMS.sigma_theta, rel=0.05)


@pytest.mark.parametrize("kind", STRESS_KINDS)
def test_stress_sets_differ_from_training_distribution_as_announced(kind):
    """Chaque condition hors distribution modifie bien le paramètre annoncé."""
    stressed = stress_params(PARAMS, kind)
    if kind == "strong_turns":
        assert stressed.turn_rate_range[0] > 1.5 * PARAMS.turn_rate_range[1]
        assert stressed.turn_probability == 1.0
    if kind == "noisy":
        assert stressed.sigma_r == 2 * PARAMS.sigma_r and stressed.sigma_theta == 2 * PARAMS.sigma_theta
    if kind == "long":
        assert stressed.n_steps == 4 * PARAMS.n_steps


def test_strong_turn_set_really_turns_faster():
    """Dans la condition « virages forts », la vitesse angulaire du cap dépasse ce qui existe à l'entraînement."""
    small = dataclasses.replace(PARAMS, n_test=40)
    strong = make_stress_test_set(small, "strong_turns")

    def max_turn_rate(batch):
        heading = np.unwrap(np.arctan2(batch.truth[..., 3], batch.truth[..., 2]), axis=1)
        return np.abs(np.diff(heading, axis=1)).max(axis=1) / small.dt

    assert strong.in_turn.any(axis=1).all()
    assert max_turn_rate(strong).min() > PARAMS.turn_rate_range[1]
    assert np.median(max_turn_rate(SPLITS["train"])) < 0.05  # entraînement : surtout lent


def test_noisy_set_has_twice_the_noise_and_long_set_is_longer():
    """Bruit doublé (σ_r ×2) et séquences 4 fois plus longues (vitesses biaisées, voir stress_params)."""
    small = dataclasses.replace(PARAMS, n_test=60)
    noisy = make_stress_test_set(small, "noisy")
    dr = noisy.z[..., 0] - np.hypot(noisy.truth[..., 0], noisy.truth[..., 1])
    assert dr.std() == pytest.approx(2 * PARAMS.sigma_r, rel=0.06)
    assert make_stress_test_set(dataclasses.replace(small, n_test=5), "long").truth.shape == (5, 320, 4)


def test_to_tensors_dtype_and_shapes():
    """Conversion en tenseurs torch float32 de mêmes formes."""
    truth, z = to_tensors(SPLITS["test"])
    assert truth.dtype == torch.float32 and truth.shape == (30, 80, 4) and z.shape == (30, 80, 2)
    truth64, _ = to_tensors(SPLITS["test"], dtype=torch.float64)
    assert truth64.dtype == torch.float64


def test_generate_batch_is_deterministic_given_rng():
    """generate_batch ne dépend que du générateur passé."""
    a = generate_batch(PARAMS, 5, np.random.default_rng(0))
    b = generate_batch(PARAMS, 5, np.random.default_rng(0))
    np.testing.assert_array_equal(a.z, b.z)
