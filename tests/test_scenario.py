"""Validations du générateur de scénarios multi-cibles (jalon C)."""

import dataclasses

import numpy as np
import pytest

from sim.scenario import ScenarioParams, generate_scenario, load_scenario, save_scenario

PARAMS = ScenarioParams(n_ships=12, n_steps=200, seed=3)
SCENARIO = generate_scenario(PARAMS)


def test_shapes_and_ship_count_in_requested_range():
    """Vérité (n_ships, n_steps, 4), un tableau de détections polaires par balayage."""
    assert SCENARIO.truth.shape == (12, 200, 4)
    assert SCENARIO.in_turn.shape == (12, 200)
    assert len(SCENARIO.detections) == len(SCENARIO.sources) == 200
    for z, src in zip(SCENARIO.detections, SCENARIO.sources):
        assert z.shape == (len(src), 2)


def test_ships_stay_in_zone_and_away_from_radar():
    """Tous les navires restent dans la zone de 20 km et à plus de min_range du radar."""
    xy = SCENARIO.truth[..., :2]
    assert np.abs(xy).max() <= PARAMS.half_size
    assert np.hypot(xy[..., 0], xy[..., 1]).min() >= PARAMS.min_range


def test_empirical_detection_probability_matches_pd():
    """La fréquence de détection des navires vaut Pd = 0,9 (à 2 points près sur 2 400 occasions)."""
    detected = sum(np.sum(src >= 0) for src in SCENARIO.sources)
    assert detected / (PARAMS.n_ships * PARAMS.n_steps) == pytest.approx(0.9, abs=0.02)


def test_clutter_count_is_poisson():
    """Fausses alarmes par balayage : moyenne = variance = λ (propriété de la loi de Poisson)."""
    counts = np.array([np.sum(src < 0) for src in SCENARIO.sources])
    assert counts.mean() == pytest.approx(PARAMS.clutter_rate, rel=0.1)
    assert counts.var() == pytest.approx(PARAMS.clutter_rate, rel=0.25)


def test_clutter_is_uniform_in_the_zone():
    """Positions des fausses alarmes uniformes : moyenne ≈ 0 et écart-type ≈ L/√12 par axe."""
    z = np.vstack([zk[src < 0] for zk, src in zip(SCENARIO.detections, SCENARIO.sources)])
    xy = np.column_stack([z[:, 0] * np.cos(z[:, 1]), z[:, 0] * np.sin(z[:, 1])])
    side = 2 * PARAMS.half_size
    assert xy.mean(axis=0) == pytest.approx([0, 0], abs=0.03 * side)
    assert xy.std(axis=0) == pytest.approx([side / np.sqrt(12)] * 2, rel=0.05)
    assert np.abs(xy).max() <= PARAMS.half_size


def test_real_detections_are_polar_measurements_of_the_right_ship():
    """Une détection réelle est proche de la mesure attendue du navire désigné (bruit ≲ 5σ)."""
    for k in range(0, 200, 20):
        for z, ship in zip(SCENARIO.detections[k], SCENARIO.sources[k]):
            if ship < 0:
                continue
            x, y = SCENARIO.truth[ship, k, :2]
            assert abs(z[0] - np.hypot(x, y)) < 5 * PARAMS.sigma_r
            dtheta = np.angle(np.exp(1j * (z[1] - np.arctan2(y, x))))
            assert abs(dtheta) < 5 * PARAMS.sigma_theta


def test_turn_mask_matches_heading_changes():
    """Un navire marqué « en virage » change de cap ; un navire jamais marqué garde un cap quasi fixe."""
    heading = np.unwrap(np.arctan2(SCENARIO.truth[..., 3], SCENARIO.truth[..., 2]), axis=1)
    swing = heading.max(axis=1) - heading.min(axis=1)
    turning = SCENARIO.in_turn.any(axis=1)
    assert turning.any() and (~turning).any()
    min_swing = PARAMS.turn_rate_range[0] * PARAMS.turn_steps_range[0] * PARAMS.dt  # ω_min·durée_min
    assert swing[turning].min() > 0.9 * min_swing
    # sans virage, seul le bruit d'accélération fait dériver le cap (marche aléatoire de la vitesse,
    # ≈ σ_a·dt·√n / v ≈ 0,3 rad pour un navire lent) : bien moins que le virage voulu
    assert swing[~turning].max() < swing[turning].min()


def test_same_seed_same_scenario_different_seed_differs():
    """Graine fixée = scénario identique ; autre graine = scénario différent."""
    again = generate_scenario(PARAMS)
    np.testing.assert_array_equal(again.truth, SCENARIO.truth)
    for a, b in zip(again.detections, SCENARIO.detections):
        np.testing.assert_array_equal(a, b)
    other = generate_scenario(dataclasses.replace(PARAMS, seed=4))
    assert not np.array_equal(other.truth, SCENARIO.truth)


def test_save_and_load_round_trip(tmp_path):
    """Un scénario sérialisé puis relu est identique (paramètres, vérité, détections, sources)."""
    path = tmp_path / "scenario.npz"
    save_scenario(SCENARIO, path)
    loaded = load_scenario(path)
    assert loaded.params == SCENARIO.params
    np.testing.assert_array_equal(loaded.truth, SCENARIO.truth)
    np.testing.assert_array_equal(loaded.in_turn, SCENARIO.in_turn)
    assert len(loaded.detections) == len(SCENARIO.detections)
    for a, b in zip(loaded.detections, SCENARIO.detections):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(loaded.sources, SCENARIO.sources):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_ships", [5, 20])
def test_supports_5_to_20_ships(n_ships):
    """Le générateur produit bien 5 ou 20 navires dans la zone."""
    scenario = generate_scenario(dataclasses.replace(PARAMS, n_ships=n_ships, n_steps=60))
    assert scenario.truth.shape[0] == n_ships


@pytest.mark.parametrize("n_steps", [30, 40, 60])
def test_short_scenarios_have_consistent_trajectory_lengths(n_steps):
    """Régression : un virage plus long que le scénario ne doit pas produire de trajectoires de longueurs différentes."""
    for seed in range(20):
        params = dataclasses.replace(PARAMS, n_steps=n_steps, turn_fraction=1.0, seed=seed)
        scenario = generate_scenario(params)
        assert scenario.truth.shape == (params.n_ships, n_steps, 4)
        assert scenario.in_turn.shape == (params.n_ships, n_steps)
