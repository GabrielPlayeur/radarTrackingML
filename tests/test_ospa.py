"""Validations de l'OSPA et des statistiques de pistes (jalon C)."""

import numpy as np
import pytest

from metrics.ospa import ospa, ospa_components, ospa_over_time
from metrics.rmse import assign_truth_to_tracks, track_statistics

C = 300.0
POINTS = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 2_000.0]])


def test_identical_sets_have_zero_distance():
    """OSPA(X, X) = 0, quel que soit l'ordre des points."""
    assert ospa(POINTS, POINTS, C) == 0.0
    assert ospa(POINTS, POINTS[::-1], C) == pytest.approx(0.0)


def test_empty_versus_non_empty_is_c_and_both_empty_is_zero():
    """Un ensemble vide contre un non vide vaut exactement c (dans les deux sens) ; deux vides : 0."""
    empty = np.zeros((0, 2))
    assert ospa(empty, POINTS, C) == C
    assert ospa(POINTS, empty, C) == C
    assert ospa(empty, empty, C) == 0.0
    assert ospa(empty, POINTS[:1], C, p=1.0) == C


def test_single_point_shift_below_cutoff_is_the_distance():
    """Un point décalé de d < c donne exactement d ; au-delà de c, la distance est plafonnée à c."""
    assert ospa(np.array([[0.0, 0.0]]), np.array([[30.0, 40.0]]), C) == pytest.approx(50.0)
    assert ospa(np.array([[0.0, 0.0]]), np.array([[3_000.0, 0.0]]), C) == pytest.approx(C)


def test_cardinality_error_formula():
    """Un point apparié (erreur d) + un point en trop : OSPA = ((d² + c²)/2)^(1/2) pour p = 2 ; (d + c)/2 pour p = 1."""
    X = np.array([[0.0, 0.0]])
    Y = np.array([[30.0, 40.0], [5_000.0, 5_000.0]])
    assert ospa(X, Y, C, p=2.0) == pytest.approx(np.sqrt((50.0**2 + C**2) / 2))
    assert ospa(X, Y, C, p=1.0) == pytest.approx((50.0 + C) / 2)


def test_symmetry_and_bound():
    """OSPA est symétrique et ne dépasse jamais c (sur 100 ensembles aléatoires)."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        X = rng.uniform(-2_000, 2_000, (rng.integers(0, 6), 2))
        Y = rng.uniform(-2_000, 2_000, (rng.integers(0, 6), 2))
        d = ospa(X, Y, C)
        assert d == pytest.approx(ospa(Y, X, C))
        assert 0.0 <= d <= C + 1e-9


def test_components_add_up():
    """total^p = localisation^p + cardinalité^p ; sans erreur de cardinalité, la composante vaut 0."""
    X = POINTS + np.array([10.0, 0.0])
    total, loc, card = ospa_components(X, POINTS, C)
    assert card == 0.0 and loc == pytest.approx(10.0) and total == pytest.approx(10.0)
    total, loc, card = ospa_components(POINTS[:2], POINTS, C)
    assert total**2 == pytest.approx(loc**2 + card**2)
    assert card == pytest.approx(C * np.sqrt(1 / 3))


def test_optimal_assignment_not_index_order():
    """L'appariement est optimal : permuter les points ne change rien, même s'ils sont croisés."""
    X = np.array([[0.0, 0.0], [100.0, 0.0]])
    Y = np.array([[102.0, 0.0], [-1.0, 0.0]])  # même paires, ordre inversé
    assert ospa(X, Y, C) == pytest.approx(np.sqrt((1.0**2 + 2.0**2) / 2))


def test_ospa_over_time_shape_and_values():
    """Une valeur par balayage : 0 si identiques, c si les pistes sont absentes."""
    truth = [POINTS, POINTS]
    tracks = [POINTS, np.zeros((0, 2))]
    np.testing.assert_allclose(ospa_over_time(truth, tracks, C), [0.0, C])


# ---------------------------------------------------------------------------------- statistiques de pistes
def synthetic(n_ships=2, n_steps=10):
    """Cibles immobiles espacées de 5 km sur l'axe x (la première en (0, 0)) ; un état de piste = [x, y, 0, 0]."""
    truth = np.zeros((n_ships, n_steps, 4))
    truth[:, :, 0] = 5_000.0 * np.arange(n_ships)[:, None]
    return truth


def perfect_run(truth):
    """Pistes parfaites : la piste s suit la cible s sous l'identifiant s."""
    n_ships, n_steps = truth.shape[:2]
    return [np.arange(n_ships)] * n_steps, [truth[:, k, :] for k in range(n_steps)]


def test_perfect_tracking_has_no_error_loss_or_false_track():
    """Suivi parfait : RMSE 0, aucune perte, aucun changement d'identité, aucune fausse piste."""
    truth = synthetic()
    stats = track_statistics(truth, *perfect_run(truth))
    assert stats.rmse == 0.0 and stats.n_losses == 0 and stats.n_id_switches == 0
    assert stats.n_false_tracks == 0 and stats.n_never_tracked == 0 and stats.tracked_fraction == 1.0


def test_rmse_of_constant_offset_and_masked_rmse():
    """Piste décalée de 10 m (resp. 40 m en virage) : RMSE globale et RMSE sur le masque correctes."""
    truth = synthetic(n_ships=1)
    ids, states = perfect_run(truth)
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 5:] = True
    states = [s + np.array([[10.0 if k < 5 else 40.0, 0, 0, 0]]) for k, s in enumerate(states)]
    stats = track_statistics(truth, ids, states, mask=mask)
    assert stats.rmse == pytest.approx(np.sqrt((10.0**2 + 40.0**2) / 2))
    assert stats.rmse_masked == pytest.approx(40.0)


def test_track_dropped_for_a_while_counts_one_loss():
    """Cible 0 sans piste aux balayages 4 à 6 : une perte de piste (tracked → untracked), pas trois."""
    truth = synthetic()
    ids, states = perfect_run(truth)
    for k in (4, 5, 6):
        ids[k], states[k] = np.array([1]), truth[1:, k, :]
    stats = track_statistics(truth, ids, states)
    assert stats.n_losses == 1 and stats.n_id_switches == 0
    assert stats.tracked_fraction == pytest.approx(17 / 20)


def test_identity_change_counts_one_switch_not_a_loss():
    """La piste qui suit la cible 0 change d'identifiant au balayage 5 : un changement d'identité."""
    truth = synthetic()
    ids, states = perfect_run(truth)
    for k in range(5, 10):
        ids[k] = np.array([7, 1])
    stats = track_statistics(truth, ids, states)
    assert stats.n_id_switches == 1 and stats.n_losses == 0


def test_far_track_is_a_false_track_and_untracked_ship_is_reported():
    """Une piste confirmée à 3 km de toute cible est fausse ; une cible sans aucune piste est « jamais pistée »."""
    truth = synthetic()
    ids = [np.array([0, 9])] * 10
    states = [np.array([truth[0, k], [2_500.0, 3_000.0, 0, 0]]) for k in range(10)]
    stats = track_statistics(truth, ids, states)
    assert stats.n_false_tracks == 1
    assert stats.n_never_tracked == 1  # la cible 1 n'a jamais de piste


def test_assignment_respects_max_distance():
    """Une piste au-delà de max_dist n'est pas associée ; deux cibles proches se répartissent deux pistes."""
    truth_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    np.testing.assert_array_equal(assign_truth_to_tracks(truth_xy, np.array([[900.0, 0.0]]), 300.0), [-1, -1])
    np.testing.assert_array_equal(
        assign_truth_to_tracks(truth_xy, np.array([[110.0, 0.0], [-5.0, 0.0]]), 300.0), [1, 0]
    )
