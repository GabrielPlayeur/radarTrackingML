"""Validations de la gestion de pistes : initialisation, confirmation M/N, suppression (jalon C)."""

import numpy as np
import pytest

from sim.scenario import ScenarioParams, generate_scenario
from sim.trajectories import TrajectoryParams, cv_trajectory, polar_measurements
from tracking.track_manager import (
    TrackManager,
    TrackManagerParams,
    initial_state_from_detection,
    make_ekf_factory,
    run_tracking,
)

DT, SIGMA_A, SIGMA_R, SIGMA_THETA = 2.0, 0.05, 25.0, 0.01
EMPTY = np.zeros((0, 2))


def new_manager(**kwargs):
    """Gestionnaire à EKF CV, règles M/N = 3/4 et K = 5 par défaut."""
    factory = make_ekf_factory(DT, SIGMA_A, SIGMA_R, SIGMA_THETA)
    return TrackManager(factory, TrackManagerParams(**kwargs))


def target_detections(n_steps, seed=0):
    """Une cible CV bruitée : vérité et une détection polaire par balayage (Pd = 1)."""
    rng = np.random.default_rng(seed)
    truth = cv_trajectory(
        np.array([4_000.0, 3_000.0, -8.0, 5.0]), TrajectoryParams(dt=DT, n_steps=n_steps), rng
    )
    return truth, [z[None, :] for z in polar_measurements(truth, SIGMA_R, SIGMA_THETA, rng)]


def test_initial_state_from_detection():
    """Position = conversion polaire → cartésien ; vitesse nulle d'écart-type σ_v ; covariance SPD."""
    x0, P0 = initial_state_from_detection(np.array([5_000.0, np.pi / 4]), 25.0, 0.01, 10.0)
    np.testing.assert_allclose(x0, [5_000 / np.sqrt(2), 5_000 / np.sqrt(2), 0.0, 0.0])
    np.testing.assert_allclose(P0[2:, 2:], 100.0 * np.eye(2))
    assert np.linalg.eigvalsh(P0).min() > 0
    # le long de la ligne de visée l'incertitude vaut σ_r, en travers r·σ_θ
    los = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    cross = np.array([-los[1], los[0]])
    assert np.sqrt(los @ P0[:2, :2] @ los) == pytest.approx(25.0)
    assert np.sqrt(cross @ P0[:2, :2] @ cross) == pytest.approx(5_000 * 0.01)


def test_track_is_confirmed_after_exactly_M_detections():
    """Avec 3/4 et une détection par balayage : provisoire aux balayages 1 et 2, confirmée au 3ᵉ."""
    _, dets = target_detections(6)
    manager = new_manager()
    confirmed = []
    for z in dets:
        manager.step(z)
        confirmed.append(len(manager.confirmed_tracks))
    assert confirmed == [0, 0, 1, 1, 1, 1]
    assert len(manager.tracks) == 1


def test_confirmed_track_keeps_its_id_and_follows_the_target():
    """Cible détectée à chaque balayage : une seule piste, même identifiant, erreur de l'ordre du bruit."""
    truth, dets = target_detections(120)
    run = run_tracking(new_manager(), dets)
    ids = {int(i) for arr in run.ids for i in arr}
    assert len(ids) == 1
    err = [np.linalg.norm(s[0, :2] - truth[k, :2]) for k, s in enumerate(run.states) if len(s)]
    assert np.mean(err[30:]) < 40.0


def test_pattern_with_one_miss_in_window_still_confirms():
    """Détections aux balayages 0, 1, 3 (raté au 2) : 3 détections parmi 4 → confirmée au balayage 3."""
    _, dets = target_detections(5)
    manager = new_manager()
    pattern = [True, True, False, True]
    states = []
    for hit, z in zip(pattern, dets):
        manager.step(z if hit else EMPTY)
        states.append(len(manager.confirmed_tracks))
    assert states == [0, 0, 0, 1]


def test_isolated_detection_is_dropped_when_M_becomes_unreachable():
    """Détection unique suivie de rien : [T], [T,F] (3 encore possible), [T,F,F] → supprimée."""
    _, dets = target_detections(1)
    manager = new_manager()
    sizes = []
    manager.step(dets[0])
    sizes.append(len(manager.tracks))
    for _ in range(3):
        manager.step(EMPTY)
        sizes.append(len(manager.tracks))
    assert sizes == [1, 1, 0, 0]


def test_confirmed_track_is_deleted_after_K_consecutive_misses():
    """Piste confirmée puis plus aucune détection : elle survit à 4 ratés et disparaît au 5ᵉ (K = 5)."""
    _, dets = target_detections(8)
    manager = new_manager(max_misses=5)
    for z in dets[:4]:
        manager.step(z)
    assert len(manager.confirmed_tracks) == 1
    alive = []
    for _ in range(5):
        manager.step(EMPTY)
        alive.append(len(manager.confirmed_tracks))
    assert alive == [1, 1, 1, 1, 0]


def test_a_detection_resets_the_miss_counter():
    """Des ratés non consécutifs ne suppriment jamais la piste (le compteur est remis à zéro)."""
    _, dets = target_detections(30)
    manager = new_manager(max_misses=3)
    for k, z in enumerate(dets):
        manager.step(z if k % 3 != 2 or k < 4 else EMPTY)  # un raté sur trois après confirmation
    assert len(manager.confirmed_tracks) == 1


def count_false_tracks(clutter_rate, n_steps, seed=2):
    """Nombre de pistes confirmées sur des fausses alarmes seules (Pd = 0 : aucun navire détecté)."""
    params = ScenarioParams(n_ships=5, n_steps=n_steps, pd=0.0, clutter_rate=clutter_rate, seed=seed)
    run = run_tracking(new_manager(), generate_scenario(params).detections)
    return len({int(i) for arr in run.ids for i in arr})


def test_clutter_alone_creates_no_confirmed_track_at_nominal_density():
    """200 balayages de fausses alarmes seules (λ = 10 sur 400 km²) : aucune fausse piste confirmée."""
    assert count_false_tracks(clutter_rate=10.0, n_steps=200) == 0


def test_dense_clutter_does_create_false_tracks():
    """Témoin : à λ = 50 (5 fois plus dense) des fausses pistes sont confirmées, donc le comptage fonctionne."""
    assert count_false_tracks(clutter_rate=50.0, n_steps=40) > 0


def test_run_tracking_records_timing_and_counts():
    """run_tracking renvoie un enregistrement par balayage, avec temps et nombre de pistes traitées."""
    _, dets = target_detections(10)
    run = run_tracking(new_manager(), dets)
    assert len(run.ids) == len(run.states) == len(run.step_times) == len(run.n_active) == 10
    assert all(t > 0 for t in run.step_times)
    assert run.n_active[0] == 0 and run.n_active[-1] == 1
