"""Scénarios maritimes multi-cibles : navires CV / virages, détections polaires, fausses alarmes.

Le pistage ne travaille pas sur le signal mais sur des détections (r, θ) : on simule donc directement
les détections, sans passer par la chaîne du jalon B. Un scénario est entièrement déterminé par sa
graine et peut être sauvegardé puis rejoué à l'identique par tous les filtres.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from sim.trajectories import TrajectoryParams, maneuver_trajectory, polar_measurements


@dataclass(frozen=True)
class ScenarioParams:
    """Paramètres d'un scénario de surveillance maritime (capteur à l'origine)."""

    n_ships: int = 10  # nombre de navires (5 à 20)
    n_steps: int = 150  # nombre de balayages
    dt: float = 2.0  # période d'antenne [s]
    half_size: float = 10_000.0  # demi-côté de la zone carrée de surveillance [m] (20 km × 20 km)
    min_range: float = 1_000.0  # distance minimale au radar (l'angle y est mal défini) [m]
    speed_range: tuple = (5.0, 15.0)  # vitesse initiale [m/s]
    turn_fraction: float = 0.5  # part des navires qui effectuent un virage
    turn_rate_range: tuple = (0.015, 0.03)  # |taux de virage| [rad/s] (signe aléatoire)
    turn_steps_range: tuple = (20, 50)  # durée du virage [balayages]
    sigma_a: float = 0.05  # accélération aléatoire de la vérité [m/s²]
    pd: float = 0.9  # probabilité de détection d'un navire par balayage
    clutter_rate: float = 10.0  # nombre moyen de fausses alarmes par balayage (loi de Poisson)
    sigma_r: float = 25.0  # bruit de distance [m]
    sigma_theta: float = 0.01  # bruit d'angle [rad]
    seed: int = 0  # graine du scénario


@dataclass
class Scenario:
    """Vérité terrain et détections d'un scénario, rejouables par tous les filtres."""

    params: ScenarioParams
    truth: np.ndarray  # (n_ships, n_steps, 4) états [x, y, vx, vy]
    in_turn: np.ndarray  # (n_ships, n_steps) booléen : le navire est en virage à cet instant
    detections: list  # n_steps tableaux (m_k, 2) de détections polaires [r, θ]
    sources: list  # n_steps tableaux (m_k,) : indice du navire détecté, −1 pour une fausse alarme


def _random_ship(params: ScenarioParams, rng: np.random.Generator):
    """Tire un navire (trajectoire, masque de virage) qui reste dans la zone et loin du radar.

    Tirage par rejet : position uniforme, cap et vitesse aléatoires, virage éventuel de taux ±ω ;
    on recommence tant que la trajectoire sort de la zone ou passe trop près de l'origine.
    """
    traj_params = TrajectoryParams(dt=params.dt, sigma_a=params.sigma_a)
    n = params.n_steps - 1  # nombre de transitions
    while True:
        pos = rng.uniform(-params.half_size, params.half_size, 2)
        speed, heading = rng.uniform(*params.speed_range), rng.uniform(-np.pi, np.pi)
        x0 = np.array([pos[0], pos[1], speed * np.cos(heading), speed * np.sin(heading)])
        turns = rng.random() < params.turn_fraction
        if turns:
            duration = int(rng.integers(params.turn_steps_range[0], params.turn_steps_range[1] + 1))
            duration = min(duration, max(1, n - 15))  # le virage doit tenir dans le scénario
            start = int(rng.integers(5, n - duration - 4))  # ≥ 5 pas de CV avant et après
            omega = rng.choice([-1, 1]) * rng.uniform(*params.turn_rate_range)
            segments = [(start, 0.0), (duration, omega), (n - start - duration, 0.0)]
        else:
            start = duration = 0
            segments = [(n, 0.0)]
        states = maneuver_trajectory(x0, segments, traj_params, rng)
        assert len(states) == params.n_steps  # garde : tous les navires ont la même durée
        in_zone = np.all(np.abs(states[:, :2]) <= params.half_size)
        far_enough = np.all(np.hypot(states[:, 0], states[:, 1]) >= params.min_range)
        if in_zone and far_enough:
            mask = np.zeros(params.n_steps, dtype=bool)
            mask[start + 1 : start + duration + 1] = True  # transitions effectuées avec ω ≠ 0
            return states, mask


def generate_scenario(params: ScenarioParams) -> Scenario:
    """Génère un scénario complet à partir de sa graine (déterministe).

    À chaque balayage : chaque navire est détecté avec la probabilité Pd (mesure polaire bruitée),
    et le nombre de fausses alarmes suit une loi de Poisson(λ), positions uniformes dans la zone.
    Les détections d'un balayage sont mélangées pour que leur ordre ne révèle rien.
    """
    rng = np.random.default_rng(params.seed)
    ships = [_random_ship(params, rng) for _ in range(params.n_ships)]
    truth = np.array([s for s, _ in ships])
    in_turn = np.array([m for _, m in ships])

    detections, sources = [], []
    for k in range(params.n_steps):
        detected = np.flatnonzero(rng.random(params.n_ships) < params.pd)
        real = polar_measurements(truth[detected, k], params.sigma_r, params.sigma_theta, rng)
        n_false = rng.poisson(params.clutter_rate)
        xy = rng.uniform(-params.half_size, params.half_size, (n_false, 2))
        false = np.column_stack([np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])])
        z = np.vstack([real, false])
        src = np.concatenate([detected, -np.ones(n_false, dtype=int)])
        order = rng.permutation(len(z))
        detections.append(z[order])
        sources.append(src[order])
    return Scenario(params, truth, in_turn, detections, sources)


def save_scenario(scenario: Scenario, path) -> None:
    """Sérialise un scénario (paramètres en JSON, tableaux en .npz) pour le rejouer à l'identique."""
    counts = np.array([len(z) for z in scenario.detections])
    np.savez_compressed(
        Path(path),
        params=json.dumps(asdict(scenario.params)),
        truth=scenario.truth,
        in_turn=scenario.in_turn,
        counts=counts,
        detections=np.vstack(scenario.detections),
        sources=np.concatenate(scenario.sources),
    )


def load_scenario(path) -> Scenario:
    """Relit un scénario écrit par save_scenario (les n-uplets de paramètres sont restaurés)."""
    with np.load(Path(path), allow_pickle=False) as data:
        raw = json.loads(str(data["params"]))
        params = ScenarioParams(
            **{f.name: tuple(raw[f.name]) if isinstance(raw[f.name], list) else raw[f.name]
               for f in fields(ScenarioParams)}
        )
        cuts = np.cumsum(data["counts"])[:-1]
        return Scenario(
            params=params,
            truth=data["truth"],
            in_turn=data["in_turn"],
            detections=np.split(data["detections"], cuts),
            sources=np.split(data["sources"], cuts),
        )
