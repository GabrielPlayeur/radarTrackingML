"""Jeux de données d'entraînement pour le gain appris : une cible, trajectoires CV + virages, mesures polaires.

Chaque trajectoire est un navire de `sim.scenario` (position et cap aléatoires, virage éventuel de taux signé)
observé sans raté ni fausse alarme : le gain appris est étudié « une cible à la fois », comme demandé au jalon D.
Les trois découpages (train / val / test) viennent de graines indépendantes dérivées d'une seule graine.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from sim.scenario import ScenarioParams, random_ship
from sim.trajectories import polar_measurements


@dataclass(frozen=True)
class DatasetParams:
    """Paramètres de génération des trajectoires (distribution d'entraînement par défaut)."""

    n_train: int = 4000
    n_val: int = 300
    n_test: int = 300
    n_steps: int = 150  # balayages par trajectoire
    dt: float = 2.0  # période d'antenne [s]
    half_size: float = 10_000.0  # demi-côté de la zone [m]
    min_range: float = 1_000.0  # distance minimale au radar [m]
    speed_range: tuple = (5.0, 15.0)  # vitesse initiale [m/s]
    turn_probability: float = 0.5  # part des trajectoires qui comportent un virage
    turn_rate_range: tuple = (0.015, 0.03)  # |taux de virage| [rad/s], signe aléatoire
    turn_steps_range: tuple = (20, 50)  # durée du virage [balayages]
    sigma_a: float = 0.05  # accélération aléatoire de la vérité [m/s²]
    sigma_r: float = 25.0  # bruit de mesure en distance [m]
    sigma_theta: float = 0.01  # bruit de mesure en angle [rad]
    seed: int = 0  # graine des découpages


@dataclass
class TrajectoryBatch:
    """Un lot de trajectoires : vérité, mesures polaires et masque de virage."""

    truth: np.ndarray  # (N, T, 4) états [x, y, vx, vy]
    z: np.ndarray  # (N, T, 2) mesures [r, θ]
    in_turn: np.ndarray  # (N, T) booléen : le navire est en virage à cet instant


def _scenario_params(params: DatasetParams) -> ScenarioParams:
    """Paramètres de navire équivalents pour réutiliser le générateur du jalon C."""
    return ScenarioParams(
        n_ships=1, n_steps=params.n_steps, dt=params.dt, half_size=params.half_size,
        min_range=params.min_range, speed_range=params.speed_range,
        turn_fraction=params.turn_probability, turn_rate_range=params.turn_rate_range,
        turn_steps_range=params.turn_steps_range, sigma_a=params.sigma_a,
    )


def generate_batch(params: DatasetParams, n: int, rng: np.random.Generator) -> TrajectoryBatch:
    """Génère n trajectoires indépendantes et leurs mesures polaires bruitées (σ_r, σ_θ)."""
    ship_params = _scenario_params(params)
    ships = [random_ship(ship_params, rng) for _ in range(n)]
    truth = np.array([states for states, _, _ in ships])
    in_turn = np.array([mask for _, mask, _ in ships])
    z = np.array([polar_measurements(states, params.sigma_r, params.sigma_theta, rng) for states in truth])
    return TrajectoryBatch(truth, z, in_turn)


def make_splits(params: DatasetParams) -> dict:
    """Découpage train / val / test : trois flux aléatoires indépendants, reproductibles par `params.seed`."""
    sizes = {"train": params.n_train, "val": params.n_val, "test": params.n_test}
    seeds = np.random.SeedSequence(params.seed).spawn(len(sizes))
    return {
        name: generate_batch(params, n, np.random.default_rng(child))
        for (name, n), child in zip(sizes.items(), seeds)
    }


# Conditions de test hors distribution (section « quand ça ne marche pas » de evaluate.py).
STRESS_KINDS = ("strong_turns", "noisy", "long")


def stress_params(params: DatasetParams, kind: str) -> DatasetParams:
    """Variante hors distribution du jeu de test.

    - strong_turns : tous les navires virent, 3 à 4 fois plus vite qu'à l'entraînement (0,06–0,10 rad/s) ;
    - noisy : bruit de mesure doublé (les filtres gardent leur R nominal) ;
    - long : séquences 4 fois plus longues que celles de l'entraînement.

    Biais de « long » : la zone est la même, et le tirage par rejet de `random_ship` (le navire doit y rester toute
    la séquence) écarte surtout les navires rapides, qui parcourent 4 fois plus de distance. Mesuré sur 300
    trajectoires : vitesse initiale moyenne 8,9 m/s au lieu de 10,0, part au-dessus de 12 m/s divisée par 2, distance
    moyenne au radar 6,4 km au lieu de 7,2. Agrandir la zone ne règle rien (vitesse 9,8 m/s même à ×4) et éloigne les
    cibles (bruit transverse r·σ_θ plus fort) : dans une zone bornée, on ne peut pas allonger les séquences sans
    changer la distribution. Les méthodes restent comparables entre elles sur ce jeu, mais une RMSE plus faible
    qu'en nominal ne prouve pas que la durée aide : les cibles y sont plus lentes et plus proches.
    """
    if kind == "strong_turns":
        return replace(params, turn_probability=1.0, turn_rate_range=(0.06, 0.10), turn_steps_range=(10, 20))
    if kind == "noisy":
        return replace(params, sigma_r=2 * params.sigma_r, sigma_theta=2 * params.sigma_theta)
    if kind == "long":
        return replace(params, n_steps=4 * params.n_steps)
    raise ValueError(f"condition inconnue : {kind!r} (attendu : {STRESS_KINDS})")


def make_stress_test_set(params: DatasetParams, kind: str) -> TrajectoryBatch:
    """Jeu de test hors distribution (flux aléatoire propre à `kind`, indépendant de train / val / test)."""
    index = 10 + STRESS_KINDS.index(kind)
    rng = np.random.default_rng(np.random.SeedSequence([params.seed, index]))
    return generate_batch(stress_params(params, kind), params.n_test, rng)


def to_tensors(batch: TrajectoryBatch, device="cpu", dtype=torch.float32) -> tuple:
    """(vérité, mesures) en tenseurs torch : (N, T, 4) et (N, T, 2)."""
    return (
        torch.as_tensor(batch.truth, dtype=dtype, device=device),
        torch.as_tensor(batch.z, dtype=dtype, device=device),
    )
