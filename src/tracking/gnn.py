"""Association de données GNN (Global Nearest Neighbour) : gating de Mahalanobis + algorithme hongrois.

Un estimateur (KF, EKF ou IMM) doit fournir `innovation_and_covariance(z)` → (y, S).
Gating : on ne considère que les détections plausibles (distance de Mahalanobis sous un seuil χ²).
Association : parmi les paires plausibles, on choisit l'affectation 1-1 de coût total minimal.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

UNGATED_COST = 1e6  # coût des paires hors porte : bien supérieur à n'importe quelle somme de d² valides


@dataclass(frozen=True)
class AssociationResult:
    """Résultat d'une association : paires (piste, détection) et éléments restés seuls."""

    matches: list  # liste de (indice de piste, indice de détection)
    unmatched_tracks: list
    unmatched_detections: list


def gate_threshold(gate_probability: float = 0.999, dof: int = 2) -> float:
    """Seuil de porte γ tel que P(d² < γ) = gate_probability, avec d² ~ χ²(dof) pour une vraie détection.

    Avec dof = 2 (distance, angle) et 99,9 %, γ ≈ 13,8 : l'ellipse de validation.
    """
    return float(chi2.ppf(gate_probability, dof))


def mahalanobis_squared(innovation: np.ndarray, S: np.ndarray) -> float:
    """Distance de Mahalanobis au carré d² = yᵀ·S⁻¹·y (distance en « nombre d'écarts-types »)."""
    return float(innovation @ np.linalg.solve(S, innovation))


def distance_matrix(estimators: list, detections: np.ndarray) -> np.ndarray:
    """Matrice d² (n_pistes × n_détections) entre chaque piste prédite et chaque détection.

    L'innovation d'angle est normalisée par l'estimateur : une détection à θ = −π+ε reste proche
    d'une piste prédite à θ = π−ε.
    """
    d2 = np.empty((len(estimators), len(detections)))
    for i, est in enumerate(estimators):
        for j, z in enumerate(detections):
            d2[i, j] = mahalanobis_squared(*est.innovation_and_covariance(z))
    return d2


def associate(
    estimators: list, detections: np.ndarray, gate_probability: float = 0.999
) -> AssociationResult:
    """GNN : affectation 1-1 qui minimise Σ d² sur les paires dans la porte (algorithme hongrois).

    Les paires hors porte (d² ≥ γ) reçoivent un coût énorme ; l'optimum maximise donc d'abord le
    nombre d'associations valides, puis minimise leur coût. Toute paire hors porte restante est écartée.
    """
    n_tracks, n_det = len(estimators), len(detections)
    if n_tracks == 0 or n_det == 0:
        return AssociationResult([], list(range(n_tracks)), list(range(n_det)))
    d2 = distance_matrix(estimators, detections)
    gamma = gate_threshold(gate_probability)
    cost = np.where(d2 < gamma, d2, UNGATED_COST)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if d2[i, j] < gamma]
    matched_tracks = {i for i, _ in matches}
    matched_dets = {j for _, j in matches}
    return AssociationResult(
        matches,
        [i for i in range(n_tracks) if i not in matched_tracks],
        [j for j in range(n_det) if j not in matched_dets],
    )
