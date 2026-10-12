"""OSPA (Optimal Sub-Pattern Assignment) : distance entre l'ensemble des cibles vraies et celui des pistes.

Une seule mesure qui pénalise à la fois l'erreur de position des pistes bien associées et les erreurs de
cardinalité (cibles non pistées, fausses pistes). Réf. : Schuhmacher, Vo, Vo, IEEE TSP 2008.

Choix des paramètres :
- c (distance de coupure) : une piste à plus de c d'une cible compte comme une cible manquée plus une fausse
  piste. On prend c ≈ taille de la porte d'association (~300 m ici) : au-delà, la piste ne peut plus être la bonne.
- p : p = 1 moyenne les erreurs, p = 2 pénalise davantage les grosses erreurs (comme un RMSE). On prend p = 2.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def ospa_components(X: np.ndarray, Y: np.ndarray, c: float, p: float = 2.0) -> tuple[float, float, float]:
    """OSPA total, composante de localisation et composante de cardinalité, avec total^p = loc^p + card^p.

    Pour |X| = m ≤ |Y| = n : d = [ (1/n)·( min_π Σ_i min(c, ‖x_i − y_π(i)‖)^p + c^p·(n − m) ) ]^(1/p).
    Deux ensembles vides : 0. Un vide et un non vide : c.
    """
    X = np.asarray(X, dtype=float).reshape(-1, 2) if np.size(X) else np.zeros((0, 2))
    Y = np.asarray(Y, dtype=float).reshape(-1, 2) if np.size(Y) else np.zeros((0, 2))
    if len(X) > len(Y):
        X, Y = Y, X
    m, n = len(X), len(Y)
    if n == 0:
        return 0.0, 0.0, 0.0
    if m == 0:
        return c, 0.0, c
    dist = np.minimum(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2), c) ** p
    rows, cols = linear_sum_assignment(dist)
    localization = dist[rows, cols].sum() / n
    cardinality = c**p * (n - m) / n
    return (
        float((localization + cardinality) ** (1 / p)),
        float(localization ** (1 / p)),
        float(cardinality ** (1 / p)),
    )


def ospa(X: np.ndarray, Y: np.ndarray, c: float, p: float = 2.0) -> float:
    """Distance OSPA d_p^(c)(X, Y) entre deux ensembles de positions (lignes = points 2D)."""
    return ospa_components(X, Y, c, p)[0]


def ospa_over_time(truth_xy: list, tracks_xy: list, c: float, p: float = 2.0) -> np.ndarray:
    """OSPA à chaque balayage : truth_xy[k] et tracks_xy[k] sont les positions (n_k, 2) à l'instant k."""
    return np.array([ospa(t, s, c, p) for t, s in zip(truth_xy, tracks_xy)])
