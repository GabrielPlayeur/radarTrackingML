"""Cohérence statistique d'un filtre : NEES (erreur d'état) et NIS (innovation).

Un filtre est « honnête » si sa covariance P décrit bien son erreur réelle. Alors, avec n la
dimension de l'état, NEES = (x−x̂)ᵀ·P⁻¹·(x−x̂) suit une loi χ²(n) et, avec m la dimension de la
mesure, NIS = yᵀ·S⁻¹·y suit une loi χ²(m). Moyenne trop haute = trop optimiste, trop basse = trop prudent.
"""

import numpy as np
from scipy.stats import chi2


def _quadratic_form(vectors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Forme quadratique vᵀ·C⁻¹·v sur les derniers axes (résolution de système, sans inverse)."""
    solved = np.linalg.solve(covariances, vectors[..., None])[..., 0]
    return np.sum(vectors * solved, axis=-1)


def nees(x_true: np.ndarray, x_est: np.ndarray, P: np.ndarray) -> np.ndarray:
    """NEES à chaque instant : ε_k = (x_k − x̂_k)ᵀ · P_k⁻¹ · (x_k − x̂_k).

    Entrées de forme (..., n, dim_x) et (..., n, dim_x, dim_x) ; sortie (..., n).
    """
    return _quadratic_form(x_true - x_est, P)


def nis(innovations: np.ndarray, S: np.ndarray) -> np.ndarray:
    """NIS à chaque instant : ν_k = y_kᵀ · S_k⁻¹ · y_k, avec y l'innovation et S sa covariance.

    Ne demande pas la vérité terrain : c'est le test de cohérence utilisable en conditions réelles.
    Les instants sans mesure (NaN) restent NaN.
    """
    valid = ~np.isnan(innovations).any(axis=-1)
    out = np.full(innovations.shape[:-1], np.nan)
    out[valid] = _quadratic_form(innovations[valid], S[valid])
    return out


def chi2_interval(dof: int, n_runs: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalle de confiance de la moyenne de n_runs valeurs indépendantes de loi χ²(dof).

    La somme de n_runs variables χ²(dof) suit χ²(n_runs·dof) ; on en prend les quantiles
    α/2 et 1−α/2 puis on divise par n_runs : [χ²_{α/2}(R·dof)/R, χ²_{1−α/2}(R·dof)/R].
    """
    alpha = 1 - confidence
    lo, hi = chi2.ppf([alpha / 2, 1 - alpha / 2], n_runs * dof)
    return lo / n_runs, hi / n_runs
