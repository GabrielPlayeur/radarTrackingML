"""KalmanNet : gain de Kalman appris par un petit GRU (Revach et al., IEEE TSP 2022), prédiction et mise à jour de Kalman conservées.

Le filtre garde la structure de Kalman : x⁻ = F·x (modèle CV seul), innovation y = z − h(x⁻), x = x⁻ + K·y.
Seul le calcul du gain K est remplacé par un réseau, qui n'a besoin ni de Q, ni de R, ni de la jacobienne :
il apprend à pondérer l'innovation à partir de ce qu'il observe (innovations, corrections passées).

Écarts assumés par rapport à l'article, pour économiser des données avec un modèle minuscule :
- le gain est exprimé dans le **repère de la ligne de visée** (axes « le long du rayon » et « en travers »),
  puis ramené en cartésien par la rotation du gisement prédit : le réseau est ainsi équivariant par rotation
  autour du radar (testé exactement) et n'a pas à réapprendre le gain pour chaque cap ;
- une seule couche GRU précédée d'une projection linéaire ;
- un a priori de gain de type alpha-bêta (a = 0,2, b = 0,03, choisis sur le jeu de validation) et un gain **borné**
  par construction, K̃ = bas + (haut − bas)·sigmoïde(sortie + logit(a priori)) : le réseau part d'un filtre stable
  et ses gains restent dans un domaine borné. Sans borne, une perturbation de 0,03 sur les poids de la tête suffit à
  faire diverger l'état (10¹⁴ m), donc à produire des NaN à l'entraînement. Les bornes diagonales (α ≤ 0,9, β ≤ 0,2)
  sont dans la zone de stabilité d'un filtre alpha-bêta à gain constant (0 < α < 2, 0 < β < 4 − 2α). Ce n'est pas
  une preuve de stabilité : le gain varie d'un pas à l'autre et a des termes croisés, et un système dont chaque
  gain figé serait stable peut quand même diverger. En pratique, les bornes suffisent à éviter les divergences.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sim.trajectories import transition_cv

N_FEATURES = 7  # innovation (2) + correction précédente (4) + distance (1)


@dataclass(frozen=True)
class KalmanNetParams:
    """Hyperparamètres du réseau et échelles de normalisation des entrées."""

    dt: float = 2.0  # période d'antenne [s]
    sigma_r: float = 25.0  # écart-type de mesure en distance [m] : normalise l'innovation
    sigma_theta: float = 0.01  # écart-type de mesure en angle [rad]
    hidden_in: int = 32  # largeur de la projection d'entrée
    hidden_gru: int = 64  # taille de l'état caché du GRU
    pos_scale: float = 50.0  # échelle des corrections de position [m]
    vel_scale: float = 1.0  # échelle des corrections de vitesse [m/s]
    range_scale: float = 10_000.0  # échelle de la distance [m]
    gain_prior_pos: float = 0.2  # a priori alpha : gain de position (diagonale)
    gain_prior_vel: float = 0.03  # a priori bêta : gain de vitesse (diagonale, avant division par dt)
    gain_max_pos: float = 0.9  # borne haute du gain de position (diagonale) ; la borne basse est 0
    gain_max_vel: float = 0.2  # borne haute du gain de vitesse (diagonale)
    gain_cross_pos: float = 0.2  # amplitude (±) des termes croisés de position
    gain_cross_vel: float = 0.03  # amplitude (±) des termes croisés de vitesse


def gain_bounds(params: KalmanNetParams) -> tuple:
    """Bornes (bas, haut) et a priori de chacun des 8 coefficients de K̃ (ordre : lignes pos. along/cross, vit. along/cross)."""
    p = params
    lo = [0, -p.gain_cross_pos, -p.gain_cross_pos, 0, 0, -p.gain_cross_vel, -p.gain_cross_vel, 0]
    hi = [p.gain_max_pos, p.gain_cross_pos, p.gain_cross_pos, p.gain_max_pos,
          p.gain_max_vel, p.gain_cross_vel, p.gain_cross_vel, p.gain_max_vel]
    prior = [p.gain_prior_pos, 0, 0, p.gain_prior_pos, p.gain_prior_vel, 0, 0, p.gain_prior_vel]
    return tuple(torch.tensor(v, dtype=torch.float64) for v in (lo, hi, prior))  # float64 : pas d'erreur d'arrondi sur l'a priori


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Ramène un angle dans [−π, π] de façon différentiable : atan2(sin a, cos a)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotate(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotation d'angle φ de vecteurs 2D (B, 2) : [c·vx − s·vy, s·vx + c·vy]."""
    return torch.stack([cos * v[:, 0] - sin * v[:, 1], sin * v[:, 0] + cos * v[:, 1]], dim=1)


class KalmanNet(nn.Module):
    """Filtre à gain appris : mêmes équations de prédiction et de mise à jour que le KF, gain K = réseau."""

    def __init__(self, params: KalmanNetParams = KalmanNetParams()):
        super().__init__()
        self.params = params
        self.register_buffer("F", torch.tensor(transition_cv(params.dt), dtype=torch.float32))
        self.embed = nn.Linear(N_FEATURES, params.hidden_in)
        self.gru = nn.GRUCell(params.hidden_in, params.hidden_gru)
        self.head = nn.Linear(params.hidden_gru, 8)  # K̃ : matrice 4×2 (lignes : pos. along/cross, vit. along/cross)
        lo, hi, prior = gain_bounds(params)
        self.register_buffer("gain_lo", lo)
        self.register_buffer("gain_hi", hi)
        # logit de la position de l'a priori dans [bas, haut] : sortie nulle du réseau = gain a priori
        self.register_buffer("gain_logit_prior", torch.logit((prior - lo) / (hi - lo)))
        with torch.no_grad():  # départ = filtre alpha-bêta stable ; le réseau n'apprend que des écarts
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()

    def gain_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """Gain borné K̃ (B, 4, 2) : bas + (haut − bas)·sigmoïde(sortie + logit(a priori))."""
        lo, hi, logit = (t.to(output.dtype) for t in (self.gain_lo, self.gain_hi, self.gain_logit_prior))
        return (lo + (hi - lo) * torch.sigmoid(output + logit)).view(-1, 4, 2)

    @property
    def n_parameters(self) -> int:
        """Nombre de paramètres appris."""
        return sum(p.numel() for p in self.parameters())

    def initial_state(self, z0: torch.Tensor) -> tuple:
        """État initial depuis la première mesure (B, 2) : position [r·cos θ, r·sin θ], vitesse nulle.

        Retourne (x, correction précédente, état caché du GRU) ; c'est la même initialisation que l'EKF
        et l'IMM du jalon C (sans leur covariance : KalmanNet n'en a pas).
        """
        x = torch.stack(
            [z0[:, 0] * torch.cos(z0[:, 1]), z0[:, 0] * torch.sin(z0[:, 1]),
             torch.zeros_like(z0[:, 0]), torch.zeros_like(z0[:, 0])], dim=1)
        return x, torch.zeros_like(x), z0.new_zeros(z0.shape[0], self.params.hidden_gru)

    def step(self, x, correction, hidden, z) -> tuple:
        """Un pas de filtre : prédiction de Kalman, features, gain par le GRU, mise à jour de Kalman.

        x⁻ = F·x ; y = z − h(x⁻) (angle normalisé) ; K̃ = réseau(features) ; x = x⁻ + T(θ̂)·K̃·[δr, r̂·δθ],
        avec T la rotation repère ligne de visée → repère cartésien et une division par dt sur la vitesse.
        Retourne (x, correction appliquée, état caché).
        """
        p = self.params
        x_pred = x @ self.F.T
        r_hat = torch.hypot(x_pred[:, 0], x_pred[:, 1]).clamp_min(1.0)
        theta_hat = torch.atan2(x_pred[:, 1], x_pred[:, 0])
        d_r = z[:, 0] - r_hat
        d_theta = wrap_angle(z[:, 1] - theta_hat)
        c, s = torch.cos(theta_hat), torch.sin(theta_hat)

        # correction précédente exprimée dans le repère de ligne de visée courant (rotation de −θ̂)
        corr_los = torch.cat(
            [rotate(correction[:, :2], c, -s) / p.pos_scale, rotate(correction[:, 2:], c, -s) / p.vel_scale], dim=1
        )
        features = torch.cat(
            [(d_r / p.sigma_r)[:, None], (d_theta / p.sigma_theta)[:, None], corr_los,
             (r_hat / p.range_scale)[:, None]], dim=1)

        hidden = self.gru(torch.relu(self.embed(features)), hidden)
        gain = self.gain_from_output(self.head(hidden))  # K̃ borné, dans le repère ligne de visée
        innovation_los = torch.stack([d_r, r_hat * d_theta], dim=1)  # [le long, en travers] en mètres
        update_los = (gain @ innovation_los[:, :, None])[:, :, 0]
        update = torch.cat([rotate(update_los[:, :2], c, s), rotate(update_los[:, 2:] / p.dt, c, s)], dim=1)
        return x_pred + update, update, hidden

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Filtre une séquence de mesures (B, T, 2) → états estimés (B, T, 4) ; l'instant 0 est l'initialisation."""
        x, correction, hidden = self.initial_state(z[:, 0])
        estimates = [x]
        for k in range(1, z.shape[1]):
            x, correction, hidden = self.step(x, correction, hidden, z[:, k])
            estimates.append(x)
        return torch.stack(estimates, dim=1)


class KalmanNetFilter:
    """Enveloppe monocible (batch de 1) avec l'interface predict / update des autres filtres, pour le chronométrage.

    KalmanNet ne fournit pas de covariance : ni gating de Mahalanobis, ni NEES, ni NIS ne sont possibles avec lui.
    """

    def __init__(self, model: KalmanNet, z0: np.ndarray):
        self.model = model.eval()
        with torch.no_grad():
            z0_t = torch.as_tensor(np.asarray(z0, dtype=np.float32))[None]
            self._x, self._corr, self._hidden = model.initial_state(z0_t)
        self._z = None

    @property
    def x(self) -> np.ndarray:
        """État estimé courant (4,)."""
        return self._x[0].numpy()

    def predict(self) -> None:
        """Rien à faire séparément : la prédiction est intégrée au pas (voir update)."""

    def update(self, z: np.ndarray) -> None:
        """Pas complet (prédiction + gain appris + mise à jour) avec la mesure z."""
        with torch.no_grad():
            z_t = torch.as_tensor(np.asarray(z, dtype=np.float32))[None]
            self._x, self._corr, self._hidden = self.model.step(self._x, self._corr, self._hidden, z_t)
