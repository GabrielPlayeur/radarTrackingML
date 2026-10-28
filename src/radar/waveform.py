"""Forme d'onde radar : paramètres du radar et chirp LFM (modulation linéaire de fréquence).

Tout est en bande de base (signal I/Q complexe) : on ne génère jamais la porteuse à 9,4 GHz,
seulement sa phase, voir simulate.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadarParams:
    """Paramètres d'un radar impulsionnel à compression d'impulsion (bande X par défaut).

    Les grandeurs dérivées (résolutions, ambiguïtés, gains) sont des propriétés : elles
    reprennent le tableau « formules à connaître » du radar impulsionnel.
    """

    c: float = 3e8  # vitesse de la lumière [m/s]
    fc: float = 9.4e9  # fréquence porteuse [Hz]
    B: float = 10e6  # bande du chirp [Hz]
    T: float = 10e-6  # durée du chirp [s]
    oversampling: float = 2.0  # fs = oversampling * B (temps rapide)
    prf: float = 1e3  # cadence de répétition des impulsions [Hz]
    n_pulses: int = 64  # impulsions intégrées dans un CPI
    r_min: float = 1_000.0  # début de la fenêtre de réception [m]
    n_fast: int = 4_000  # échantillons par impulsion (temps rapide)
    seed: int = 0  # graine du bruit

    @property
    def wavelength(self) -> float:
        """Longueur d'onde λ = c / fc (≈ 3,2 cm en bande X)."""
        return self.c / self.fc

    @property
    def fs(self) -> float:
        """Fréquence d'échantillonnage du temps rapide : fs = oversampling · B."""
        return self.oversampling * self.B

    @property
    def n_pulse_samples(self) -> int:
        """Nombre d'échantillons occupés par le chirp : L = T · fs."""
        return int(round(self.T * self.fs))

    @property
    def chirp_rate(self) -> float:
        """Pente du chirp : k = B / T [Hz/s]."""
        return self.B / self.T

    @property
    def range_resolution(self) -> float:
        """Résolution en distance : ΔR = c / (2B)."""
        return self.c / (2 * self.B)

    @property
    def range_bin(self) -> float:
        """Pas entre deux échantillons distance : c / (2·fs) (plus fin que ΔR si fs > B)."""
        return self.c / (2 * self.fs)

    @property
    def unambiguous_range(self) -> float:
        """Distance non ambiguë : R_max = c / (2·PRF)."""
        return self.c / (2 * self.prf)

    @property
    def unambiguous_velocity(self) -> float:
        """Vitesse radiale non ambiguë : |v| < λ·PRF / 4 (f_d < PRF/2 avec f_d = 2v/λ)."""
        return self.wavelength * self.prf / 4

    @property
    def velocity_resolution(self) -> float:
        """Résolution en vitesse : Δv = λ·PRF / (2N)."""
        return self.wavelength * self.prf / (2 * self.n_pulses)

    @property
    def doppler_bin(self) -> float:
        """Pas de la FFT Doppler : PRF / N [Hz]."""
        return self.prf / self.n_pulses

    @property
    def compression_gain(self) -> float:
        """Gain de SNR de la compression d'impulsion : G = T · fs (= 2·B·T avec fs = 2B)."""
        return self.T * self.fs


def chirp_at(t: np.ndarray, params: RadarParams) -> np.ndarray:
    """Chirp LFM évalué analytiquement aux instants t (nul hors de [0, T[).

    Le chirp est s(t) = exp(jπ·(B/T)·t²) pour 0 ≤ t < T : sa fréquence instantanée croît
    linéairement de 0 à B. L'évaluation analytique permet des retards fractionnaires exacts.
    """
    inside = (t >= 0) & (t < params.T)
    return np.where(inside, np.exp(1j * np.pi * params.chirp_rate * t**2), 0)


def reference_chirp(params: RadarParams) -> np.ndarray:
    """Réplique du chirp échantillonnée à fs sur n_fast points (complétée par des zéros).

    C'est le signal de référence du filtre adapté : ref[n] = s(n / fs), n = 0 … n_fast-1.
    """
    if params.n_pulse_samples > params.n_fast:
        raise ValueError("n_fast doit être ≥ T·fs pour contenir le chirp")
    return chirp_at(np.arange(params.n_fast) / params.fs, params)


def autocorrelation_width_3db(params: RadarParams, oversampling: float = 32.0) -> float:
    """Largeur à −3 dB (en puissance) du lobe principal de l'autocorrélation du chirp, en secondes.

    Autocorrélation r(τ) = ∫ s(t)·s*(t−τ) dt, calculée par FFT sur une grille fine. On cherche
    le retard où |r| tombe à |r(0)|/√2, la largeur vaut deux fois ce retard. Théorie : 0,886 / B.
    """
    fs = oversampling * params.B
    t = np.arange(int(round(params.T * fs))) / fs
    s = chirp_at(t, params)
    n = 2 * len(s)
    spectrum = np.fft.fft(s, n)
    r = np.abs(np.fft.ifft(np.abs(spectrum) ** 2))[: len(s)]
    r = r / r[0]
    k = int(np.argmax(r < 1 / np.sqrt(2)))  # premier échantillon sous −3 dB
    # interpolation linéaire entre k-1 et k pour affiner le point de croisement
    frac = (r[k - 1] - 1 / np.sqrt(2)) / (r[k - 1] - r[k])
    return 2 * (k - 1 + frac) / fs
