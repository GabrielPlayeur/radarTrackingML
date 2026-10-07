"""Traitement du cube radar : compression d'impulsion, FFT Doppler, CA-CFAR 2D.

Chaîne : rx (N × n_fast) → compression (temps rapide) → FFT Doppler (temps lent) → carte
distance-Doppler complexe → puissance → CFAR → détections (booléens).
"""

import numpy as np
from scipy.ndimage import uniform_filter

from radar.waveform import RadarParams, reference_chirp


def compress_pulses(rx: np.ndarray, params: RadarParams) -> np.ndarray:
    """Compression d'impulsion : corrélation de chaque impulsion avec la réplique du chirp.

    Filtre adapté y[n] = Σ_k rx[k+n]·conj(ref[k]), calculé par FFT : Y = FFT(rx)·conj(FFT(ref)).
    Le pic est de largeur ≈ 1/B et le SNR est multiplié par T·fs.
    """
    ref_spectrum = np.fft.fft(reference_chirp(params))
    return np.fft.ifft(np.fft.fft(rx, axis=1) * np.conj(ref_spectrum), axis=1)


def doppler_window(n_pulses: int, window: str = "hann") -> np.ndarray:
    """Fenêtre de pondération du temps lent : « hann » (lobes secondaires bas) ou « rect »."""
    if window == "hann":
        return np.hanning(n_pulses)
    if window == "rect":
        return np.ones(n_pulses)
    raise ValueError(f"fenêtre inconnue : {window!r}")


def doppler_fft(compressed: np.ndarray, window: str = "hann") -> np.ndarray:
    """FFT sur le temps lent (axe 0), fenêtrée, avec la fréquence nulle au centre.

    X[k] = Σ_m w[m]·y[m]·exp(−j2π·k·m/N). Un écho de fréquence Doppler f_d (en Hz) tombe dans la
    case N/2 + f_d·N/PRF après fftshift. La phase porte la vitesse : ne jamais garder que le module.
    """
    w = doppler_window(compressed.shape[0], window)
    return np.fft.fftshift(np.fft.fft(compressed * w[:, None], axis=0), axes=0)


def range_doppler_map(rx: np.ndarray, params: RadarParams, window: str = "hann") -> np.ndarray:
    """Carte distance-Doppler complexe (N × n_fast) : compression puis FFT Doppler."""
    return doppler_fft(compress_pulses(rx, params), window)


def range_axis(params: RadarParams) -> np.ndarray:
    """Distance de chaque échantillon du temps rapide : R[n] = r_min + n·c/(2·fs)."""
    return params.r_min + np.arange(params.n_fast) * params.range_bin


def velocity_axis(params: RadarParams) -> np.ndarray:
    """Vitesse radiale de chaque case Doppler (v > 0 = s'éloigne) : v = −λ·f/2.

    Le signe vient de la phase exp(−j2π·fc·τ) : une cible qui s'éloigne a une fréquence négative.
    """
    freq = np.fft.fftshift(np.fft.fftfreq(params.n_pulses, 1 / params.prf))
    return -params.wavelength * freq / 2


def ca_cfar_2d(
    power: np.ndarray,
    pfa: float,
    guard: tuple[int, int] = (2, 2),
    reference: tuple[int, int] = (2, 6),
) -> np.ndarray:
    """CA-CFAR 2D (Cell-Averaging) sur une carte de puissance (axes : Doppler, distance).

    Pour chaque case, le bruit est la moyenne de l'anneau de cellules de référence (fenêtre
    externe moins fenêtre de garde). Seuil = α·bruit avec α = n·(Pfa^(−1/n) − 1), n = nombre de
    cellules de référence, valable pour un bruit en loi exponentielle (détecteur quadratique).
    `guard` et `reference` sont des demi-largeurs (Doppler, distance). L'axe Doppler est circulaire
    (FFT), l'axe distance ne l'est pas : aux bords, n est plus petit et le seuil s'adapte.
    """
    guard_size = tuple(2 * g + 1 for g in guard)
    outer_size = tuple(2 * (g + r) + 1 for g, r in zip(guard, reference))
    modes = ("wrap", "constant")  # Doppler circulaire, distance bornée (zéros hors carte)

    def ring_sum(x: np.ndarray) -> np.ndarray:
        """Somme de x sur l'anneau de référence (boîte externe − boîte de garde)."""
        outer = uniform_filter(x, size=outer_size, mode=modes, cval=0.0) * np.prod(outer_size)
        inner = uniform_filter(x, size=guard_size, mode=modes, cval=0.0) * np.prod(guard_size)
        return outer - inner

    n_ref = np.rint(ring_sum(np.ones_like(power)))
    noise = ring_sum(power) / n_ref
    alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1.0)
    return power > alpha * noise
