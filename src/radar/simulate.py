"""Simulation d'un CPI radar : cube complexe (N impulsions × n_fast échantillons) en bande de base.

Convention de signe : v > 0 = cible qui s'éloigne (distance croissante). Avec le terme de phase
exp(−j2π·fc·τ), la fréquence Doppler observée en temps lent est alors −2v/λ (négative).
"""

from dataclasses import dataclass

import numpy as np

from radar.waveform import RadarParams, chirp_at


@dataclass(frozen=True)
class Target:
    """Cible ponctuelle non fluctuante : distance initiale, vitesse radiale, amplitude."""

    r0: float  # distance au début du CPI [m]
    v: float  # vitesse radiale [m/s], > 0 = s'éloigne
    amplitude: float = 1.0  # amplitude de l'écho (module du chirp reçu)


def noise_sigma(snr_db: float) -> float:
    """Écart-type du bruit complexe pour un SNR par échantillon donné (cible d'amplitude 1).

    SNR = a² / σ² avec a = 1 (puissance du signal dans l'impulsion) donc σ = 10^(−SNR_dB/20).
    Une cible d'amplitude a a donc un SNR de 20·log10(a) + snr_db.
    """
    return 10 ** (-snr_db / 20)


def simulate_cpi(
    params: RadarParams,
    targets: list[Target],
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cube reçu rx[m, n] : écho de chaque cible, retardé de τ_m = 2(R0 + v·m/PRF)/c, plus bruit.

    Écho = a · s(t − τ_m) · exp(−j2π·fc·τ_m) : le retard évalué analytiquement donne la distance
    (temps rapide), la phase de porteuse qui évolue d'une impulsion à l'autre donne la vitesse
    (temps lent). Bruit : complexe gaussien de variance σ² (σ²/2 par voie I et Q), omis si snr_db=None.
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)
    t_fast = 2 * params.r_min / params.c + np.arange(params.n_fast) / params.fs  # (n_fast,)
    t_slow = np.arange(params.n_pulses) / params.prf  # (N,)

    rx = np.zeros((params.n_pulses, params.n_fast), dtype=complex)
    for target in targets:
        tau = 2 * (target.r0 + target.v * t_slow[:, None]) / params.c  # (N, 1)
        rx += (
            target.amplitude
            * chirp_at(t_fast[None, :] - tau, params)
            * np.exp(-2j * np.pi * params.fc * tau)
        )

    if snr_db is not None:
        sigma = noise_sigma(snr_db)
        noise = rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
        rx += sigma * noise / np.sqrt(2)
    return rx
