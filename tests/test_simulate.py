"""Validations du simulateur de CPI (jalon B)."""

import numpy as np
import pytest

from radar.simulate import Target, noise_sigma, simulate_cpi
from radar.waveform import RadarParams

PARAMS = RadarParams()


def test_cube_shape_and_dtype():
    """Le cube a la forme (N impulsions, n_fast) et il est complexe."""
    rx = simulate_cpi(PARAMS, [Target(3_000.0, 5.0)])
    assert rx.shape == (PARAMS.n_pulses, PARAMS.n_fast)
    assert np.iscomplexobj(rx)


def test_echo_has_amplitude_a_and_lasts_L_samples():
    """Sans bruit, l'écho a un module a pendant exactement L = T·fs échantillons (±1)."""
    rx = simulate_cpi(PARAMS, [Target(3_000.0, 0.0, amplitude=0.7)])
    modulus = np.abs(rx[0])
    assert modulus.max() == pytest.approx(0.7)
    assert abs(np.count_nonzero(modulus) - PARAMS.n_pulse_samples) <= 1


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 13.0])
def test_noise_power_matches_requested_snr(snr_db):
    """Sur du bruit seul, la puissance moyenne |n|² vaut σ² = 10^(−SNR/10) (à 2 % près)."""
    rx = simulate_cpi(PARAMS, [], snr_db=snr_db)
    assert np.mean(np.abs(rx) ** 2) == pytest.approx(noise_sigma(snr_db) ** 2, rel=0.02)
    # bruit circulaire : même puissance sur les voies I et Q
    assert np.var(rx.real) == pytest.approx(np.var(rx.imag), rel=0.02)


def test_same_seed_gives_same_cube_and_different_seed_does_not():
    """Graine fixée = simulation reproductible."""
    targets = [Target(3_000.0, 5.0)]
    a = simulate_cpi(PARAMS, targets, snr_db=0.0)
    b = simulate_cpi(PARAMS, targets, snr_db=0.0)
    c = simulate_cpi(PARAMS, targets, snr_db=0.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_slow_time_phase_gives_doppler_minus_2v_over_lambda():
    """La phase de l'écho tourne de −2π·(2v/λ)/PRF par impulsion (cible qui s'éloigne)."""
    v = 4.0
    rx = simulate_cpi(PARAMS, [Target(3_000.0, v)])
    n_peak = int(np.median(np.flatnonzero(np.abs(rx[0]))))  # échantillon au milieu de l'écho
    phase_step = np.angle(rx[1, n_peak] / rx[0, n_peak])
    expected = -2 * np.pi * (2 * v / PARAMS.wavelength) / PARAMS.prf
    # le décalage du chirp entre impulsions est négligeable : même échantillon, même enveloppe
    assert np.angle(np.exp(1j * (phase_step - expected))) == pytest.approx(0.0, abs=0.05)
