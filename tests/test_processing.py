"""Validations théoriques de la chaîne de traitement : pic, gains, CFAR (jalon B)."""

import numpy as np
import pytest

from radar.processing import (
    ca_cfar_2d,
    compress_pulses,
    doppler_fft,
    doppler_window,
    range_axis,
    range_doppler_map,
    velocity_axis,
)
from radar.simulate import Target, noise_sigma, simulate_cpi
from radar.waveform import RadarParams

PARAMS = RadarParams()


def expected_cell(params, target):
    """Case (Doppler, distance) théorique, calculée sans passer par les axes du module."""
    range_cell = (target.r0 - params.r_min) / (params.c / (2 * params.fs))
    f_d = -2 * target.v / params.wavelength  # signe : cible qui s'éloigne
    doppler_cell = params.n_pulses / 2 + f_d * params.n_pulses / params.prf
    return doppler_cell, range_cell


@pytest.mark.parametrize("r0, v", [(3_000.0, 5.0), (5_000.0, -6.0), (12_345.0, 0.0)])
def test_peak_in_expected_range_and_doppler_cell(r0, v):
    """Le pic de la carte distance-Doppler tombe dans la case distance et Doppler attendue."""
    target = Target(r0, v)
    rd = range_doppler_map(simulate_cpi(PARAMS, [target]), PARAMS)
    doppler_idx, range_idx = np.unravel_index(np.argmax(np.abs(rd)), rd.shape)
    doppler_cell, range_cell = expected_cell(PARAMS, target)
    assert abs(range_idx - range_cell) <= 0.5 + 1e-9
    assert abs(doppler_idx - doppler_cell) <= 0.5 + 1e-9
    # cohérence avec les axes physiques fournis par le module
    assert abs(range_axis(PARAMS)[range_idx] - r0) <= PARAMS.range_bin / 2 + 1e-6
    assert abs(velocity_axis(PARAMS)[doppler_idx] - v) <= PARAMS.velocity_resolution / 2 + 1e-6


def test_compression_gain_is_T_times_fs():
    """Gain de compression mesuré = SNR_sortie / SNR_entrée ≈ T·fs (= 200), à 5 % près."""
    snr_db, amplitude = -10.0, 1.0
    # pic d'un écho sans bruit, calé sur un échantillon distance (retard entier en échantillons)
    r0 = PARAMS.r_min + 400 * PARAMS.range_bin
    clean = compress_pulses(simulate_cpi(PARAMS, [Target(r0, 0.0, amplitude)]), PARAMS)
    peak_power = np.abs(clean[0, 400]) ** 2
    # bruit seul compressé
    noise = compress_pulses(simulate_cpi(PARAMS, [], snr_db=snr_db), PARAMS)
    noise_power = np.mean(np.abs(noise) ** 2)

    snr_in = amplitude**2 / noise_sigma(snr_db) ** 2
    snr_out = peak_power / noise_power
    assert snr_out / snr_in == pytest.approx(PARAMS.T * PARAMS.fs, rel=0.05)


@pytest.mark.parametrize("window", ["rect", "hann"])
def test_doppler_integration_gain(window):
    """Gain d'intégration Doppler = (Σw)²/Σw² (= N sans fenêtre, 2N/3 avec Hann), à 5 % près."""
    k = 20  # case Doppler entière : v = k·Δv, avec Δv = λ·PRF/(2N)
    v = k * PARAMS.velocity_resolution
    r0 = PARAMS.r_min + 400 * PARAMS.range_bin
    clean = range_doppler_map(simulate_cpi(PARAMS, [Target(r0, v)]), PARAMS, window)
    peak_power = np.abs(clean[PARAMS.n_pulses // 2 - k, 400]) ** 2
    noise = range_doppler_map(simulate_cpi(PARAMS, [], snr_db=0.0), PARAMS, window)
    noise_power = np.mean(np.abs(noise) ** 2)

    gain_total = peak_power / noise_power  # SNR_entrée = 1 (0 dB, a = 1)
    w = doppler_window(PARAMS.n_pulses, window)
    gain_doppler = w.sum() ** 2 / (w**2).sum()
    if window == "rect":
        assert gain_doppler == pytest.approx(PARAMS.n_pulses)
    assert gain_total == pytest.approx(PARAMS.compression_gain * gain_doppler, rel=0.05)


def test_doppler_fft_is_centered_on_zero_frequency():
    """Un signal constant en temps lent (cible immobile) tombe dans la case N/2."""
    constant = np.ones((PARAMS.n_pulses, 4), dtype=complex)
    spectrum = np.abs(doppler_fft(constant, "rect"))
    assert np.all(np.argmax(spectrum, axis=0) == PARAMS.n_pulses // 2)


@pytest.mark.parametrize("pfa", [1e-3, 1e-4])
def test_cfar_empirical_pfa_matches_target_within_factor_2(pfa):
    """Sur bruit seul, la fréquence de fausses alarmes du CA-CFAR est dans [Pfa/2, 2·Pfa]."""
    rng = np.random.default_rng(123)
    n_alarms, n_cells = 0, 0
    for _ in range(8):
        rd = range_doppler_map(simulate_cpi(PARAMS, [], snr_db=0.0, rng=rng), PARAMS)
        detections = ca_cfar_2d(np.abs(rd) ** 2, pfa)
        n_alarms += detections.sum()
        n_cells += detections.size
    empirical = n_alarms / n_cells
    assert pfa / 2 <= empirical <= 2 * pfa, f"Pfa visée {pfa}, mesurée {empirical:.2e}"


def test_cfar_pfa_excess_comes_from_correlated_cells():
    """La formule α = n(Pfa^(−1/n) − 1) suppose des cellules de référence indépendantes.

    Avec fs = B et une fenêtre rectangulaire, les cases sont indépendantes : la Pfa mesurée tombe sur la visée.
    Avec fs = 2B et Hann (réglage par défaut), les cases voisines sont corrélées, le bruit de référence est moins
    bien moyenné et la Pfa dépasse nettement la visée : c'est la cause de l'excès, pas une erreur de seuil.
    """
    pfa = 1e-3

    def measured(oversampling, window):
        params = RadarParams(oversampling=oversampling, n_fast=int(2000 * oversampling))
        rng = np.random.default_rng(7)
        n_alarms = n_cells = 0
        for _ in range(6):
            rd = range_doppler_map(simulate_cpi(params, [], snr_db=0.0, rng=rng), params, window)
            detections = ca_cfar_2d(np.abs(rd) ** 2, pfa)
            n_alarms += detections.sum()
            n_cells += detections.size
        return n_alarms / n_cells / pfa

    assert measured(1.0, "rect") == pytest.approx(1.0, abs=0.12)
    assert measured(2.0, "hann") > 1.2


def test_cfar_detects_target_and_stays_quiet_far_from_it():
    """Une cible forte est détectée ; aucune détection loin de sa case (hors lobes secondaires)."""
    target = Target(3_000.0, 5.0)
    rng = np.random.default_rng(7)
    rd = range_doppler_map(simulate_cpi(PARAMS, [target], snr_db=-15.0, rng=rng), PARAMS)
    detections = ca_cfar_2d(np.abs(rd) ** 2, pfa=1e-6)
    doppler_cell, range_cell = expected_cell(PARAMS, target)
    assert detections[int(round(doppler_cell)), int(round(range_cell))]
    doppler_idx, range_idx = np.nonzero(detections)
    far = np.abs(range_idx - range_cell) > 30
    assert far.sum() <= 1  # au plus une fausse alarme isolée (Pfa·256 000 cases ≈ 0,26)


def test_cfar_threshold_adapts_to_noise_level():
    """CFAR : multiplier la puissance du bruit par 100 ne change pas les détections (seuil ∝ bruit)."""
    rd = range_doppler_map(simulate_cpi(PARAMS, [], snr_db=0.0), PARAMS)
    power = np.abs(rd) ** 2
    np.testing.assert_array_equal(ca_cfar_2d(power, 1e-3), ca_cfar_2d(100.0 * power, 1e-3))
