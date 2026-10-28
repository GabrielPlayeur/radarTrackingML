"""Validations théoriques de la forme d'onde (jalon B)."""

import dataclasses

import numpy as np
import pytest

from radar.waveform import (
    RadarParams,
    autocorrelation_width_3db,
    chirp_at,
    reference_chirp,
)


@pytest.mark.parametrize("bandwidth", [10e6, 20e6])
def test_autocorrelation_width_is_0_886_over_B(bandwidth):
    """La largeur à −3 dB de l'autocorrélation du chirp vaut 0,886/B (à 5 % près)."""
    params = RadarParams(B=bandwidth)
    width = autocorrelation_width_3db(params)
    assert width == pytest.approx(0.886 / bandwidth, rel=0.05)


def test_chirp_has_unit_modulus_inside_and_zero_outside():
    """Le chirp est de module 1 dans [0, T[ et nul ailleurs."""
    params = RadarParams()
    t = np.array([-1e-6, 0.0, params.T / 2, params.T - 1e-9, params.T, 2 * params.T])
    modulus = np.abs(chirp_at(t, params))
    np.testing.assert_allclose(modulus, [0, 1, 1, 1, 0, 0])


def test_reference_chirp_occupies_T_times_fs_samples():
    """La réplique contient exactement L = T·fs échantillons non nuls."""
    params = RadarParams()
    ref = reference_chirp(params)
    assert len(ref) == params.n_fast
    assert np.count_nonzero(ref) == params.n_pulse_samples == 200


def test_reference_chirp_rejects_too_short_window():
    """Une fenêtre plus courte que le chirp est refusée."""
    with pytest.raises(ValueError):
        reference_chirp(dataclasses.replace(RadarParams(), n_fast=100))


def test_formula_table_in_X_band():
    """Le tableau « formules à connaître » en bande X (B = 10 MHz, PRF = 1 kHz, N = 64)."""
    p = RadarParams()
    assert p.wavelength == pytest.approx(0.032, rel=0.01)
    assert p.range_resolution == pytest.approx(15.0)
    assert p.unambiguous_range == pytest.approx(150e3)
    assert p.unambiguous_velocity == pytest.approx(8.0, rel=0.01)
    assert p.velocity_resolution == pytest.approx(0.25, rel=0.01)
    assert 2 * 10.0 / p.wavelength == pytest.approx(625.0, rel=0.01)  # f_d pour v = 10 m/s
    assert p.B * p.T == pytest.approx(100.0)
