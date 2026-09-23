"""Helpers used by E6 (data overview), E7 (texture probe), E8 (realism check)."""

import numpy as np
import pytest

import mendeley as M
from realism_check import shape_features
from texture_probe import phase_randomise, rescale_rms


@pytest.mark.parametrize("rec, cond", [
    ("Looped/No-leak/LO_NL_0.18 LPS", "0.18 LPS"),
    ("Branched/Gasket Leak/BR_GL_ND", "ND"),
    ("Branched/No-leak/BR_NL_Transient_NN", "Transient"),
])
def test_flow_condition(rec, cond):
    assert M.flow_condition(rec) == cond


def test_rescale_rms_exact():
    w = np.random.default_rng(0).normal(0, 3, (5, 2, 2000)).astype(np.float32)
    out = rescale_rms(w, 0.1)
    assert np.allclose(np.sqrt((out.astype(np.float64) ** 2).mean(axis=(1, 2))), 0.1, rtol=1e-4)


def test_phase_randomise_keeps_spectrum_and_shared_phase_keeps_cross_phase():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(2100)
    w = np.stack([s[:2000], s[50:2050]])[None].astype(np.float32)    # channel 2 leads by 50
    for shared in (True, False):
        out = phase_randomise(w, np.random.default_rng(1), shared=shared)
        X, Y = np.fft.rfft(w, axis=-1), np.fft.rfft(out, axis=-1)
        assert np.allclose(np.abs(X), np.abs(Y), rtol=1e-3, atol=1e-3)
    out = phase_randomise(w, np.random.default_rng(1), shared=True)
    cross_in = np.angle(np.fft.rfft(w[0, 0]) * np.conj(np.fft.rfft(w[0, 1])))
    cross_out = np.angle(np.fft.rfft(out[0, 0]) * np.conj(np.fft.rfft(out[0, 1])))
    assert np.allclose(np.exp(1j * cross_in), np.exp(1j * cross_out), atol=1e-3)


def test_realism_features_ignore_loudness():
    w = np.random.default_rng(0).normal(0, 1, (4, 2, 2000))
    assert np.allclose(shape_features(w), shape_features(7.5 * w), atol=1e-6)
