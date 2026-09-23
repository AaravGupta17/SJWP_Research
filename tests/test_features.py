import numpy as np
import pytest

import features as F
from gccphat import gcc_phat, score_sample, MAX_LAG_SAMPLES


def test_gcc_phat_recovers_known_delay_and_sign():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(2400)
    d = 17
    x1 = src[200:2200]                 # sensor 1 hears it d samples later
    x2 = src[200 + d:2200 + d]
    peak, lag = gcc_phat(x1, x2, MAX_LAG_SAMPLES)
    assert abs(lag) == d
    # swapping the channels flips the sign of the lag
    _, lag_swapped = gcc_phat(x2, x1, MAX_LAG_SAMPLES)
    assert lag_swapped == -lag
    assert peak > 0.3


def test_gcc_position_moves_toward_the_sensor_that_hears_first():
    rng = np.random.default_rng(1)
    src = rng.standard_normal(2400)
    ch2_first = np.stack([src[0:2000], src[10:2010]])      # ch2 hears it 10 samples earlier
    ch1_first = np.stack([src[10:2010], src[0:2000]])
    _, pos_a = score_sample(ch2_first)
    _, pos_b = score_sample(ch1_first)
    assert pos_a != pos_b
    assert (pos_a - 0.5) * (pos_b - 0.5) < 0     # opposite sides of the midpoint


def test_rms_and_dc():
    w = np.zeros((2, 2, 100), np.float32)
    w[0] += 2.0
    w[1, 0] = np.tile([1.0, -1.0], 50)
    assert F.rms(w) == pytest.approx([2.0, np.sqrt(0.5)])
    assert F.dc_offset(w) == pytest.approx([2.0, 0.0])


def test_band_energy_peaks_in_the_right_band():
    fs, t = 5000, np.arange(2000) / 5000
    tone = np.sin(2 * np.pi * 1000 * t)                     # 1 kHz
    w = np.stack([tone, tone])[None].astype(np.float32)
    be = F.band_energy(w, fs)[0]
    edges = F.BAND_EDGES_HZ
    band = next(i for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) if lo <= 1000 < hi)
    assert np.argmax(be) == band
