"""Public-dataset loaders: grouping (no site in both train and test), the
common band limit, and loudness-free features."""

import numpy as np
import pytest

import public_data as P
from cross_dataset import features_1ch, standardise


@pytest.mark.parametrize("stem, site", [
    ("003_05593-20180927", "05593"),
    ("004_07573-20180817", "07573"),
    ("1.3.02.0345", "h1.3"),
    ("2.3.05.0430", "h2.3"),
    ("00241_20180818_1445", "00241"),
])
def test_hk_site(stem, site):
    assert P.hk_site(stem) == site


def test_hk_repeat_recordings_share_a_site():
    assert P.hk_site("1.3.01.0330") == P.hk_site("1.3.04.0415")


@pytest.mark.parametrize("stem, rec", [
    ("NA-NA-NA-NA-hydrophone_0-1", "NA-NA-NA-NA-hydrophone"),
    ("NA-NA-NA-NA-hydrophone_0-1_1", "NA-NA-NA-NA-hydrophone"),
    ("pe-zone 2-0.32 MPa-0.95 ms-noise logger_3-4", "pe-zone 2-0.32 MPa-0.95 ms-noise logger"),
    ("NA-NA-0.1545 MPa-3.36 ms-hydrophone-0-1", "NA-NA-0.1545 MPa-3.36 ms-hydrophone"),
])
def test_dongguan_clips_of_one_recording_share_a_group(stem, rec):
    assert P.dongguan_recording(stem) == rec


def test_common_format_is_band_limited_and_5khz():
    fs = 25600
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 3500 * t)   # 3.5 kHz must be removed
    y = P.to_common(x, fs)
    assert len(y) == pytest.approx(5000, abs=2)
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / 5000)
    assert spec[np.argmin(abs(f - 500))] > 100 * spec[np.argmin(abs(f - 2400)):].max()


def test_windows_and_concat():
    a = P._pack([(np.zeros(4500, np.float32), 1, "g1", "d", "m"),
                 (np.zeros(2000, np.float32), 0, "g2", "d", "m")])
    assert a.x.shape == (3, 2000) and list(a.y) == [1, 1, 0]
    b = P.Windows.concat([a, a.subset(a.y == 0)])
    assert len(b) == 4


def test_features_are_loudness_free():
    x = np.random.default_rng(0).normal(0, 1, (6, 2000))
    assert np.allclose(features_1ch(standardise(x)), features_1ch(standardise(50 * x)), atol=1e-5)


def test_mendeley_branched_is_dropped_but_other_datasets_kept():
    meta = np.array(["Branched/0.18 LPS", "Looped/ND", "Branched/ND", "Branched/pvc"])
    ds = np.array(["mendeley_hyd", "mendeley_hyd", "mendeley_acc", "dongguan"])
    w = P.Windows(np.zeros((4, 8), np.float32), np.array([0, 1, 1, 0]),
                  np.array(["a", "b", "c", "d"]), ds, meta)
    kept = P.drop_mendeley_branched(w)
    assert list(kept.group) == ["b", "d"]
