"""Mendeley loading, labelling, contamination flags and input scaling,
tested on a tiny fake directory tree (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

import mendeley as M


def _write_csv(path, x, fs):
    t = np.arange(len(x)) / fs
    pd.DataFrame({"Time": t, "Value": x}).to_csv(path, index=False)


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    fs = 10000
    layout = [("Looped", "Orifice Leak", "LO_OL_0.47 LPS"),
              ("Looped", "No-leak", "LO_NL_0.47 LPS"),
              ("Branched", "Gasket Leak", "BR_GL_0.18 LPS"),
              ("Branched", "No-leak", "BR_NL_0.18 LPS_N")]
    for topo, cond, stem in layout:
        d = tmp_path / topo / cond
        d.mkdir(parents=True)
        amp = 0.01 if cond == "No-leak" else 0.1
        for ch in ("A1", "A2"):
            _write_csv(d / f"{stem}_{ch}.csv", rng.normal(0, amp, 9000), fs)
    (tmp_path / "Looped" / "No-leak" / "orphan_A1.csv").write_text("Time,Value\n0,0\n")
    return tmp_path


def test_discover_pairs_labels_and_noise_bank_flag(fake_root):
    recs = M.discover(fake_root, "accelerometer")
    assert len(recs) == 4                         # orphan without A2 is skipped
    by = {(r.topology, r.condition): r for r in recs}
    assert by[("Looped", "Orifice Leak")].label == 1
    assert by[("Looped", "No-leak")].label == 0
    assert by[("Branched", "No-leak")].in_noise_bank
    assert not by[("Looped", "No-leak")].in_noise_bank
    assert not by[("Branched", "Gasket Leak")].in_noise_bank


def test_sensor_stem_matches_between_sensors():
    assert M.sensor_stem("BR_NL_0.18 LPS_N_A1.csv") == "BR_NL_0.18 LPS_N"
    assert M.sensor_stem("BR_NL_0.18 LPS_N_H2.raw") == "BR_NL_0.18 LPS_N"


def test_windowset_resamples_and_groups(fake_root):
    ws = M.load_windowset(fake_root, verbose=False)
    # 9000 samples @10 kHz -> 4500 @5 kHz -> 2 windows of 2000 per recording
    assert ws.x.shape == (8, 2, 2000)
    assert len(np.unique(ws.group)) == 4
    for g in np.unique(ws.group):
        assert len(np.unique(ws.y[ws.group == g])) == 1


def test_fixed_scale_matches_training_convention(fake_root):
    ws = M.load_windowset(fake_root, verbose=False)
    calib = ws.x[(ws.topology == "Branched") & (ws.y == 0)]
    ref = M.calibrate_ref_scale(calib)
    scaled = M.fixed_scale_windows(ws.x, ref)
    rms = np.sqrt((scaled ** 2).mean(axis=(1, 2)))
    # calibration no-leak windows land at RMS ~0.1, like training background
    nl = (ws.topology == "Branched") & (ws.y == 0)
    assert rms[nl].mean() == pytest.approx(0.1, rel=0.05)
    # loudness differences are preserved (leak windows are 10x louder here)
    assert rms[ws.y == 1].mean() > 5 * rms[ws.y == 0].mean()


def test_zscore_erases_loudness(fake_root):
    ws = M.load_windowset(fake_root, verbose=False)
    z = M.zscore_windows(ws.x)
    rms = np.sqrt((z ** 2).mean(axis=(1, 2)))
    assert np.allclose(rms, 1.0, atol=0.05)       # every window looks the same loudness


def test_legacy_file_max_norm(fake_root):
    x1 = np.array([0.0, 2.0, -4.0, 1.0] * 1000, np.float32)
    w = M.to_windows(x1, x1, 2000, file_norm="max")
    assert np.abs(w).max() == pytest.approx(1.0)
