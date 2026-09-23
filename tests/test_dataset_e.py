"""Model E synthesiser: the physics fixes must do what the docstring claims."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model_E"))
import dataset_c                                          # noqa: E402
import dataset_e as E                                     # noqa: E402

ALL_OFF = {k: False for k in E.REALISM_DEFAULTS}


class _SilentBank:
    noise_rms = 0.01
    def get_window(self):
        return np.zeros((2, 2000), np.float32)


def _make(cls, realism=None):
    d = cls.__new__(cls)
    d.signal_length, d.fs, d.augment = 2000, 5000, False
    d.noise_bank = _SilentBank()
    d.snr_override_db, d.include_pressure_dc = 10, False
    if cls is E.LeakDatasetE:
        d.realism = {**E.REALISM_DEFAULTS, **(realism or {})}
    return d


def _leak_cfg(dl, dr, **kw):
    c = dict(leak_status=1, wave_speed=1200.0, alpha=0.001, pipe_length=100.0,
             pipe_roughness=100.0, pipe_material="CI", flow_velocity=0.0, demand=0.0,
             p_left=40.0, p_right=40.0, d_left=dl, d_right=dr, torricelli_amp=1e-4)
    c.update(kw)
    return c


def _xcorr_lag(a, b, maxlag=600):
    full = np.correlate(a, b, "full")
    lags = np.arange(-len(a) + 1, len(a))
    m = np.abs(lags) <= maxlag
    return lags[m][np.argmax(full[m])]


def _tdoa_tracking(gen, n=60):
    rng = np.random.default_rng(0)
    true, est = [], []
    for i in range(n):
        dl, dr = rng.uniform(5, 60, 2)
        np.random.seed(i)
        x = gen(_leak_cfg(dl, dr)).astype(np.float64)
        true.append((dl - dr) / 1200 * 5000)
        est.append(_xcorr_lag(x[0], x[1]))
    return np.corrcoef(true, est)[0, 1]


def test_model_c_synthetic_leak_has_no_recoverable_tdoa():
    """Documents the Model C finding (INTEGRITY_LOG): the zero-delay shared
    component hides the time delay from cross-correlation."""
    assert abs(_tdoa_tracking(_make(dataset_c.LeakDataset).generate_signal)) < 0.3


def test_model_e_delay_fix_makes_tdoa_recoverable():
    d = _make(E.LeakDatasetE, {**ALL_OFF, "delay": True})
    assert _tdoa_tracking(d.generate_signal) > 0.95


def test_model_e_without_delay_fix_reproduces_c_propagation():
    d = _make(E.LeakDatasetE, ALL_OFF)
    assert abs(_tdoa_tracking(d.generate_signal)) < 0.3


def test_fractional_delay_is_linear_not_circular():
    T, pad = 2000, 2000
    src = np.zeros(T + 2 * pad)
    src[pad + T - 5] = 1.0                    # impulse near the END of the window
    out = E.fractional_delay(src, 20, T, pad)
    assert np.abs(out[:100]).max() < 1e-6     # no wrap-around to the start
    assert np.argmax(np.abs(E.fractional_delay(src, -20, T, pad))) == T - 25


def test_no_dc_offset_by_default():
    np.random.seed(0)
    d = _make(E.LeakDatasetE, {**ALL_OFF, "no_dc": True})
    x = d.generate_signal(_leak_cfg(0, 0, leak_status=0))
    assert np.abs(x.mean()) < 1e-6


def test_background_is_diverse():
    """With [E2] on, the no-leak class is not only the noise-bank texture."""
    d = _make(E.LeakDatasetE, {**ALL_OFF, "background": True})
    np.random.seed(1)
    windows = [d.generate_signal(_leak_cfg(0, 0, leak_status=0)) for _ in range(40)]
    non_silent = sum(np.abs(w).max() > 0 for w in windows)
    assert 10 < non_silent < 40               # mix of bank (silent stand-in) and coloured noise


def test_all_switches_on_runs_and_is_bounded():
    d = _make(E.LeakDatasetE)
    d.snr_override_db = None
    for i in range(20):
        np.random.seed(i)
        x = d.generate_signal(_leak_cfg(30, 50) if i % 2 else _leak_cfg(0, 0, leak_status=0))
        assert x.shape == (2, 2000) and np.isfinite(x).all() and np.abs(x).max() <= 10


def test_leak_rows_without_source_are_dropped(tmp_path, monkeypatch):
    cols = dict(Pipe_Length_m=100, Pipe_Diameter_m=0.15, Pipe_Roughness=100, Pipe_Material="CI",
                Avg_Flow_Velocity_mps=0.5, Avg_Flow_Rate_lps=5, Acoustic_Propagation_Speed_mps=1200,
                Attenuation_Alpha_per_m=0.001, Leak_Area_m2=1e-5, Avg_Pressure_at_Leak=40,
                Leak_Flow_Lps=1.0, Sensor_Positions_m="0,80", All_Sensor_Pressures_m="40,40")
    rows = [dict(cols, Leak_Status=0),
            dict(cols, Leak_Status=1, Leak_Distance_Left_m=30, Leak_Distance_Right_m=50),
            dict(cols, Leak_Status=1, Leak_Distance_Left_m=0, Leak_Distance_Right_m=0)]
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    index = tmp_path / "index.csv"
    pd.DataFrame({"file_path": [str(raw)] * 3, "row_idx": [0, 1, 2],
                  "file_type": ["base", "leak", "leak"], "demand_multiplier": [1.0] * 3}
                 ).to_csv(index, index=False)
    monkeypatch.setattr(dataset_c, "_NOISE_BANK", _SilentBank())
    base = dataset_c.LeakDataset(str(index))
    fixed = E.LeakDatasetE(str(index))
    assert len(base) == 3 and len(fixed) == 2
