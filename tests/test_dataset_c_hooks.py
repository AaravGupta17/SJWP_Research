"""The two experiment hooks added to model_C/dataset_c.py must (a) leave the
default behaviour unchanged and (b) do what E3 needs."""

import numpy as np
import pytest

import dataset_c


@pytest.fixture
def ds(monkeypatch):
    class _Bank:                              # quiet stand-in for the Mendeley noise bank
        noise_rms = 0.01
        def get_window(self):
            return np.random.normal(0, 0.01, (2, 2000)).astype(np.float32)
    d = dataset_c.LeakDataset.__new__(dataset_c.LeakDataset)
    d.signal_length, d.fs, d.augment = 2000, 5000, False
    d.noise_bank = _Bank()
    d.snr_override_db, d.include_pressure_dc = None, True
    return d


def _leak(**kw):
    c = dict(leak_status=1, wave_speed=1200.0, alpha=0.001, pipe_length=100.0,
             pipe_diameter=0.15, pipe_roughness=100.0, pipe_material="CI",
             flow_velocity=0.5, flow_rate=5.0, p_left=40.0, p_right=40.0,
             d_left=30.0, d_right=50.0, leak_flow=1.0, leak_area=1e-5,
             torricelli_amp=1e-4, pressure=40.0, leak_pos=0.375, pos_valid=1.0,
             demand=1.0, sensor_left=0.0, sensor_right=80.0)
    c.update(kw)
    return c


def test_pressure_dc_toggle(ds):
    np.random.seed(0)
    with_dc = ds.generate_signal(_leak(leak_status=0))
    ds.include_pressure_dc = False
    np.random.seed(0)
    without = ds.generate_signal(_leak(leak_status=0))
    assert abs(with_dc.mean()) > 5 * abs(without.mean())


def test_snr_override_controls_leak_loudness(ds):
    ds.include_pressure_dc = False
    rms = {}
    for snr in (-20, 20):
        ds.snr_override_db = snr
        np.random.seed(1)
        rms[snr] = np.sqrt((ds.generate_signal(_leak()) ** 2).mean())
    assert rms[20] > 3 * rms[-20]


def test_default_is_unchanged_from_original_formula(ds):
    """snr_override_db=None must keep the Torricelli-mapped SNR."""
    np.random.seed(2)
    a = ds.generate_signal(_leak())
    ds.snr_override_db = None
    np.random.seed(2)
    b = ds.generate_signal(_leak())
    assert np.array_equal(a, b)
