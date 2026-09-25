"""Model F data pipeline: each [F*] change must do what its docstring claims."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model_F"))
sys.path.insert(0, str(ROOT / "Model_E"))
import augment_f as A                                     # noqa: E402
import dataset_e as E                                     # noqa: E402
import pregen_f                                           # noqa: E402
import train_f                                            # noqa: E402
from test_dataset_e import _leak_cfg, _make               # noqa: E402


def _power_at(x, hz):
    f = np.fft.rfftfreq(x.shape[-1], 1 / A.FS)
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2)[..., np.argmin(abs(f - hz))].mean()


def test_band_limit_keeps_passband_and_removes_above_2khz():
    x = np.random.default_rng(0).standard_normal((20, A.T))
    y = A.band_limit(x)
    assert _power_at(y, 500) / _power_at(x, 500) == pytest.approx(1, abs=0.01)
    assert _power_at(y, 2300) < 1e-6 * _power_at(x, 2300)


def test_joint_zscore_removes_loudness_but_keeps_channel_ratio():
    x = np.random.default_rng(1).standard_normal((2, A.T)) * np.array([[1.0], [0.5]])
    a, b = A.joint_zscore(x), A.joint_zscore(x * 1000)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a[0].std() / a[1].std() == pytest.approx(2, rel=0.01)


def test_random_eq_is_shared_by_both_channels():
    rng = np.random.default_rng(2)
    x = np.stack([A.coloured_noise(rng)] * 2)
    y = A.random_eq(x, rng)
    # only the small per-channel tilt differs: channels stay nearly identical
    assert np.corrcoef(y[0], y[1])[0, 1] > 0.9


def test_leak_component_is_in_background_units_and_band_limited():
    outs = []
    for rms in (0.01, 0.3):
        d = _make(E.LeakDatasetE)
        d.noise_bank.noise_rms = rms
        np.random.seed(0)
        outs.append(pregen_f.leak_component(d, _leak_cfg(10.0, 30.0)))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-4, atol=1e-6)   # float32 source
    assert _power_at(outs[0], 2400) < 1e-6 * _power_at(outs[0], 800)


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    groups = np.array([f"g{i // 4}" for i in range(n)])
    src = np.where(np.arange(n) < 20, "hongkong", "dongguan")
    np.savez(tmp_path / "bank.npz", single=rng.standard_normal((n, A.T)).astype(np.float16),
             s_y=(np.arange(n) % 2).astype(np.int8), s_group=groups, s_source=src,
             s_dataset=src, s_val=np.zeros(n, bool),
             pairs=rng.standard_normal((3, 2, A.T)).astype(np.float16),
             p_group=np.array(["m"] * 3), p_val=np.zeros(3, bool))
    d = tmp_path / "train"
    d.mkdir()
    lab = np.zeros((10, 4), np.float32)
    lab[::2, 0] = 1
    row = np.full(10, -1, np.int32)
    row[::2] = np.arange(5)
    np.save(d / "labels.npy", lab)
    np.save(d / "leak_row.npy", row)
    np.save(d / "leak.npy", (rng.standard_normal((5, 2, A.T)) * 0.5).astype(np.float16))
    return tmp_path


def test_excluded_source_is_absent_from_backgrounds_and_real_rows(cache):
    bank = train_f.Bank(cache / "bank.npz", val=False, exclude=["hongkong"])
    assert len(bank.single) == 20 and set(bank.group) == {f"g{i}" for i in range(5, 10)}


def test_items_are_reproducible_per_epoch_and_change_between_epochs(cache):
    bank = train_f.Bank(cache / "bank.npz", val=False)
    ds = train_f.MixDataset(cache, "train", bank, 0.3, seed=0)
    ds.epoch = 1
    a, b = ds[5][0], ds[5][0]
    ds.epoch = 2
    c = ds[5][0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_labels_are_balanced_in_both_real_and_synthetic_rows(cache):
    bank = train_f.Bank(cache / "bank.npz", val=False)
    ds = train_f.MixDataset(cache, "train", bank, 0.5, seed=0)
    items = [ds[i] for i in range(600)]
    y = np.array([t[1][0].item() for t in items])
    syn = np.array([t[2].item() for t in items]) == 1
    assert abs(y[syn].mean() - 0.5) < 0.08 and abs(y[~syn].mean() - 0.5) < 0.08
    assert all(t[0].shape == (2, A.T) and abs(t[0].std() - 1) < 0.05 for t in items[:50])
