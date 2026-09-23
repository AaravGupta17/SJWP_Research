"""
mendeley.py — loading the Mendeley pipe-testbed recordings
===========================================================
Directory layout expected (as used by model_C/mendely_eval.py):

    <root>/<Topology>/<Condition>/<files>
      Topology  : Looped | Branched
      Condition : Orifice Leak | Longitudinal Crack | Circumferential Crack |
                  Gasket Leak | No-leak
      files     : accelerometer  -> *A1*.csv / *A2*.csv  (time, ..., value)
                  hydrophone     -> *H1*.raw / *H2*.raw  (int32 PCM, 8 kHz)

Each A1/A2 (or H1/H2) pair is one *recording*. Windows cut from the same
recording are correlated, so the recording id is carried through as the
grouping variable for splits and bootstrap CIs.

Noise-bank contamination
------------------------
Model C/D training used Mendeley *Branched no-leak* hydrophone recordings
as the background noise (model_C/dataset_c.py TRAIN_NOISE_FILES). The
accelerometer recordings of the same runs are a different sensor but the
same physical events. We therefore flag EVERY Branched no-leak recording
as `in_noise_bank` (conservative), and the clean zero-shot test uses
Looped recordings only.

Input scaling (the cause of the original ~99% false-alarm rate)
---------------------------------------------------------------
Training divided every synthetic window by a fixed reference
(10 x background-noise RMS), so leak-free training windows had RMS ~0.1.
The original real-data evaluation z-scored every window (RMS = 1.0), and
experiments/loudness_probe.py shows the checkpoints call ANY input with
RMS >= 0.1 a leak. Modes here:
  "zscore" — legacy per-window joint z-score (reproduces the old result)
  "fixed"  — divide by 10 x RMS of *calibration* no-leak windows, clip
             to +-10: the same convention as training. Calibration windows
             must never come from the test recordings.
"""

import hashlib
import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

LEAK_TYPES = ("Orifice Leak", "Longitudinal Crack", "Circumferential Crack", "Gasket Leak")
NO_LEAK = "No-leak"
TOPOLOGIES = ("Looped", "Branched")
HYDROPHONE_FS = 8000
TRAIN_REF_MULTIPLIER = 10.0     # dataset_c.py: ref_scale = noise_rms * 10
TRAIN_CLIP = 10.0               # dataset_c.py: np.clip(..., -10, 10)


@dataclass(frozen=True)
class Recording:
    topology: str
    condition: str              # leak type or "No-leak"
    ch1: Path
    ch2: Path

    @property
    def label(self) -> int:
        return 0 if self.condition == NO_LEAK else 1

    @property
    def rec_id(self) -> str:
        return f"{self.topology}/{self.condition}/{sensor_stem(self.ch1.name)}"

    @property
    def in_noise_bank(self) -> bool:
        return self.topology == "Branched" and self.condition == NO_LEAK


def sensor_stem(name: str) -> str:
    """'BR_NL_0.18 LPS_N_A1.csv' -> 'BR_NL_0.18 LPS_N' (sensor tag removed)."""
    stem = Path(name).stem
    return re.sub(r"[_\- ]?(A|H)[12]$", "", stem)


def flow_condition(rec_id: str) -> str:
    """Operating condition from a recording id.
    'Looped/No-leak/LO_NL_0.18 LPS' -> '0.18 LPS'; '..._ND' -> 'ND';
    '..._Transient' -> 'Transient'. Hydrophone ids may carry an extra
    '_N'/'_NN' suffix, which is ignored."""
    parts = rec_id.split("/")[-1].split("_")
    return parts[2] if len(parts) > 2 else "unknown"


def _partner(path: Path, tag1: str, tag2: str) -> Path:
    # replace only the LAST occurrence of the channel tag
    i = path.name.rfind(tag1)
    return path.with_name(path.name[:i] + tag2 + path.name[i + len(tag1):])


def discover(root: Path, sensor: str = "accelerometer") -> list:
    """Find all channel-paired recordings under root."""
    if sensor == "accelerometer":
        ext, t1, t2 = ".csv", "A1", "A2"
    elif sensor == "hydrophone":
        ext, t1, t2 = ".raw", "H1", "H2"
    else:
        raise ValueError(sensor)
    recs = []
    for topo in TOPOLOGIES:
        for cond in LEAK_TYPES + (NO_LEAK,):
            folder = Path(root) / topo / cond
            if not folder.is_dir():
                continue
            for f1 in sorted(folder.glob(f"*{t1}*{ext}")):
                f2 = _partner(f1, t1, t2)
                if f2.exists():
                    recs.append(Recording(topo, cond, f1, f2))
    return recs


# ── Signal loading ─────────────────────────────────────────────────────────────

def _read_accelerometer_csv(path: Path):
    df = pd.read_csv(path, header=0, low_memory=False)
    t = df.iloc[:, 0].to_numpy(np.float64)
    fs = int(round(1.0 / np.median(np.diff(t[:1000]))))
    return df.iloc[:, -1].to_numpy(np.float64), fs


def _read_hydrophone_raw(path: Path):
    return np.fromfile(path, dtype=np.int32).astype(np.float64), HYDROPHONE_FS


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def load_recording(rec: Recording, fs_out: int = 5000, cache_dir: Path = None):
    """Returns (ch1, ch2) resampled to fs_out, raw amplitude (no normalisation).
    Optionally caches the resampled pair as .npy to avoid re-parsing CSVs."""
    cache_file = None
    if cache_dir is not None:
        h = hashlib.md5(f"{rec.ch1.resolve()}|{fs_out}".encode()).hexdigest()[:12]
        cache_file = Path(cache_dir) / f"{sensor_stem(rec.ch1.name)}_{h}.npy"
        if cache_file.exists():
            arr = np.load(cache_file)
            return arr[0], arr[1]
    reader = _read_accelerometer_csv if rec.ch1.suffix == ".csv" else _read_hydrophone_raw
    x1, fs1 = reader(rec.ch1)
    x2, fs2 = reader(rec.ch2)
    x1, x2 = _resample(x1, fs1, fs_out), _resample(x2, fs2, fs_out)
    n = min(len(x1), len(x2))
    x1, x2 = x1[:n].astype(np.float32), x2[:n].astype(np.float32)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, np.stack([x1, x2]))
    return x1, x2


def to_windows(x1: np.ndarray, x2: np.ndarray, length: int = 2000,
               file_norm: str = "none") -> np.ndarray:
    """Non-overlapping (N, 2, length) windows.
    file_norm="max" divides each channel by its whole-file max |x| (the
    legacy behaviour, which erases loudness differences between files)."""
    if file_norm == "max":
        x1 = x1 / (np.abs(x1).max() or 1.0)
        x2 = x2 / (np.abs(x2).max() or 1.0)
    elif file_norm != "none":
        raise ValueError(file_norm)
    n = min(len(x1), len(x2)) // length
    if n == 0:
        return np.zeros((0, 2, length), np.float32)
    w = np.stack([x1[:n * length].reshape(n, length), x2[:n * length].reshape(n, length)], axis=1)
    return w.astype(np.float32)


# ── Model-input scaling ────────────────────────────────────────────────────────

def zscore_windows(w: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Legacy: per-window joint z-score (model_C/mendely_eval.py normalize_joint)."""
    mu = w.mean(axis=(1, 2), keepdims=True)
    sd = w.std(axis=(1, 2), keepdims=True) + 1e-8
    return np.clip((w - mu) / sd, -clip, clip).astype(np.float32)


def calibrate_ref_scale(no_leak_windows: np.ndarray) -> float:
    """Training convention: ref = 10 x mean per-window RMS of background noise."""
    if len(no_leak_windows) == 0:
        raise ValueError("need calibration no-leak windows")
    per_window_rms = np.sqrt(np.mean(no_leak_windows.astype(np.float64) ** 2, axis=(1, 2)))
    return float(TRAIN_REF_MULTIPLIER * per_window_rms.mean())


def fixed_scale_windows(w: np.ndarray, ref_scale: float) -> np.ndarray:
    return np.clip(w / ref_scale, -TRAIN_CLIP, TRAIN_CLIP).astype(np.float32)


# ── Convenience: a whole windowed dataset ──────────────────────────────────────

@dataclass
class WindowSet:
    x: np.ndarray            # (N, 2, T) raw-amplitude windows
    y: np.ndarray            # (N,) 1 = leak
    group: np.ndarray        # (N,) recording id
    topology: np.ndarray     # (N,)
    condition: np.ndarray    # (N,)
    noise_bank: np.ndarray   # (N,) bool

    def subset(self, mask) -> "WindowSet":
        return WindowSet(self.x[mask], self.y[mask], self.group[mask],
                         self.topology[mask], self.condition[mask], self.noise_bank[mask])

    def __len__(self):
        return len(self.y)


def load_windowset(root: Path, sensor: str = "accelerometer", fs_out: int = 5000,
                   length: int = 2000, file_norm: str = "none",
                   cache_dir: Path = None, verbose: bool = True) -> WindowSet:
    recs = discover(root, sensor)
    if not recs:
        raise FileNotFoundError(f"No {sensor} recordings found under {root}")
    xs, ys, gs, ts, cs, nb = [], [], [], [], [], []
    for r in recs:
        x1, x2 = load_recording(r, fs_out, cache_dir)
        w = to_windows(x1, x2, length, file_norm)
        if verbose:
            print(f"  {r.rec_id:60s} {len(w):5d} windows")
        xs.append(w)
        n = len(w)
        ys += [r.label] * n
        gs += [r.rec_id] * n
        ts += [r.topology] * n
        cs += [r.condition] * n
        nb += [r.in_noise_bank] * n
    return WindowSet(np.concatenate(xs), np.array(ys), np.array(gs), np.array(ts),
                     np.array(cs), np.array(nb, dtype=bool))
