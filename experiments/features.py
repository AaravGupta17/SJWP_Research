"""
features.py — classical (non-learned) features for 2-channel windows
=====================================================================
Used as honest comparison points for AcousticLeakNet:
  - rms          : overall loudness (the "louder = leak" detector)
  - dc           : mean absolute channel offset (checks the DC shortcut)
  - band_energy  : log energy in fixed frequency bands (input to a small
                   logistic-regression baseline trained on real data)
  - gcc_peak     : GCC-PHAT peak height, the standard TDOA coherence score
"""

import numpy as np
from scipy.signal import welch

import _common  # noqa: F401  (puts baselines/ and model_C/ on sys.path)
from gccphat import gcc_phat, MAX_LAG_SAMPLES   # baselines/gccphat.py

BAND_EDGES_HZ = (10, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500)


def rms(windows: np.ndarray) -> np.ndarray:
    w = windows.astype(np.float64)
    return np.sqrt(np.mean(w ** 2, axis=(1, 2)))


def dc_offset(windows: np.ndarray) -> np.ndarray:
    return np.abs(windows.astype(np.float64).mean(axis=2)).mean(axis=1)


def band_energy(windows: np.ndarray, fs: int = 5000,
                edges=BAND_EDGES_HZ) -> np.ndarray:
    """(N, 2, T) -> (N, n_bands) log10 band power, averaged over channels."""
    f, pxx = welch(windows.astype(np.float64), fs=fs, nperseg=256, axis=-1)
    pxx = pxx.mean(axis=1)                               # (N, n_freq)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        out.append(np.log10(pxx[:, m].sum(axis=1) + 1e-20))
    return np.stack(out, axis=1)


def gcc_peak(windows: np.ndarray, max_lag: int = MAX_LAG_SAMPLES):
    """Returns (peak_height, peak_lag_samples), each shape (N,)."""
    peaks = np.empty(len(windows))
    lags = np.empty(len(windows), dtype=int)
    for i, w in enumerate(windows):
        peaks[i], lags[i] = gcc_phat(w[0].astype(np.float64), w[1].astype(np.float64), max_lag)
    return peaks, lags
