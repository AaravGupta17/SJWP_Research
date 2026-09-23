"""
gccphat.py — Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
=============================================================================
The standard TDOA baseline in acoustic source-localization (Knapp &
Carter, 1976). Whitens the cross-power spectrum before correlating —
each frequency bin contributes by phase only, which sharpens the peak
and is more robust to the coloured noise this dataset simulates than
raw cross-correlation.

Run the same way as crosscorr.py:
    python ../baselines/gccphat.py
"""

import numpy as np
from tqdm import tqdm

from common import (
    CACHE_ROOT,
    TEST_SPLITS, load_split_arrays, compute_metrics, print_summary,
    save_results, ASSUMED_WAVE_SPEED_MPS, SAMPLING_RATE_HZ,
    MEAN_SENSOR_SEPARATION_M,
)

_SLOWEST_WAVE_SPEED_MPS = 400.0  # PVC — see crosscorr.py for rationale
MAX_LAG_SAMPLES = int(np.ceil(
    (MEAN_SENSOR_SEPARATION_M / _SLOWEST_WAVE_SPEED_MPS) * SAMPLING_RATE_HZ
))


def gcc_phat(x1: np.ndarray, x2: np.ndarray, max_lag: int):
    """Returns (peak_score, peak_lag_samples)."""
    n = len(x1) + len(x2)
    n_fft = 1 << (n - 1).bit_length()  # next power of 2, for speed

    X1 = np.fft.rfft(x1, n=n_fft)
    X2 = np.fft.rfft(x2, n=n_fft)

    cross = X1 * np.conj(X2)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cross_phat = cross / denom

    corr = np.fft.irfft(cross_phat, n=n_fft)
    corr = np.concatenate((corr[-(len(x1) - 1):], corr[:len(x1)]))
    lags = np.arange(-(len(x1) - 1), len(x1))

    window = np.abs(lags) <= max_lag
    corr_win = corr[window]
    lags_win = lags[window]

    peak_idx = np.argmax(np.abs(corr_win))
    return float(np.abs(corr_win[peak_idx])), int(lags_win[peak_idx])


def score_sample(sig: np.ndarray):
    x1, x2 = sig[0], sig[1]
    peak_score, peak_lag = gcc_phat(x1, x2, MAX_LAG_SAMPLES)

    delay_s = peak_lag / SAMPLING_RATE_HZ
    delta_d = ASSUMED_WAVE_SPEED_MPS * delay_s
    pos_est = 0.5 - delta_d / (2 * MEAN_SENSOR_SEPARATION_M)
    pos_est = float(np.clip(pos_est, 0.0, 1.0))

    return peak_score, pos_est


def run_split(split: str) -> dict:
    sig, det, pos, sev, pos_valid = load_split_arrays(split)

    scores = np.zeros(len(sig), dtype=np.float32)
    pos_pred = np.zeros(len(sig), dtype=np.float32)

    for i in tqdm(range(len(sig)), desc=f"GCC-PHAT on {split}",
                  unit="sample", dynamic_ncols=True):
        scores[i], pos_pred[i] = score_sample(sig[i])

    return compute_metrics(split, scores, det, pos_pred, pos, pos_valid)


def main():
    all_results = []
    for split in TEST_SPLITS:
        try:
            res = run_split(split)
        except FileNotFoundError:
            print(f"Cache not found for {split} in {CACHE_ROOT} — run the pregen script first")
            continue
        all_results.append(res)

    if all_results:
        print_summary("GCC-PHAT", all_results)
        save_results("gccphat", all_results)


if __name__ == "__main__":
    main()