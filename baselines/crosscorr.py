"""
crosscorr.py — Classical Cross-Correlation Baseline
=====================================================
Cross-correlate the two sensor channels, use the peak correlation
magnitude as a detection score, and read the leak position off the
peak lag (TDOA). No training. No pipe metadata at inference — matches
AcousticLeakNet's test-time condition in evaluate.py (scalars zeroed).

Run from scripts/ (or wherever evaluate.py runs from):
    python ../baselines/crosscorr.py
"""

import numpy as np
from tqdm import tqdm

from common import (
    CACHE_ROOT,
    TEST_SPLITS, load_split_arrays, compute_metrics, print_summary,
    save_results, ASSUMED_WAVE_SPEED_MPS, SAMPLING_RATE_HZ,
    MEAN_SENSOR_SEPARATION_M,
)

# Search window for the correlation peak — a leak signal can't arrive
# more than (separation / slowest expected wave speed) apart. PVC (400
# m/s, see dataset.py MATERIAL_ACOUSTIC) is the slowest material.
_SLOWEST_WAVE_SPEED_MPS = 400.0
MAX_LAG_SAMPLES = int(np.ceil(
    (MEAN_SENSOR_SEPARATION_M / _SLOWEST_WAVE_SPEED_MPS) * SAMPLING_RATE_HZ
))


def score_sample(sig: np.ndarray):
    """
    sig: (2, T). Returns (detection_score, position_estimate in [0,1]).
    """
    x1, x2 = sig[0], sig[1]
    T = len(x1)

    full_corr = np.correlate(x1, x2, mode="full")
    lags = np.arange(-(T - 1), T)

    window = np.abs(lags) <= MAX_LAG_SAMPLES
    corr_win = full_corr[window]
    lags_win = lags[window]

    norm = (np.linalg.norm(x1) * np.linalg.norm(x2)) + 1e-8
    corr_win_normalised = corr_win / norm

    peak_idx = np.argmax(np.abs(corr_win_normalised))
    peak_score = float(np.abs(corr_win_normalised[peak_idx]))
    peak_lag = int(lags_win[peak_idx])

    delay_s = peak_lag / SAMPLING_RATE_HZ
    delta_d = ASSUMED_WAVE_SPEED_MPS * delay_s
    pos_est = 0.5 - delta_d / (2 * MEAN_SENSOR_SEPARATION_M)
    pos_est = float(np.clip(pos_est, 0.0, 1.0))

    return peak_score, pos_est


def run_split(split: str) -> dict:
    sig, det, pos, sev, pos_valid = load_split_arrays(split)

    scores = np.zeros(len(sig), dtype=np.float32)
    pos_pred = np.zeros(len(sig), dtype=np.float32)

    for i in tqdm(range(len(sig)), desc=f"Cross-correlation on {split}",
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
        print_summary("Cross-Correlation", all_results)
        save_results("crosscorr", all_results)


if __name__ == "__main__":
    main()