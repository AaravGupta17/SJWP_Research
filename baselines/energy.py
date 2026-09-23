"""
energy.py — Signal-Energy (RMS) Detector Baseline
===================================================
The simplest possible detector: score each 2-channel window by its RMS
amplitude. No training, no timing, no frequency information.

Why this baseline matters: in the synthetic data, a leak adds energy on
top of the same background noise, so loudness alone may separate the
classes. If this one-line detector matches AcousticLeakNet's AUROC on a
test set, that test set cannot show that the network learned anything
beyond "louder = leak". Localisation is not attempted: position is
always predicted as 0.5 (the midpoint), so the reported pos MAE is the
"always guess the middle" floor that any localiser has to beat.

    python baselines/energy.py              # scores ../cache_c
    LEAKNET_CACHE=cache python baselines/energy.py
"""

import numpy as np

from common import (
    CACHE_ROOT, TEST_SPLITS, load_split_arrays, compute_metrics,
    print_summary, save_results,
)


def score_windows(sig: np.ndarray) -> np.ndarray:
    """sig: (N, 2, T) -> RMS over both channels and time, shape (N,)."""
    sig = sig.astype(np.float64)
    return np.sqrt(np.mean(sig ** 2, axis=(1, 2)))


def run_split(split: str) -> dict:
    sig, det, pos, sev, pos_valid = load_split_arrays(split)
    scores = score_windows(sig)
    pos_pred = np.full(len(sig), 0.5, dtype=np.float32)   # no localisation
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
        print_summary("RMS Energy", all_results)
        save_results("energy", all_results)


if __name__ == "__main__":
    main()
