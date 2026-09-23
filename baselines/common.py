"""
common.py — Shared utilities for classical (non-learned) baselines
====================================================================
Cross-correlation and GCC-PHAT both need the same thing: load the
pre-generated test caches, run a per-sample scoring function, and
report metrics in exactly the same format as evaluate.py so numbers
drop straight into the comparison table.

Deliberately mirrors evaluate.py's CachedDataset / metric definitions —
same AUROC/F1/pos-MAE conventions, same test splits — so "baseline X
gets AUROC 0.81, AcousticLeakNet gets 0.94" is an apples-to-apples
comparison and not an artifact of different eval code.
"""

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    mean_absolute_error, r2_score, confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Which pre-generated cache to score. Defaults to Model C's cache (the
# headline model); set LEAKNET_CACHE=cache for the Model B cache, etc.
CACHE_ROOT   = REPO_ROOT / os.environ.get("LEAKNET_CACHE", "cache_c")
RESULTS_DIR  = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SPLITS = ["test_network_3", "test_network_6", "test_network_8"]
NETWORK_NAMES = {
    "test_network_3": "L-TOWN (Network 3)",
    "test_network_6": "KY15 (Network 6)",
    "test_network_8": "Richmond (Network 8)",
}

MEAN_SENSOR_SEPARATION_M = 40.4  # metres — matches evaluate.py exactly

# Fixed wave-speed assumption used by classical TDOA baselines. Neither
# method gets access to pipe metadata at inference (evaluate.py zeroes
# the scalar features too), so this is the single generic number a
# real deployed cross-correlation system would hard-code. Reported
# separately from AcousticLeakNet's material-agnostic performance —
# this constant is exactly the crutch the learned model doesn't need.
ASSUMED_WAVE_SPEED_MPS = 1200.0
SAMPLING_RATE_HZ = 5000


class CachedDataset(Dataset):
    """Identical to evaluate.py's CachedDataset — raw 2-channel signal
    plus the 4 label columns [detect, pos, severity, pos_valid]."""

    def __init__(self, split: str):
        cache_dir = CACHE_ROOT / split
        self.signals = np.load(str(cache_dir / "signals.npy"), mmap_mode="r")
        self.labels = np.load(str(cache_dir / "labels.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy().astype(np.float32)
        lab = self.labels[idx]
        return sig, lab[0], lab[1], lab[2], lab[3]


def load_split_arrays(split: str):
    """Load a whole split into memory as plain numpy arrays (classical
    baselines are cheap enough per-sample that a DataLoader adds
    nothing but overhead here)."""
    ds = CachedDataset(split)
    sig = np.stack([ds[i][0] for i in range(len(ds))])
    det = np.array([ds[i][1] for i in range(len(ds))], dtype=np.float32)
    pos = np.array([ds[i][2] for i in range(len(ds))], dtype=np.float32)
    sev = np.array([ds[i][3] for i in range(len(ds))], dtype=np.float32)
    pos_valid = np.array([ds[i][4] for i in range(len(ds))], dtype=np.float32)
    return sig, det, pos, sev, pos_valid


def compute_metrics(split: str, scores: np.ndarray, det_true: np.ndarray,
                     pos_pred: np.ndarray, pos_true: np.ndarray,
                     pos_valid: np.ndarray,
                     sev_pred: np.ndarray = None, sev_true: np.ndarray = None) -> dict:
    """Same metric set/thresholds as evaluate.py's evaluate_split."""
    valid = ~np.isnan(scores)
    scores_v = scores[valid]
    det_v = det_true[valid]

    auroc = roc_auc_score(det_v, scores_v)

    # AUROC (threshold-free) is the headline number for these baselines.
    # Classical correlation/GCC-PHAT scores have no natural 0.5 decision
    # boundary the way a trained sigmoid does, so F1/accuracy below use a
    # median-split threshold on the scores themselves (not on the labels —
    # no test-label leakage, but note this in the writeup as a caveat
    # specific to the classical methods).
    y_pred = (scores_v >= np.median(scores_v)).astype(int)
    f1 = f1_score(det_v, y_pred)
    acc = accuracy_score(det_v, y_pred)
    cm = confusion_matrix(det_v, y_pred)

    leak_mask = det_true == 1
    pos_mask = leak_mask & (pos_valid == 1)
    pos_mae_norm = (mean_absolute_error(pos_true[pos_mask], pos_pred[pos_mask])
                    if pos_mask.sum() > 0 else float("nan"))
    pos_mae_m = pos_mae_norm * MEAN_SENSOR_SEPARATION_M

    sev_r2 = float("nan")
    if sev_pred is not None and leak_mask.sum() > 1:
        sev_r2 = r2_score(sev_true[leak_mask], sev_pred[leak_mask])

    net_name = NETWORK_NAMES.get(split, split)
    return {
        "network": net_name,
        "auroc": round(float(auroc), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "pos_mae_norm": round(float(pos_mae_norm), 4) if not np.isnan(pos_mae_norm) else None,
        "pos_mae_metres": round(float(pos_mae_m), 2) if not np.isnan(pos_mae_norm) else None,
        "sev_r2": round(float(sev_r2), 4) if not np.isnan(sev_r2) else None,
        "n_total": int(len(det_v)),
        "n_leak": int(det_v.sum()),
        "confusion_matrix": cm.tolist(),
    }


def print_summary(method_name: str, all_results: list):
    print(f"\n{'='*55}")
    print(f"  {method_name} — SUMMARY (Cross-Network Generalisation)")
    print(f"{'='*55}")
    print(f"  {'Network':<25} {'AUROC':>8} {'F1':>8} {'Acc':>8} {'MAE(m)':>8}")
    print(f"  {'-'*57}")
    for r in all_results:
        pos = f"{r['pos_mae_metres']:.2f}m" if r["pos_mae_metres"] else "  N/A"
        print(f"  {r['network']:<25} {r['auroc']:>8.4f} {r['f1']:>8.4f} "
              f"{r['accuracy']:>8.4f} {pos:>8}")
    aurocs = [r["auroc"] for r in all_results]
    print(f"\n  Mean AUROC: {np.mean(aurocs):.4f}")
    print(f"  Min  AUROC: {min(aurocs):.4f}")


def save_results(method_slug: str, all_results: list):
    out = [dict(r, cache=CACHE_ROOT.name) for r in all_results]
    suffix = "" if CACHE_ROOT.name == "cache_c" else f"_{CACHE_ROOT.name}"
    out_path = RESULTS_DIR / f"test_results_{method_slug}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")