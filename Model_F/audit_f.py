"""
audit_f.py — shortcut audit of the Model F training data (the E2 check)
==========================================================================
E2 found that on Model C's synthetic data PEAK AMPLITUDE alone reaches
AUROC 0.92-0.97, so a network need not learn what a leak sounds like.
This runs the same kind of check on Model F's training mixtures: if any
trivial feature separates leak from no-leak, Model F can take a shortcut.

  trivial features   RMS, peak, channel RMS ratio, channel correlation,
                     kurtosis (each alone, AUROC folded to >= 0.5)
  reference          5-fold logistic regression on E9's spectral features

    python Model_F/audit_f.py            # synthetic rows, then real rows
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import train_f as F                                       # noqa: E402
from _common import record_run                            # noqa: E402
from cross_dataset import features_1ch                    # noqa: E402


def trivial_features(x: np.ndarray) -> dict:
    c = x - x.mean(-1, keepdims=True)
    return {
        "rms": x.std(axis=(1, 2)),
        "peak": np.abs(x).max(axis=(1, 2)),
        "channel_rms_ratio": np.abs(np.log(x[:, 0].std(1) / (x[:, 1].std(1) + 1e-12))),
        "channel_correlation": np.array([np.corrcoef(a, b)[0, 1] for a, b in x]),
        "kurtosis": (c ** 4).mean(-1).mean(-1) / (c.var(-1).mean(-1) ** 2 + 1e-12),
    }


def audit(x: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for k, v in trivial_features(x).items():
        a = roc_auc_score(y, np.nan_to_num(v))
        out[k] = max(a, 1 - a)
    s = cross_val_predict(LogisticRegression(max_iter=3000), features_1ch(x[:, 0].astype(np.float32)),
                          y, cv=5, method="decision_function")
    out["spectral_logreg_5fold"] = roc_auc_score(y, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_f")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cache = F.ROOT / args.cache
    bank = F.Bank(cache / "bank.npz", val=False)
    results = {}
    for name, real_frac in (("synthetic_rows", 0.0), ("real_rows", 1.0)):
        ds = F.MixDataset(cache, "train", bank, real_frac, args.seed, fixed=True)
        items = [ds[i] for i in range(args.n)]
        x = np.stack([t[0].numpy() for t in items]).astype(np.float64)
        y = np.array([t[1][0].item() for t in items])
        results[name] = audit(x, y)
        print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in results[name].items()))
    s = results["synthetic_rows"]
    worst = max((v, k) for k, v in s.items() if k != "spectral_logreg_5fold")
    record_run("f_shortcut_audit", vars(args), results,
               f"synthetic rows: best trivial feature {worst[1]} {worst[0]:.3f}, "
               f"spectral logreg {s['spectral_logreg_5fold']:.3f}")


if __name__ == "__main__":
    main()
