"""
realism_check.py — E8: how distinguishable are synthetic windows from real ones?
==================================================================================
A number for "more realistic": a logistic-regression discriminator is
trained to tell REAL windows from SYNTHETIC windows, using only features
that do not depend on overall loudness (so calibration/scaling can't
explain the result):

  - band-energy SHAPE (log band energy minus log total energy)
  - kurtosis and crest factor (impulsiveness)
  - inter-channel correlation at zero lag, and peak normalised
    cross-correlation within the physical lag window

Out-of-fold AUROC with 5-fold grouped CV (real windows grouped by
recording): 0.5 = indistinguishable, 1.0 = trivially separable. Reported
separately for no-leak and leak windows, for Model C and Model E.

Real data: Branched recordings ONLY. Looped is the clean test set and is
never used to tune or judge the synthesiser.

Needs the EPANET CSVs + hydrophone noise bank (like snr_sweep.py) and the
Mendeley accelerometer data.

    python experiments/realism_check.py
    python experiments/realism_check.py --n 2000 --csv val_sampled.csv
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import mendeley as M
from _common import DATASETS, PLOTS_DIR, REPO_ROOT, record_run
from metrics import point_metrics


def shape_features(w: np.ndarray) -> np.ndarray:
    """Loudness-independent features, (N, 2, T) -> (N, k)."""
    w = w.astype(np.float64)
    be = F.band_energy(w)
    total = np.log10(np.sum(10 ** be, axis=1, keepdims=True) + 1e-30)
    shape = be - total
    k = kurtosis(w, axis=-1).mean(axis=1, keepdims=True)
    crest = (np.abs(w).max(axis=-1) / (np.sqrt((w ** 2).mean(axis=-1)) + 1e-12)).mean(axis=1, keepdims=True)
    a = w[:, 0] - w[:, 0].mean(axis=-1, keepdims=True)
    b = w[:, 1] - w[:, 1].mean(axis=-1, keepdims=True)
    na = np.sqrt((a ** 2).sum(-1)) * np.sqrt((b ** 2).sum(-1)) + 1e-12
    zero_lag = ((a * b).sum(-1) / na)[:, None]
    L = F.MAX_LAG_SAMPLES
    n = 1 << (2 * w.shape[-1] - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    xc = np.concatenate([xc[:, -L:], xc[:, :L + 1]], axis=1)
    peak = (np.abs(xc).max(axis=1) / na)[:, None]
    return np.hstack([shape, k, crest, zero_lag, peak])


def synth_windows(dataset_cls, csv_path: Path, n: int, seed: int, work: Path, **kw):
    df = pd.read_csv(csv_path)
    sub = pd.concat([g.sample(n=min(len(g), n // 2), random_state=seed)
                     for _, g in df.groupby(df["file_type"] == "leak")])
    tmp = work / f"realism_{csv_path.stem}_{seed}.csv"
    sub.to_csv(tmp, index=False)
    ds = dataset_cls(str(tmp), augment=False, **kw)
    np.random.seed(seed)
    xs, ys = [], []
    for i in range(len(ds)):
        s, _, d, *_ = ds[i]
        if d.item() >= 0:
            xs.append(s.numpy()); ys.append(int(d.item()))
    return np.stack(xs), np.array(ys)


def discriminability(real, synth, real_groups, seed) -> float:
    X = np.vstack([shape_features(real), shape_features(synth)])
    y = np.r_[np.ones(len(real), int), np.zeros(len(synth), int)]
    groups = np.r_[real_groups, [f"synth{i}" for i in range(len(synth))]]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return point_metrics(y, oof)["auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-root", type=Path, default=DATASETS / "Accelerometer" / "Accelerometer")
    ap.add_argument("--csv", default="val_sampled.csv", help="index CSV in data/csv to synthesise from")
    ap.add_argument("--n", type=int, default=1500, help="synthetic rows (class-balanced)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc = M.load_windowset(args.acc_root, "accelerometer", cache_dir=REPO_ROOT / "cache_mendeley",
                           verbose=False)
    real = acc.subset(acc.topology == "Branched")          # never Looped
    work = REPO_ROOT / "cache_sweep"
    work.mkdir(exist_ok=True)
    csv_path = REPO_ROOT / "data" / "csv" / args.csv

    os.chdir(REPO_ROOT / "model_C")                       # dataset paths are ../datasets/...
    sys.path.insert(0, str(REPO_ROOT / "Model_E"))
    from dataset_c import LeakDataset
    from dataset_e import LeakDatasetE
    synth = {"Model C": synth_windows(LeakDataset, csv_path, args.n, args.seed, work),
             "Model E": synth_windows(LeakDatasetE, csv_path, args.n, args.seed, work)}

    results = {}
    for name, (sx, sy) in synth.items():
        results[name] = {}
        for lab, cls in ((0, "no_leak"), (1, "leak")):
            rm = real.y == lab
            results[name][cls] = discriminability(real.x[rm], sx[sy == lab], real.group[rm], args.seed)
        print(f"{name}: real-vs-synthetic AUROC  no-leak {results[name]['no_leak']:.3f} | "
              f"leak {results[name]['leak']:.3f}   (0.5 = indistinguishable)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, (lab, cls) in enumerate(((0, "no-leak"), (1, "leak"))):
        sets = {"real (Branched)": real.x[real.y == lab]}
        sets.update({n: sx[sy == lab] for n, (sx, sy) in synth.items()})
        for n, w in sets.items():
            f, p = welch(w.astype(np.float64), fs=5000, nperseg=256, axis=-1)
            p = p.mean(axis=(0, 1))
            ax[j].semilogy(f, p / p.sum(), label=n, lw=2.5 if n.startswith("real") else 1.5)
        ax[j].set_title(f"{cls}: normalised mean spectrum")
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / "e8_realism_check.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    summary = "; ".join(f"{n}: no-leak {r['no_leak']:.3f}, leak {r['leak']:.3f}" for n, r in results.items())
    record_run("e8_realism_check", {**vars(args), "acc_root": str(args.acc_root)}, results, summary)


if __name__ == "__main__":
    main()
