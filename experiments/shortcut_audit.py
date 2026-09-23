"""
shortcut_audit.py — E2: can trivial features solve the synthetic test sets?
=============================================================================
AcousticLeakNet reaches AUROC 1.000 on the three held-out synthetic
networks. This script asks whether that number says anything about pipe
acoustics, or whether the synthetic leak/no-leak classes are separable by
something trivial. For each test network it reports the AUROC of:

  rms        overall loudness (one number per window)
  dc         mean absolute DC offset (dataset_c.py adds p * 3e-4 to each
             channel, and leak/no-leak rows take pressure from different
             CSV columns)
  max_abs    peak amplitude
  clip_frac  fraction of samples at the +-10 clip limit
  gcc_peak   GCC-PHAT coherence peak (classical TDOA detector)
  model      AcousticLeakNet as evaluated
  model_noDC AcousticLeakNet with each channel's mean subtracted first

Reading the output:
  - trivial feature AUROC ~1.0  -> the benchmark cannot distinguish a
    physics-aware detector from a loudness meter; report it that way.
  - model_noDC << model         -> the model relies on the DC shortcut.

Needs the pre-generated cache (model_C/pregen_c.py), default ../cache_c.

    python experiments/shortcut_audit.py
    python experiments/shortcut_audit.py --cache cache_d --ckpt best_model_d.pt
"""

import argparse

import numpy as np

import features as F
from _common import REPO_ROOT, load_model, predict_proba, record_run
from metrics import point_metrics

SPLITS = {"test_network_3": "L-TOWN (Network 3)",
          "test_network_6": "KY15 (Network 6)",
          "test_network_8": "Richmond (Network 8)"}


def stratified_sample(labels: np.ndarray, n_max: int, rng) -> np.ndarray:
    if len(labels) <= n_max:
        return np.arange(len(labels))
    idx = []
    for c in (0, 1):
        ci = np.flatnonzero(labels == c)
        k = min(len(ci), n_max // 2)
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def auroc(y, s):
    return point_metrics(y, s)["auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_c")
    ap.add_argument("--ckpt", default="best_model_c_v4.pt")
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="per split, class-balanced subsample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model, _ = load_model(args.ckpt)
    results = {}
    for split, name in SPLITS.items():
        d = REPO_ROOT / args.cache / split
        if not (d / "signals.npy").exists():
            print(f"skip {split}: no cache at {d}")
            continue
        sig = np.load(d / "signals.npy", mmap_mode="r")
        lab = np.load(d / "labels.npy", mmap_mode="r")
        if len(sig) == 0:
            print(f"skip {split}: cache is empty")
            continue
        y_all = lab[:, 0].astype(int)
        idx = (np.arange(len(y_all)) if args.max_samples == 0
               else stratified_sample(y_all, args.max_samples, rng))
        x = np.asarray(sig[idx], dtype=np.float32)
        y = y_all[idx]

        gcc, _ = F.gcc_peak(x)
        p = predict_proba(model, x, logits=True)
        p_nodc = predict_proba(model, x - x.mean(axis=2, keepdims=True), logits=True)
        r = {
            "n": int(len(y)), "n_leak": int(y.sum()),
            "auroc": {
                "rms": auroc(y, F.rms(x)),
                "dc": auroc(y, F.dc_offset(x)),
                "max_abs": auroc(y, np.abs(x).max(axis=(1, 2))),
                "clip_frac": auroc(y, (np.abs(x) >= 9.999).mean(axis=(1, 2))),
                "gcc_peak": auroc(y, gcc),
                "model": auroc(y, p),
                "model_noDC": auroc(y, p_nodc),
            },
            "mean_dc": {"leak": float(F.dc_offset(x[y == 1]).mean()),
                        "no_leak": float(F.dc_offset(x[y == 0]).mean())},
        }
        results[name] = r
        print(f"\n{name}  (n={r['n']}, leak={r['n_leak']})")
        for k, v in r["auroc"].items():
            print(f"  {k:11s} AUROC = {v:.4f}" if v is not None else f"  {k:11s} n/a")
        print(f"  mean |DC|: leak {r['mean_dc']['leak']:.4f}  no-leak {r['mean_dc']['no_leak']:.4f}")

    if results:
        summary = "; ".join(f"{n.split(' ')[0]}: rms {r['auroc']['rms']:.3f}, "
                            f"model {r['auroc']['model']:.3f}, noDC {r['auroc']['model_noDC']:.3f}"
                            for n, r in results.items())
        record_run("e2_shortcut_audit", vars(args), results, summary)


if __name__ == "__main__":
    main()
