"""
pregenerate_d.py — Pre-generate Model D signals to disk
========================================================
Identical to pregenerate_c.py but:
  - Imports dataset_d.py (Model D signal synthesis)
  - Writes to ../cache_d/ (does not overwrite Model C cache)

Usage:
    python pregenerate_d.py --split all
    python pregenerate_d.py --split train
    python pregenerate_d.py --split val

IMPORTANT: Delete ../cache_d/ before running if regenerating.
    Windows: rmdir /s /q ..\cache_d
    Linux:   rm -rf ../cache_d
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from dataset_d import LeakDataset

CACHE_ROOT = Path("../cache_d")

SPLIT_CSV = {
    "train":          "../csv/train_sampled.csv",
    "val":            "../csv/val_sampled.csv",
    "test_network_3": "../csv/test_network_3.csv",
    "test_network_6": "../csv/test_network_6.csv",
    "test_network_8": "../csv/test_network_8.csv",
}


def verify_cache(signals: np.ndarray, labels: np.ndarray, split_name: str):
    print(f"\n  [verify] Running amplitude preservation check for {split_name}...")
    leak_mask = labels[:, 0] == 1
    n_leak    = leak_mask.sum()

    if n_leak < 100:
        print(f"  [verify] Too few leak samples ({n_leak}) — skipping.")
        return

    sample_size  = min(2000, n_leak)
    leak_indices = np.where(leak_mask)[0][:sample_size]
    leak_signals = signals[leak_indices]
    leak_flows   = labels[leak_indices, 2]

    noleak_mask    = labels[:, 0] == 0
    noleak_indices = np.where(noleak_mask)[0][:sample_size]
    noleak_signals = signals[noleak_indices]

    rms_leak   = np.sqrt(np.mean(leak_signals   ** 2, axis=(1, 2)))
    rms_noleak = np.sqrt(np.mean(noleak_signals ** 2, axis=(1, 2)))
    rms_std    = rms_leak.std()
    rms_mean   = rms_leak.mean()
    separation = rms_leak.mean() / max(rms_noleak.mean(), 1e-8)
    corr       = np.corrcoef(rms_leak, leak_flows[:len(rms_leak)])[0, 1] \
                 if leak_flows.std() > 1e-6 else 0.0

    print(f"  [verify] Leak RMS mean:     {rms_mean:.4f}")
    print(f"  [verify] Leak RMS std:      {rms_std:.4f}  (want > 0.05)")
    print(f"  [verify] NoLeak RMS mean:   {rms_noleak.mean():.4f}")
    print(f"  [verify] Leak/NoLeak sep:   {separation:.3f}x  (want 1.5-4x)")
    print(f"  [verify] RMS-flow corr:     {corr:.4f}   (want > 0.10)")
    print(f"  [verify] Dual-sensor leaks: {int((labels[:, 3] == 1).sum()):,}")

    if rms_std < 0.01:
        print("  [ERROR] RMS std near zero — amplitude not preserved!")
        raise RuntimeError("Cache verification failed.")
    elif corr < 0.05:
        print("  [WARN]  Low RMS-flow correlation — severity learning may be weak.")
    else:
        print("  [OK]    Cache is valid for training.")


def pregenerate_split(split_name: str):
    csv_path     = SPLIT_CSV[split_name]
    out_dir      = CACHE_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    signals_path = out_dir / "signals.npy"
    labels_path  = out_dir / "labels.npy"

    if signals_path.exists() and labels_path.exists():
        print(f"Cache exists for {split_name} — skipping. "
              f"Delete {out_dir} to regenerate.")
        return

    print(f"\n{'='*55}\n  Pregenerating Model D: {split_name}\n{'='*55}")

    ds = LeakDataset(csv_path, signal_length=2000,
                     sampling_frequency=5000, augment=False)
    N  = len(ds)
    print(f"  {N:,} samples | {N * 2 * 2000 * 4 / 1e9:.2f} GB signals")

    signals = np.lib.format.open_memmap(
        str(signals_path), mode="w+",
        dtype=np.float32, shape=(N, 2, 2000))
    labels = np.lib.format.open_memmap(
        str(labels_path), mode="w+",
        dtype=np.float32, shape=(N, 4))

    CHUNK = 500
    with tqdm(total=N, unit="samples", dynamic_ncols=True) as pbar:
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            for i in range(start, end):
                r          = ds[i]
                signals[i] = r[0].numpy()
                labels[i]  = [r[2].item(), r[3].item(),
                               r[4].item(), r[5].item()]
            signals.flush()
            labels.flush()
            pbar.update(end - start)
            time.sleep(0.01)

    leak_pct = (labels[:, 0] == 1).mean() * 100
    print(f"  Done | Leak: {leak_pct:.1f}% | Saved: {out_dir}")
    verify_cache(signals, labels, split_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                        choices=list(SPLIT_CSV.keys()) + ["all"])
    args = parser.parse_args()
    CACHE_ROOT.mkdir(exist_ok=True)
    splits = list(SPLIT_CSV.keys()) if args.split == "all" else [args.split]
    for s in splits:
        pregenerate_split(s)
    print("\nModel D pregeneration complete. Run: python train_d.py")


if __name__ == "__main__":
    main()