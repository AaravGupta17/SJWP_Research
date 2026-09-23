"""
pregenerate.py — Pre-generate all signals to disk
==================================================
Generates train, val, and test signals using IDENTICAL code.
This guarantees zero train/test signal mismatch.

Usage:
    python pregenerate.py --split all         # run everything
    python pregenerate.py --split train       # just train
    python pregenerate.py --split val
    python pregenerate.py --split test_network_3
    python pregenerate.py --split test_network_6
    python pregenerate.py --split test_network_8
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasetnew import LeakDataset

CACHE_ROOT = Path("../cache_v2")

SPLIT_CSV = {
    "train":          "../csv/train_sampled.csv",
    "val":            "../csv/val_sampled.csv",
    "test_network_3": "../csv/test_network_3.csv",
    "test_network_6": "../csv/test_network_6.csv",
    "test_network_8": "../csv/test_network_8.csv",
}


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

    print(f"\n{'='*55}\n  Pregenerating: {split_name}\n{'='*55}")

    ds = LeakDataset(csv_path, signal_length=2000,
                     sampling_frequency=5000, augment=False)
    N  = len(ds)
    print(f"  {N:,} samples | "
          f"{N * 2 * 2000 * 4 / 1e9:.2f} GB signals")

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

    leak_pct = (labels[:, 0] == 1).mean() * 100
    print(f"  ✓ Done | Leak: {leak_pct:.1f}% | Saved: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                        choices=list(SPLIT_CSV.keys()) + ["all"])
    args = parser.parse_args()
    CACHE_ROOT.mkdir(exist_ok=True)

    splits = list(SPLIT_CSV.keys()) if args.split == "all" else [args.split]
    for s in splits:
        pregenerate_split(s)
    print("\n✓ All pregenerations complete. Run: python train.py")


if __name__ == "__main__":
    main()