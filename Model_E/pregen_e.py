"""
pregen_e.py — Pre-generate Model E signals to ../cache_e
=========================================================
Same as model_C/pregen_c.py but uses dataset_e.LeakDatasetE, and seeds
the random generator per split so the cache is reproducible (pregen_c
did not seed).

Usage (from Model_E/):
    python pregen_e.py --split all
    python pregen_e.py --split train
    python pregen_e.py --split all --no interferers sensor    # realism ablation

Delete ../cache_e before regenerating.
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset_e import LeakDatasetE, REALISM_DEFAULTS

SPLIT_CSV = {
    "train":          "../data/csv/train_sampled.csv",
    "val":            "../data/csv/val_sampled.csv",
    "test_network_3": "../data/csv/test_Network_3.csv",
    "test_network_6": "../data/csv/test_Network_6.csv",
    "test_network_8": "../data/csv/test_Network_8.csv",
}


def pregenerate_split(split: str, cache_root: Path, realism: dict, seed: int):
    out_dir = cache_root / split
    sig_path, lab_path = out_dir / "signals.npy", out_dir / "labels.npy"
    if sig_path.exists() and lab_path.exists():
        print(f"Cache exists for {split} — skipping. Delete {out_dir} to regenerate.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed + zlib.crc32(split.encode()) % 10_000)

    ds = LeakDatasetE(SPLIT_CSV[split], realism=realism, augment=False)
    N = len(ds)
    signals = np.lib.format.open_memmap(str(sig_path), mode="w+", dtype=np.float32, shape=(N, 2, 2000))
    labels = np.lib.format.open_memmap(str(lab_path), mode="w+", dtype=np.float32, shape=(N, 4))
    for i in tqdm(range(N), desc=split, unit="samples", dynamic_ncols=True):
        r = ds[i]
        signals[i] = r[0].numpy()
        labels[i] = [r[2].item(), r[3].item(), r[4].item(), r[5].item()]
    signals.flush()
    labels.flush()
    bad = int((labels[:, 0] < 0).sum())
    print(f"  {split}: {N:,} samples | leak {(labels[:, 0] == 1).mean():.1%} | failed rows {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=list(SPLIT_CSV) + ["all"])
    ap.add_argument("--cache", default="../cache_e")
    ap.add_argument("--no", nargs="*", default=[], choices=list(REALISM_DEFAULTS),
                    help="realism switches to turn OFF (for ablations)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    realism = {k: (k not in args.no) for k in REALISM_DEFAULTS}
    cache_root = Path(args.cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "realism.json").write_text(json.dumps({"realism": realism, "seed": args.seed}, indent=2))
    for s in (list(SPLIT_CSV) if args.split == "all" else [args.split]):
        pregenerate_split(s, cache_root, realism, args.seed)
    print("\nDone. Train with: cd ../model_C && python train_c.py --cache cache_e --prefix e")


if __name__ == "__main__":
    main()
