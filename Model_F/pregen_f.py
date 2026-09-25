"""
pregen_f.py — Model F: pre-generate the LEAK component only, to ../cache_f
===========================================================================
Model E's leak physics (fractional delays [E1], SNR range [E6], measured
plastic attenuation [E9], rows without a leak source dropped [E8]) with NO
background, interferers, sensor response or gain: those are drawn fresh for
every training sample in train_f.py, so each leak is heard over many
different backgrounds instead of one.

The leak signal is stored in units of the background RMS: Model E scales a
leak to noise_rms x 10^(SNR/20), so dividing by noise_rms keeps its SNR when
it is later added to a unit-RMS background. It is band-limited to 2 kHz [F3].

Per split, in ../cache_f/<split>/:
  labels.npy  (N, 4) float32   leak, position, flow (severity), position valid
  leak_row.npy (N,) int32      row in leak.npy, -1 for no-leak rows
  leak.npy    (N_leak, 2, 2000) float16

Usage (from the repo root):
    python Model_F/pregen_f.py --split train val
    python Model_F/pregen_f.py --split train --max-rows 400000
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model_E"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_f import band_limit                          # noqa: E402
from dataset_e import LeakDatasetE                        # noqa: E402

SPLIT_CSV = {
    "train": "data/csv/train_sampled.csv",
    "val": "data/csv/val_sampled.csv",
}


def leak_component(ds: LeakDatasetE, c: dict) -> np.ndarray:
    """(2, T) leak signal in background-RMS units, band-limited."""
    out = np.zeros((2, ds.signal_length))
    ds._leak(out, c)
    return band_limit(out / ds.noise_bank.noise_rms)


def pregenerate_split(split: str, cache_root: Path, seed: int, max_rows: int = None):
    out_dir = cache_root / split
    if (out_dir / "labels.npy").exists():
        print(f"Cache exists for {split}: skipping. Delete {out_dir} to regenerate.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed + zlib.crc32(split.encode()) % 10_000)

    cwd = os.getcwd()
    os.chdir(ROOT / "model_C")                 # index CSVs use paths relative to model_C/
    try:
        ds = LeakDatasetE(str(ROOT / SPLIT_CSV[split]), augment=False)
    finally:
        os.chdir(cwd)
    idx = np.array(ds._valid_idx)
    if max_rows and len(idx) > max_rows:
        idx = np.sort(np.random.choice(idx, max_rows, replace=False))
    cfgs = [ds._cache[i] for i in idx]
    is_leak = np.array([c["leak_status"] == 1 for c in cfgs])

    labels = np.array([[c["leak_status"], c["leak_pos"], c["leak_flow"], c["pos_valid"]]
                       for c in cfgs], np.float32)
    leak_row = np.full(len(cfgs), -1, np.int32)
    leak_row[is_leak] = np.arange(is_leak.sum())
    leak = np.lib.format.open_memmap(str(out_dir / "leak.npy"), mode="w+", dtype=np.float16,
                                     shape=(int(is_leak.sum()), 2, ds.signal_length))
    for r in tqdm(np.flatnonzero(is_leak), desc=split, unit="leaks", dynamic_ncols=True):
        leak[leak_row[r]] = leak_component(ds, cfgs[r])
    leak.flush()
    np.save(out_dir / "leak_row.npy", leak_row)
    np.save(out_dir / "labels.npy", labels)       # written last: marks the split complete
    print(f"  {split}: {len(cfgs):,} rows | leak {is_leak.mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "val"], choices=list(SPLIT_CSV))
    ap.add_argument("--cache", default="cache_f")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None, help="random subset of rows per split")
    args = ap.parse_args()
    for s in args.split:
        pregenerate_split(s, ROOT / args.cache, args.seed, args.max_rows)
    print("\nNext: python Model_F/train_f.py")


if __name__ == "__main__":
    main()
