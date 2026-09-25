"""
bank_f.py — Model F: collect the real recordings used for training
====================================================================
Writes ../cache_f/bank.npz with

  single   (N, 2000) float16, unit RMS   one-sensor windows from the public
                                         datasets: Hong Kong (noise loggers,
                                         hydrophones) and Dongguan, BOTH classes.
                                         No-leak windows are backgrounds [F1];
                                         all windows are real labelled rows.
  s_y, s_group, s_source, s_val          label, recording/site, source, and
                                         whether the group is held out for
                                         validation (20% of groups per source)
  pairs    (M, 2, 2000) float16          two-sensor Mendeley BRANCHED no-leak
                                         hydrophone windows (the recordings
                                         Models C-E already used as noise).
  p_group, p_val

Mendeley LOOPED is never read: it is the clean test set. Mendeley labels
are not used at all, so a Mendeley result stays a new-network test.
All windows: 5 kHz, band-limited to 2 kHz (experiments/public_data.py).

    python Model_F/bank_f.py                  # from the repo root
"""

import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import mendeley as M                                      # noqa: E402
import public_data as P                                   # noqa: E402

SOURCE = {"hk_noiselogger": "hongkong", "hk_hydrophone": "hongkong", "dongguan": "dongguan"}
VAL_FRACTION = 0.2


def is_val_group(group: str) -> bool:
    """Deterministic 20% of groups, independent of file order."""
    return zlib.crc32(group.encode()) % 100 < VAL_FRACTION * 100


def usable(x: np.ndarray) -> np.ndarray:
    """Drop silent / dead windows (RMS far below the file's median)."""
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=tuple(range(1, x.ndim))))
    return rms > 1e-3 * (np.median(rms) + 1e-30)


def mendeley_branched_pairs(root: Path = None):
    root = root or P.DATASETS / "Hydrophone" / "Hydrophone"
    xs, gs = [], []
    for rec in M.discover(root, "hydrophone"):
        if not rec.in_noise_bank:                  # Branched no-leak only
            continue
        x1, x2 = M.load_recording(rec, fs_out=P.FS)
        w = M.to_windows(P.to_common(x1, P.FS), P.to_common(x2, P.FS), P.WIN)
        xs.append(w)
        gs += [f"md:{rec.rec_id}"] * len(w)
    if not xs:
        return np.zeros((0, 2, P.WIN), np.float32), np.array([])
    return np.concatenate(xs), np.array(gs)


def main():
    out = ROOT / "cache_f" / "bank.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Public datasets (5 kHz, band-limited to 2 kHz):")
    W = P.load(list(SOURCE))
    keep = usable(W.x)
    W = W.subset(keep)
    x = W.x.astype(np.float64)
    x = x / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))
    source = np.array([SOURCE[d] for d in W.dataset])
    s_val = np.array([is_val_group(g) for g in W.group])

    print("Mendeley Branched no-leak hydrophone pairs:")
    try:
        pairs, p_group = mendeley_branched_pairs()
    except FileNotFoundError as e:
        print(f"  skipped: {e}")
        pairs, p_group = np.zeros((0, 2, P.WIN), np.float32), np.array([])
    if len(pairs):
        k = usable(pairs)
        pairs, p_group = pairs[k].astype(np.float64), p_group[k]
        pairs = pairs / np.sqrt(np.mean(pairs ** 2, axis=(1, 2), keepdims=True))
    p_val = np.array([is_val_group(g) for g in p_group], bool)

    np.savez(out, single=x.astype(np.float16), s_y=W.y.astype(np.int8), s_group=W.group,
             s_source=source, s_dataset=W.dataset, s_val=s_val,
             pairs=pairs.astype(np.float16), p_group=p_group, p_val=p_val)
    for s in sorted(set(source)):
        m = source == s
        print(f"  {s:9s} {m.sum():6d} windows | {len(set(W.group[m])):4d} groups | "
              f"leak {W.y[m].mean():.0%} | val groups {len(set(W.group[m & s_val]))}")
    print(f"  mendeley  {len(pairs):6d} two-sensor no-leak windows | "
          f"{len(set(p_group))} recordings")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
