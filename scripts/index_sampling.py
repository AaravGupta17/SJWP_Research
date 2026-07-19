"""
sample.py — Resample train/val/test indices by network
=======================================================
Train  : Networks 1, 2, 4, 5  (balanced 50/50 leak/base)
Val    : Network 7             (balanced 50/50 leak/base)
Test   : Networks 3, 6, 8     (separate files, natural balance)
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_NETWORKS = ["Network_1", "Network_2", "Network_4", "Network_5"]
VAL_NETWORKS   = ["Network_7"]
TEST_NETWORKS  = ["Network_3", "Network_6", "Network_8"]

TRAIN_INDEX = "../csv/train_index.csv"
TEST_INDEX  = "../csv/test_index.csv"
OUT_DIR     = Path("../csv")

SAMPLES_PER_GROUP = 15000   # per network+material+demand+file_type group
SEED = 42
np.random.seed(SEED)


def balanced_sample(df: pd.DataFrame, n_per_type: int) -> pd.DataFrame:
    """Sample equal leak and base rows."""
    leak = df[df["file_type"] == "leak"]
    base = df[df["file_type"] == "base"]
    n    = min(n_per_type, len(leak), len(base))
    return pd.concat([
        leak.sample(n=n, random_state=SEED),
        base.sample(n=n, random_state=SEED)
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def sample_by_networks(index_csv: str, networks: list,
                       balanced: bool = True) -> pd.DataFrame:
    print(f"Loading {index_csv}...")
    df = pd.read_csv(index_csv)
    df = df[df["network_id"].isin(networks)].copy()
    print(f"  Rows matching {networks}: {len(df):,}")

    if not balanced:
        return df.reset_index(drop=True)

    # Stratified balanced sample per network+material+demand
    groups  = df.groupby(["network_id", "material_id", "demand_multiplier"])
    sampled = []
    for name, group in groups:
        sampled.append(balanced_sample(group, SAMPLES_PER_GROUP))

    result = pd.concat(sampled, ignore_index=True)
    result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return result


# ── Train ──────────────────────────────────────────────────────────────────────
print("=" * 50)
print("TRAIN (Networks 1, 2, 4, 5)")
train_df = sample_by_networks(TRAIN_INDEX, TRAIN_NETWORKS, balanced=True)
train_df.to_csv(OUT_DIR / "train_sampled.csv", index=False)
print(f"  Total: {len(train_df):,}")
print(f"  Leak ratio: {train_df['file_type'].eq('leak').mean():.3f}")
print(f"  Per network:\n{train_df['network_id'].value_counts().to_string()}")

# ── Val ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("VAL (Network 7)")
val_df = sample_by_networks(TRAIN_INDEX, VAL_NETWORKS, balanced=True)
val_df.to_csv(OUT_DIR / "val_sampled.csv", index=False)
print(f"  Total: {len(val_df):,}")
print(f"  Leak ratio: {val_df['file_type'].eq('leak').mean():.3f}")

# ── Test per network ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST (Networks 3, 6, 8 — separate files)")
for net in TEST_NETWORKS:
    df = sample_by_networks(TEST_INDEX, [net], balanced=True)
    out = OUT_DIR / f"test_{net.lower()}.csv"
    df.to_csv(out, index=False)
    print(f"  {net}: {len(df):,} samples → {out}")

print("\nDone. Files saved to ../csv/")
print("  train_sampled.csv")
print("  val_sampled.csv")
print("  test_network_3.csv")
print("  test_network_6.csv")
print("  test_network_8.csv")