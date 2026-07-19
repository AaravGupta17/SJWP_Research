import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataset import LeakDataset

SIGNAL_LENGTH = 2000
N_CHANNELS = 2

def pregenerate_test(network, index_csv, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    signals_path = cache_dir / "signals.npy"
    labels_path  = cache_dir / "labels.npy"
    idx_path     = cache_dir / "split_idx.npz"

    print(f"\nPregenerating {network}...")
    ds = LeakDataset(index_csv, signal_length=SIGNAL_LENGTH, augment=False)
    N = len(ds)
    print(f"Samples: {N:,}")

    signals = np.lib.format.open_memmap(
        str(signals_path), mode="w+",
        dtype=np.float32, shape=(N, N_CHANNELS, SIGNAL_LENGTH)
    )
    labels = np.lib.format.open_memmap(
        str(labels_path), mode="w+",
        dtype=np.float32, shape=(N, 4)
    )

    CHUNK = 1000
    with tqdm(total=N, unit="samples") as pbar:
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            for i in range(start, end):
                sig, scalars, label, pos, sev, pos_valid = ds[i]
                signals[i] = sig.numpy()
                labels[i]  = [label.item(), pos.item(), sev.item(), pos_valid.item()]
            signals.flush()
            labels.flush()
            pbar.update(end - start)

    # Save all indices (no split needed for test)
    idx = np.arange(N)
    np.savez(str(idx_path), train=idx, val=idx)
    print(f"Done — {network}: {N:,} samples cached at {cache_dir}")


if __name__ == "__main__":
    networks = {
        "Network_3": ("../csv/test_Network_3.csv", "../cache/test_Network_3"),
        "Network_6": ("../csv/test_Network_6.csv", "../cache/test_Network_6"),
        "Network_8": ("../csv/test_Network_8.csv", "../cache/test_Network_8"),
    }

    for network, (index_csv, cache_dir) in networks.items():
        pregenerate_test(network, index_csv, cache_dir)

    print("\nAll test networks pregenerated!")