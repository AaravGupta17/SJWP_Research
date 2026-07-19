import numpy as np

for name, path in [("L-TOWN", "../cache_c/test_network_3"),
                   ("KY15", "../cache_c/test_network_6"),
                   ("Richmond", "../cache_c/test_network_8")]:
    labels = np.load(f"{path}/labels.npy", mmap_mode="r")
    leak_mask = labels[:, 0] == 1
    pos_valid = labels[:, 3] == 1
    dual_leak = leak_mask & pos_valid

    print(f"\n{name}:")
    print(f"  Total samples:        {len(labels):,}")
    print(f"  Leak samples:         {leak_mask.sum():,}")
    print(f"  Dual-sensor leaks:    {dual_leak.sum():,} ({dual_leak.sum() / leak_mask.sum() * 100:.1f}% of leaks)")
    print(f"  Single-sensor leaks:  {(leak_mask & ~pos_valid).sum():,}")