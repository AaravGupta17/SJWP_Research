import pandas as pd
import numpy as np

df = pd.read_csv("../csv/test_index.csv")
df = df.dropna(subset=["demand_multiplier"])

for network in ["Network_3", "Network_6", "Network_8"]:
    net_df = df[df["network_id"] == network].copy()
    leak_df = net_df[net_df["file_type"] == "leak"]
    base_df = net_df[net_df["file_type"] == "base"]

    print(f"{network}: {len(leak_df)} leak, {len(base_df)} base")

    n = len(base_df)
    if n == 0:
        print(f"  WARNING: No base rows for {network}, using 5000 leak samples only")
        combined = leak_df.sample(n=min(5000, len(leak_df)), random_state=42)
    else:
        leak_sampled = leak_df.sample(n=min(n, len(leak_df)), random_state=42)
        combined = pd.concat([leak_sampled, base_df]).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

    out_path = f"../csv/test_{network}.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Saved {len(combined)} samples → {out_path}")