import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

DATA_ROOT = '../test'

TRAIN_NETWORKS = [
    "Network 1",
    "Network 2",
    "Network 3",
    "Network 4"
]

SAMPLES_PER_NETWORK = 300000  # Total per network
DEMAND_GROUPS = [0.7, 1.0, 1.2]

all_samples = []

def extract_demand_from_filename(filename):
    match = re.search(r"(0\.7|1\.0|1\.2)", filename)
    if match:
        return float(match.group(1))
    return None

for network in TRAIN_NETWORKS:
    network_path = os.path.join(DATA_ROOT, network)
    print(f"\nProcessing {network}")

    # Store samples grouped by demand
    demand_buckets = {d: [] for d in DEMAND_GROUPS}

    for root, dirs, files in os.walk(network_path):
        for file in files:
            if file.lower().endswith(".csv"):
                demand = extract_demand_from_filename(file)

                if demand not in DEMAND_GROUPS:
                    continue

                file_path = os.path.join(root, file)

                try:
                    num_rows = sum(1 for _ in open(file_path)) - 1
                except:
                    continue

                for row_idx in range(num_rows):
                    demand_buckets[demand].append({
                        "file_path": file_path,
                        "row_idx": row_idx,
                        "network": network,
                        "demand_multiplier": demand
                    })

    # Now sample equally from each demand group
    samples_per_demand = SAMPLES_PER_NETWORK // len(DEMAND_GROUPS)

    for demand in DEMAND_GROUPS:
        bucket = demand_buckets[demand]
        print(f"{network} - Demand {demand}: {len(bucket)} rows available")

        if len(bucket) > samples_per_demand:
            bucket = np.random.choice(bucket, samples_per_demand, replace=False)

        all_samples.extend(bucket)

print("\nTotal training samples:", len(all_samples))

train_index_df = pd.DataFrame(all_samples)
train_index_df.to_csv("train_index_sampled.csv", index=False)

print("Stratified sampled train index saved.")