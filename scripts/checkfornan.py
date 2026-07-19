import pandas as pd

df = pd.read_csv("../csv/train_index_sampled.csv")

print(f"Total rows: {len(df)}")
print(f"\n--- NaN check ---")
print(df.isna().sum())

print(f"\n--- Class balance ---")
print(df['file_type'].value_counts())

print(f"\n--- Network balance ---")
print(df.groupby(['file_type', 'network_id']).size().unstack())

print(f"\n--- Demand balance ---")
print(df.groupby(['file_type', 'demand_multiplier']).size().unstack())

print(f"\n--- Material balance ---")
print(df.groupby(['file_type', 'material_id']).size().unstack())

print(f"\n--- Shuffle check (first 10 file_types) ---")
print(df['file_type'].head(10).tolist())  # should not be all same class