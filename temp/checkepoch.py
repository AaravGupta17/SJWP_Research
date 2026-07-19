import pandas as pd
df = pd.read_csv("../csv/train_index.csv")
n5 = df[df["network_id"] == "Network_5"]
print(f"Network_5 total: {len(n5)}")
print(f"Base rows: {len(n5[n5['file_type']=='base'])}")
print(f"Leak rows: {len(n5[n5['file_type']=='leak'])}")