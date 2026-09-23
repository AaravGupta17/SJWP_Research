from dataset_c import LeakDataset
ds = LeakDataset("../csv/val_sampled.csv")
# Find a leak sample
for i in range(len(ds)):
    r = ds[i]
    if r[2].item() == 1:
        print(f"leak_status={r[2].item()} leak_pos={r[3].item()} leak_flow={r[4].item()} pos_valid={r[5].item()}")
        break