import os
import csv
from tqdm import tqdm

DATA_ROOT = "../datasets/NetworkList"
TRAIN_NETWORKS = ["Network_1", "Network_2", "Network_4", "Network_5", "Network_7"]
TEST_NETWORKS = ["Network_3", "Network_6", "Network_8"]
TRAIN_OUTPUT = "../csv/train_index.csv"
TEST_OUTPUT = "../csv/test_index.csv"

FIELDNAMES = [
    "file_path", "row_idx", "network_id", "material_id",
    "file_type", "leak_scenario", "demand_multiplier"
]

material_map = {}
material_counter = 0


def process_networks(network_list, output_csv):
    global material_counter

    total_samples = 0

    with open(output_csv, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for network in network_list:
            network_path = os.path.join(DATA_ROOT, network)
            if not os.path.exists(network_path):
                print(f"Network path not found: {network_path}")
                continue

            for material in os.listdir(network_path):
                material_path = os.path.join(network_path, material)
                if not os.path.isdir(material_path):
                    continue

                if material not in material_map:
                    material_map[material] = material_counter
                    material_counter += 1
                material_id = material_map[material]

                for file in tqdm(os.listdir(material_path), desc=f"{network}-{material}"):
                    if not file.endswith(".csv"):
                        continue

                    file_path = os.path.join(material_path, file)
                    name = file.replace(".csv", "")
                    parts = name.split("-")

                    try:
                        if parts[0] == "base":
                            file_type = "base"
                            leak_scenario = -1
                            demand = float(parts[1].replace("_", "."))
                        elif parts[0] == "leak":
                            file_type = "leak"
                            leak_scenario = int(parts[1])
                            demand = float(parts[2].replace("_", "."))
                        else:
                            print(f"Unrecognised file format: {file}")
                            continue
                    except (IndexError, ValueError) as e:
                        print(f"Could not parse filename {file}: {e}")
                        continue

                    try:
                        with open(file_path, "r") as f:
                            num_rows = sum(1 for _ in f) - 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    batch = []
                    for row_idx in range(num_rows):
                        batch.append({
                            "file_path": file_path,
                            "row_idx": row_idx,
                            "network_id": network,
                            "material_id": material_id,
                            "file_type": file_type,
                            "leak_scenario": leak_scenario,
                            "demand_multiplier": demand
                        })
                        if len(batch) >= 1000:
                            writer.writerows(batch)
                            batch = []
                            total_samples += 1000

                    if batch:
                        writer.writerows(batch)
                        total_samples += len(batch)

    return total_samples


os.makedirs("../csv", exist_ok=True)

print("Building training index...")
train_total = process_networks(TRAIN_NETWORKS, TRAIN_OUTPUT)
print(f"Total training samples: {train_total}")

print("\nBuilding test index...")
test_total = process_networks(TEST_NETWORKS, TEST_OUTPUT)
print(f"Total test samples: {test_total}")

print("\nMaterial mapping:", material_map)