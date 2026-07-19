import os
import json
import wntr
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

EVAL_ROOT = r"C:\Users\armaan\Downloads\EvalData203428"

MODEL1_FILE = "eval1.json"
MODEL2_FILE = "n_eval.json"

# Network mapping from your image
NETWORKS = {
    1: ("NW_Model1", "../inp/NW_Model1.inp"),
    2: ("Net3", "../inp/Net3_(BWSN-2)_Morph_Error_Free_1s-WQ.inp"),
    3: ("L-TOWN", "../inp/L-TOWN.inp"),
    4: ("BWSN", "../inp/BWSN_Network.inp"),
    5: ("KY9", "../inp/ky9.inp"),
    6: ("KY15", "../inp/ky15.inp"),
    7: ("Net6", "../inp/Net6.inp"),
    8: ("Richmond", "../inp/Richmond_skeleton.inp")
}

# ==========================================
# PIPE COUNT FROM INP
# ==========================================

pipe_counts = {}

for net_id, (name, path) in NETWORKS.items():
    if os.path.exists(path):
        wn = wntr.network.WaterNetworkModel(path)
        pipe_counts[name] = len(wn.pipe_name_list)
    else:
        pipe_counts[name] = ""

# ==========================================
# LOAD JSON
# ==========================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_json(json_data):
    if isinstance(json_data, dict):
        json_data = list(json_data.values())
    return json_data

model1 = parse_json(load_json(os.path.join(EVAL_ROOT, MODEL1_FILE)))
model2 = parse_json(load_json(os.path.join(EVAL_ROOT, MODEL2_FILE)))

# ==========================================
# BUILD TABLE 2
# ==========================================

table2_rows = []

for m1, m2 in zip(model1, model2):

    network_raw = m1["network"]

    if "L-TOWN" in network_raw or "Anytown" in network_raw:
        net = "L-TOWN"
    elif "KY15" in network_raw:
        net = "KY15"
    elif "Richmond" in network_raw or "Kentucky" in network_raw:
        net = "Richmond"
    else:
        continue

    pipes = str(pipe_counts.get(net, ""))

    # Model A
    table2_rows.append([
        net,
        pipes,
        "Model A",
        f"{m1.get('auroc', ''):.4f}" if m1.get("auroc") else "",
        f"{m1.get('f1', ''):.4f}" if m1.get("f1") else "",
        f"{m1.get('pos_mae', ''):.4f}" if m1.get("pos_mae") else "",
        "",
        f"{m1.get('sev_r2', ''):.2f}" if m1.get("sev_r2") else ""
    ])

    # Model B
    table2_rows.append([
        net,
        pipes,
        "Model B",
        f"{m2.get('auroc', ''):.4f}" if m2.get("auroc") else "",
        f"{m2.get('f1', ''):.4f}" if m2.get("f1") else "",
        f"{m2.get('pos_mae', ''):.4f}" if m2.get("pos_mae") else "",
        "",
        f"{m2.get('sev_r2', ''):.2f}" if m2.get("sev_r2") else ""
    ])

table2_columns = [
    "Network",
    "Pipes",
    "Model",
    "AUROC",
    "F1 Score",
    "Localisation MAE (p)",
    "Physical Error (m)",
    "Severity R²"
]

df2 = pd.DataFrame(table2_rows, columns=table2_columns)

# ==========================================
# BUILD TABLE 1 (NOW WITH REAL PIPE COUNTS)
# ==========================================

table1_data = [
    [1, "NW_Model1", "Training", pipe_counts.get("NW_Model1", ""), "Mixed", "Included in 981,408"],
    [2, "Net3", "Training", pipe_counts.get("Net3", ""), "Mixed", "Included in 981,408"],
    [4, "BWSN", "Training", pipe_counts.get("BWSN", ""), "Mixed", "Included in 981,408"],
    [5, "KY9", "Training", pipe_counts.get("KY9", ""), "Mixed", "Included in 981,408"],
    [7, "Net6", "Validation", pipe_counts.get("Net6", ""), "Mixed", "91,896"],
    [3, "L-TOWN", "Testing", pipe_counts.get("L-TOWN", ""), "Dual + Single", "21,720"],
    [6, "KY15", "Testing", pipe_counts.get("KY15", ""), "Dual + Single", "15,888"],
    [8, "Richmond", "Testing", pipe_counts.get("Richmond", ""), "Predominantly Single", "1,056"],
]

table1_columns = [
    "Network ID",
    "Network Name",
    "Role",
    "Pipes",
    "Sensor Configuration",
    "Samples Used"
]

df1 = pd.DataFrame(table1_data, columns=table1_columns)

# ==========================================
# EXPORT FUNCTION
# ==========================================

def save_table(df, filename, title):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# EXPORT
# ==========================================

save_table(df1, "Table1.png", "Table 1. Dataset Composition and Network Split")
save_table(df2, "Table2.png", "Table 2. Cross-Network Performance Comparison — Model A vs Model B")

print("Tables generated successfully.")