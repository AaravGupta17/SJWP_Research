"""
evaluate_breakdown_d.py — AcousticLeakNet Model D Ablation Study
=================================================================
Identical to evaluate_breakdown.py except:
  - Loads best_model_d.pt  (or latest checkpoint if best not saved yet)
  - Reads from cache_d  (Model D cache uses 4 leak types + freq-dep attenuation)

Usage:
    python evaluate_breakdown_d.py

    # If training stopped early and best_model_d.pt wasn't saved, point at
    # the latest epoch checkpoint instead:
    python evaluate_breakdown_d.py --ckpt ../models/model_d_epoch18.pt

Output:
    Prints full breakdown tables to console.
    Saves evaluate_breakdown_d_results.csv for paper tables.
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CKPT = Path("../models/best_model_d.pt")
CACHE_ROOT   = Path("../cache_d")          # ← Model D cache
CSV_DIR      = Path("../csv")

TEST_SPLITS = {
    "Network_3 (L-TOWN, 905 pipes)":  "test_network_3",
    "Network_6 (KY15)":               "test_network_6",
    "Network_8 (Richmond, 44 pipes)": "test_network_8",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Cached dataset that also returns metadata ─────────────────────────────────
class CachedDatasetWithMeta(Dataset):
    def __init__(self, split_name: str):
        cache_dir    = CACHE_ROOT / split_name
        self.signals = np.load(str(cache_dir / "signals.npy"), mmap_mode="r")
        self.labels  = np.load(str(cache_dir / "labels.npy"),  mmap_mode="r")

        index_csv  = CSV_DIR / f"{split_name}.csv"
        self.index = pd.read_csv(str(index_csv))

        assert len(self.signals) == len(self.index), \
            f"Cache/index mismatch: {len(self.signals)} vs {len(self.index)}"

        self._extract_metadata()
        print(f"  [{split_name}] {len(self.signals):,} samples loaded")

    def _extract_metadata(self):
        df = self.index.copy()

        self.demand    = df["demand_multiplier"].values.astype(np.float32)
        self.file_type = df["file_type"].values

        def extract_material(path):
            for mat in ["CI", "DI", "PVC", "STEEL"]:
                if f"/{mat}/" in str(path).replace("\\", "/") or \
                   f"\\{mat}\\" in str(path):
                    return mat
            return "UNKNOWN"

        def extract_network(path):
            p = str(path).replace("\\", "/")
            for n in ["Network_1","Network_2","Network_3","Network_4",
                      "Network_5","Network_6","Network_7","Network_8"]:
                if f"/{n}/" in p:
                    return n
            return "UNKNOWN"

        self.material = np.array([extract_material(p) for p in df["file_path"].values])
        self.network  = np.array([extract_network(p)  for p in df["file_path"].values])

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals[idx].copy().astype(np.float32))
        lab = self.labels[idx]
        return (
            sig,
            torch.tensor(lab[0], dtype=torch.float32),  # leak_status
            torch.tensor(lab[1], dtype=torch.float32),  # position
            torch.tensor(lab[2], dtype=torch.float32),  # severity
            torch.tensor(lab[3], dtype=torch.float32),  # pos_valid
            idx,
        )


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    model.eval()
    records = []
    for batch in tqdm(loader, desc="  Inference", leave=False, dynamic_ncols=True):
        sig, true_det, true_pos, true_sev, pos_valid, idxs = batch
        sig     = sig.to(DEVICE)
        scalars = torch.zeros(sig.size(0), 11, device=DEVICE)

        pred_det, pred_pos, pred_sev = model(sig, scalars)
        probs = torch.sigmoid(pred_det).cpu().numpy()
        p_pos = pred_pos.cpu().numpy()
        p_sev = pred_sev.cpu().numpy()

        for i, idx in enumerate(idxs.numpy()):
            records.append({
                "pred_prob":  float(probs[i]),
                "pred_pos":   float(p_pos[i]),
                "pred_sev":   float(p_sev[i]),
                "true_label": float(true_det[i].item()),
                "true_pos":   float(true_pos[i].item()),
                "true_sev":   float(true_sev[i].item()),
                "pos_valid":  float(pos_valid[i].item()),
                "material":   dataset.material[idx],
                "demand":     float(dataset.demand[idx]),
                "network":    dataset.network[idx],
                "file_type":  dataset.file_type[idx],
            })
    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────
def safe_auroc(y_true, y_prob):
    y_true = np.array(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

def safe_f1(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) >= thresh).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(f1_score(y_true, y_pred, zero_division=0))

def safe_r2(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    if len(y_true) < 2 or y_true.std() < 1e-6:
        return float("nan")
    return float(r2_score(y_true, y_pred))

def compute_metrics(df_group):
    if len(df_group) == 0:
        return {"N": 0, "N_leak": 0, "N_pos": 0,
                "AUROC": float("nan"), "F1": float("nan"),
                "PosMAE": float("nan"), "SevR2": float("nan")}

    auroc = safe_auroc(df_group["true_label"], df_group["pred_prob"])
    f1    = safe_f1(df_group["true_label"],    df_group["pred_prob"])

    loc_df  = df_group[(df_group["true_label"] == 1) & (df_group["pos_valid"] == 1)]
    pos_mae = float(mean_absolute_error(loc_df["true_pos"], loc_df["pred_pos"])) \
              if len(loc_df) >= 5 else float("nan")

    sev_df = df_group[df_group["true_label"] == 1]
    sev_r2 = safe_r2(sev_df["true_sev"], sev_df["pred_sev"]) \
             if len(sev_df) >= 5 else float("nan")

    return {
        "N":      len(df_group),
        "N_leak": int((df_group["true_label"] == 1).sum()),
        "N_pos":  len(loc_df),
        "AUROC":  auroc,
        "F1":     f1,
        "PosMAE": pos_mae,
        "SevR2":  sev_r2,
    }


def print_table(title, rows):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Group':<35} {'N':>7} {'AUROC':>8} {'F1':>8} {'PosMAE':>9} {'SevR²':>8}")
    print(f"{'-'*80}")
    for group, m in rows:
        auroc = f"{m['AUROC']:.4f}"  if not np.isnan(m['AUROC'])  else "   —"
        f1    = f"{m['F1']:.4f}"     if not np.isnan(m['F1'])     else "   —"
        pos   = f"{m['PosMAE']:.4f}" if not np.isnan(m['PosMAE']) else "   —"
        sev   = f"{m['SevR2']:.4f}"  if not np.isnan(m['SevR2'])  else "   —"
        print(f"{group:<35} {m['N']:>7,} {auroc:>8} {f1:>8} {pos:>9} {sev:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Path to Model D checkpoint (default: best_model_d.pt)")
    args = parser.parse_args()

    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Cache root: {CACHE_ROOT}")

    from model import AcousticLeakNet
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = AcousticLeakNet(signal_length=2000, n_scalars=11,
                            base_channels=64, dropout=0.3).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch','?')} | "
          f"Val AUROC: {ckpt.get('val_auroc','?')} | "
          f"Val SevR²: {ckpt.get('val_sev_r2','?')}")

    all_results = []
    for net_label, split_name in TEST_SPLITS.items():
        print(f"\nProcessing {net_label}...")
        dataset = CachedDatasetWithMeta(split_name)
        df      = run_inference(model, dataset)
        df["split"] = net_label
        all_results.append(df)

    full_df = pd.concat(all_results, ignore_index=True)

    # ── TABLE 1: By Network ───────────────────────────────────────────────────
    rows = []
    for net in sorted(full_df["network"].unique()):
        rows.append((net, compute_metrics(full_df[full_df["network"] == net])))
    rows.append(("ALL NETWORKS (combined)", compute_metrics(full_df)))
    print_table("BREAKDOWN BY NETWORK", rows)

    # ── TABLE 2: By Pipe Material ─────────────────────────────────────────────
    rows = []
    for mat in ["CI", "DI", "PVC", "STEEL"]:
        rows.append((f"Material: {mat}", compute_metrics(full_df[full_df["material"] == mat])))
    print_table("BREAKDOWN BY PIPE MATERIAL", rows)

    # ── TABLE 3: By Demand Multiplier ─────────────────────────────────────────
    rows = []
    for dm in sorted(full_df["demand"].unique()):
        sub = full_df[full_df["demand"].round(2) == round(dm, 2)]
        rows.append((f"Demand × {dm:.1f}", compute_metrics(sub)))
    print_table("BREAKDOWN BY DEMAND MULTIPLIER", rows)

    # ── TABLE 4: Network × Material cross-tab ────────────────────────────────
    rows = []
    for net in sorted(full_df["network"].unique()):
        for mat in ["CI", "DI", "PVC", "STEEL"]:
            sub = full_df[(full_df["network"] == net) & (full_df["material"] == mat)]
            if len(sub) > 0:
                rows.append((f"{net} × {mat}", compute_metrics(sub)))
    print_table("BREAKDOWN BY NETWORK × MATERIAL", rows)

    # ── TABLE 5: Dual vs single sensor ───────────────────────────────────────
    rows = []
    leak_df = full_df[full_df["true_label"] == 1]
    dual    = leak_df[leak_df["pos_valid"] == 1]
    single  = leak_df[leak_df["pos_valid"] == 0]
    rows.append((f"Dual-sensor  (n_leak={len(dual):,})",
                 compute_metrics(full_df[full_df["pos_valid"] == 1])))
    rows.append((f"Single-sensor (n_leak={len(single):,})",
                 compute_metrics(full_df[full_df["pos_valid"] == 0])))
    print_table("BREAKDOWN BY SENSOR CONFIGURATION", rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY STATISTICS — MODEL D")
    print(f"{'='*80}")
    print(f"Total test samples:        {len(full_df):,}")
    print(f"Total leak samples:        {int(full_df['true_label'].sum()):,}")
    print(f"Total no-leak samples:     {int((full_df['true_label']==0).sum()):,}")
    print(f"Dual-sensor leak samples:  "
          f"{int(((full_df['true_label']==1)&(full_df['pos_valid']==1)).sum()):,}")
    print(f"\nMaterial distribution:")
    print(full_df["material"].value_counts().to_string())
    print(f"\nDemand distribution:")
    print(full_df["demand"].value_counts().sort_index().to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = "evaluate_breakdown_d_results.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"\nFull per-sample results saved to: {out_csv}")


if __name__ == "__main__":
    main()