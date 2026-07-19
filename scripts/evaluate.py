"""
evaluate.py — Test Evaluation on Networks 3, 6, 8
===================================================
Evaluates trained model on each test network separately.
Reports: AUROC, F1, Accuracy, Position MAE, Severity R²

Run after: python train.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             mean_absolute_error, r2_score,
                             confusion_matrix, classification_report)
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from model import AcousticLeakNet

CACHE_ROOT  = Path("../cache")
MODELS_DIR  = Path("../models")
PLOTS_DIR   = Path("../plots")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SPLITS = ["test_network_3", "test_network_6", "test_network_8"]
NETWORK_NAMES = {
    "test_network_3": "L-TOWN (Network 3)",
    "test_network_6": "KY15 (Network 6)",
    "test_network_8": "Richmond (Network 8)",
}

MEAN_SENSOR_SEPARATION_M = 40.4  # metres, computed from training data


class CachedDataset(Dataset):
    def __init__(self, split: str):
        cache_dir    = CACHE_ROOT / split
        self.signals = np.load(str(cache_dir / "signals.npy"), mmap_mode="r")
        self.labels  = np.load(str(cache_dir / "labels.npy"),  mmap_mode="r")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy().astype(np.float32)
        lab = self.labels[idx]
        return (torch.from_numpy(sig),
                torch.tensor(lab[0], dtype=torch.float32),
                torch.tensor(lab[1], dtype=torch.float32),
                torch.tensor(lab[2], dtype=torch.float32),
                torch.tensor(lab[3], dtype=torch.float32))


@torch.no_grad()
def evaluate_split(model, split: str, device) -> dict:
    ds     = CachedDataset(split)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    all_prob, all_true = [], []
    all_pos_p, all_pos_t = [], []
    all_sev_p, all_sev_t = [], []

    model.eval()
    pbar = tqdm(loader, desc=f"Evaluating {split}", unit="batch",
                dynamic_ncols=True)

    for sig, true_det, true_pos, true_sev, pos_valid in pbar:
        sig     = sig.to(device)
        scalars = torch.zeros(sig.size(0), 9, device=device)

        pred_det, pred_pos, pred_sev = model(sig, scalars)
        probs  = torch.sigmoid(pred_det).cpu().numpy()
        det_np = true_det.numpy()
        pv_np  = pos_valid.numpy()

        all_prob.extend(probs)
        all_true.extend(det_np)

        lm = det_np == 1
        if lm.sum() > 0:
            all_sev_p.extend(pred_sev.cpu().numpy()[lm])
            all_sev_t.extend(true_sev.numpy()[lm])
        pm = lm & (pv_np == 1)
        if pm.sum() > 0:
            all_pos_p.extend(pred_pos.cpu().numpy()[pm])
            all_pos_t.extend(true_pos.numpy()[pm])

    pbar.close()

    all_prob = np.array(all_prob)
    all_true = np.array(all_true)

    # Remove any NaN predictions
    valid    = ~np.isnan(all_prob)
    all_prob = all_prob[valid]
    all_true = all_true[valid]
    y_pred   = (all_prob >= 0.5).astype(int)

    auroc   = roc_auc_score(all_true, all_prob)
    f1      = f1_score(all_true, y_pred)
    acc     = accuracy_score(all_true, y_pred)
    cm      = confusion_matrix(all_true, y_pred)
    pos_mae_norm = mean_absolute_error(all_pos_t, all_pos_p) if all_pos_t else float("nan")
    pos_mae_m    = pos_mae_norm * MEAN_SENSOR_SEPARATION_M
    sev_r2  = r2_score(all_sev_t, all_sev_p) if len(all_sev_t) > 1 else float("nan")

    net_name = NETWORK_NAMES.get(split, split)
    print(f"\n{'='*55}")
    print(f"  {net_name}")
    print(f"{'='*55}")
    print(f"  AUROC    : {auroc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Pos MAE  : {pos_mae_norm:.4f} normalized | {pos_mae_m:.2f} metres")
    print(f"  Sev R²   : {sev_r2:.4f}")
    print(f"\n{classification_report(all_true, y_pred, target_names=['Base','Leak'])}")

    return {
        "network":        net_name,
        "auroc":          round(auroc, 4),
        "f1":             round(f1, 4),
        "accuracy":       round(acc, 4),
        "pos_mae_norm":   round(pos_mae_norm, 4) if not np.isnan(pos_mae_norm) else None,
        "pos_mae_metres": round(pos_mae_m, 2)    if not np.isnan(pos_mae_norm) else None,
        "sev_r2":         round(sev_r2, 4)       if not np.isnan(sev_r2)       else None,
        "n_total":        int(len(all_true)),
        "n_leak":         int(all_true.sum()),
        "confusion_matrix": cm.tolist(),
        "probs":          all_prob.tolist(),
        "true":           all_true.tolist(),
    }


def plot_results(all_results: list):
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))

    for i, res in enumerate(all_results):
        probs = np.array(res["probs"])
        true  = np.array(res["true"])
        axes[0, i].hist(probs[true == 0], bins=40, alpha=0.6,
                        label="Base", color="steelblue", density=True)
        axes[0, i].hist(probs[true == 1], bins=40, alpha=0.6,
                        label="Leak", color="crimson", density=True)
        axes[0, i].axvline(0.5, color="black", linestyle="--")
        axes[0, i].set_title(f"{res['network']}\nAUROC={res['auroc']:.4f}")
        axes[0, i].set_xlabel("Detection Probability")
        axes[0, i].legend()

        cm = np.array(res["confusion_matrix"])
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, i],
                        xticklabels=["Base", "Leak"],
                        yticklabels=["Base", "Leak"])
        else:
            axes[1, i].imshow(cm, cmap="Blues")
            for row in range(2):
                for col in range(2):
                    axes[1, i].text(col, row, str(cm[row, col]),
                                    ha="center", va="center", fontsize=12)
            axes[1, i].set_xticks([0, 1])
            axes[1, i].set_xticklabels(["Base", "Leak"])
            axes[1, i].set_yticks([0, 1])
            axes[1, i].set_yticklabels(["Base", "Leak"])
        axes[1, i].set_title(f"Confusion Matrix\nF1={res['f1']:.4f}")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("True")

    plt.suptitle("Test Evaluation — L-TOWN, KY15, Richmond (Unseen Networks)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_evaluation.png", dpi=150)
    print(f"\nSaved: {PLOTS_DIR}/test_evaluation.png")


def print_summary(all_results: list):
    print(f"\n{'='*55}")
    print(f"  SUMMARY — Cross-Network Generalisation")
    print(f"{'='*55}")
    print(f"  {'Network':<25} {'AUROC':>8} {'F1':>8} {'Acc':>8} {'MAE(m)':>8}")
    print(f"  {'-'*57}")
    for r in all_results:
        pos = f"{r['pos_mae_metres']:.2f}m" if r["pos_mae_metres"] else "  N/A"
        print(f"  {r['network']:<25} {r['auroc']:>8.4f} {r['f1']:>8.4f} "
              f"{r['accuracy']:>8.4f} {pos:>8}")

    aurocs = [r["auroc"] for r in all_results]
    print(f"\n  Mean AUROC: {np.mean(aurocs):.4f}")
    print(f"  Min  AUROC: {min(aurocs):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading best model...")
    ckpt  = torch.load(MODELS_DIR / "best_model.pt",
                       map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = AcousticLeakNet(
        signal_length=2000, n_scalars=9,
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded epoch {ckpt['epoch']} | "
          f"val AUROC={ckpt['val_auroc']:.4f} (Net6 — Network 7)")

    all_results = []
    for split in TEST_SPLITS:
        cache_path = CACHE_ROOT / split / "signals.npy"
        if not cache_path.exists():
            print(f"Cache not found for {split} — run pregenerate.py first")
            continue
        res = evaluate_split(model, split, device)
        all_results.append(res)

    if all_results:
        plot_results(all_results)
        print_summary(all_results)

        results_out = [{k: v for k, v in r.items()
                        if k not in ("probs", "true")}
                       for r in all_results]
        with open(RESULTS_DIR / "test_results.json", "w") as f:
            json.dump(results_out, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR}/test_results.json")


if __name__ == "__main__":
    main()