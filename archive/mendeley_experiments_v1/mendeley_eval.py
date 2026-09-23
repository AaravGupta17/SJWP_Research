"""
mendeley_eval.py — Real-World Validation on Mendeley Hydrophone Dataset
========================================================================
Evaluates trained model on real lab recordings from:
Aghashahi et al. (2023) — 47m PVC pipe testbed, 8kHz hydrophones

File naming convention:
  LO = Looped topology, BR = Branched topology
  OL = Orifice Leak, LC = Longitudinal Crack,
  CC = Circumferential Crack, GL = Gasket Leak, NL = No Leak
  0.18/0.47 LPS = flow rate, ND = No Demand
  N = with background noise, NN = no background noise
  H1/H2 = Hydrophone 1 and 2

This is sim-to-real transfer validation:
  Model trained ENTIRELY on EPANET simulation
  Tested on REAL lab recordings — zero real data in training
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from model import AcousticLeakNet

MENDELEY_ROOT = Path("../datasets/Hydrophone/Hydrophone")
MODELS_DIR    = Path("../models")
RESULTS_DIR   = Path("../results")
PLOTS_DIR     = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Model parameters — must match training
SIGNAL_LENGTH = 2000
FS_MODEL      = 5000   # model was trained at 5000 Hz
FS_MENDELEY   = 8000   # Mendeley recordings are at 8000 Hz

LEAK_TYPES = ["Orifice Leak"]
NO_LEAK    = "No-leak"
TOPOLOGIES = ["Looped", "Branched"]


def load_raw_signal(path: str) -> np.ndarray:
    """Load Mendeley .raw file — signed 32-bit PCM at 8000 Hz."""
    raw  = np.fromfile(path, dtype=np.int32)
    sig  = raw.astype(np.float32)
    # Normalise to [-1, 1] range
    mx = np.abs(sig).max()
    if mx > 0:
        sig = sig / mx
    return sig


def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Simple decimation/interpolation to match model's expected fs."""
    from scipy.signal import resample_poly
    from math import gcd
    g   = gcd(target_fs, orig_fs)
    up  = target_fs // g
    down = orig_fs  // g
    return resample_poly(sig, up, down).astype(np.float32)


def normalize_channel(sig: np.ndarray) -> np.ndarray:
    """Z-score normalisation — identical to training preprocessing."""
    mu  = np.mean(sig)
    std = np.std(sig) + 1e-8
    return np.clip((sig - mu) / std, -5.0, 5.0)


def slice_to_windows(h1: np.ndarray, h2: np.ndarray,
                     window_len: int) -> np.ndarray:
    """
    Slice two channel signals into windows of length window_len.
    Returns (N, 2, window_len) array.
    """
    min_len = min(len(h1), len(h2))
    n_windows = min_len // window_len
    windows = []
    for i in range(n_windows):
        s = i * window_len
        e = s + window_len
        ch0 = normalize_channel(h1[s:e])
        ch1 = normalize_channel(h2[s:e])
        windows.append(np.stack([ch0, ch1]))
    return np.array(windows, dtype=np.float32)


def find_paired_files(folder: Path):
    """Find H1/H2 pairs in a folder."""
    files = list(folder.glob("*.raw"))
    h1_files = sorted([f for f in files if "H1" in f.name])
    pairs = []
    for h1 in h1_files:
        h2_name = h1.name.replace("H1", "H2")
        h2 = folder / h2_name
        if h2.exists():
            pairs.append((h1, h2))
    return pairs


@torch.no_grad()
def run_inference(model, windows: np.ndarray, device) -> np.ndarray:
    """Run model on windows, return detection probabilities."""
    probs = []
    batch_size = 64
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i+batch_size]).to(device)
        scalars = torch.zeros(batch.size(0), 11, device=device)
        pred_det, _, _ = model(batch, scalars)
        probs.extend(torch.sigmoid(pred_det).cpu().numpy())
    return np.array(probs)


def evaluate(model, device):
    all_probs  = []
    all_labels = []
    results_by_type = {}
    results_by_topology = {}

    for topology in TOPOLOGIES:
        topo_dir = MENDELEY_ROOT / topology
        if not topo_dir.exists():
            print(f"Topology not found: {topo_dir}")
            continue

        topo_probs  = []
        topo_labels = []

        # Process leak types
        for leak_type in LEAK_TYPES:
            leak_dir = topo_dir / leak_type
            if not leak_dir.exists():
                continue

            pairs = find_paired_files(leak_dir)
            type_probs = []

            for h1_path, h2_path in pairs:
                h1_raw = load_raw_signal(str(h1_path))
                h2_raw = load_raw_signal(str(h2_path))

                # Resample from 8000 Hz to 5000 Hz
                h1 = resample_signal(h1_raw, FS_MENDELEY, FS_MODEL)
                h2 = resample_signal(h2_raw, FS_MENDELEY, FS_MODEL)

                windows = slice_to_windows(h1, h2, SIGNAL_LENGTH)
                if len(windows) == 0:
                    continue

                probs = run_inference(model, windows, device)
                type_probs.extend(probs)
                topo_probs.extend(probs)
                all_probs.extend(probs)

            n = len(type_probs)
            all_labels.extend([1] * n)
            topo_labels.extend([1] * n)

            key = f"{topology}/{leak_type}"
            if type_probs:
                results_by_type[key] = {
                    "mean_prob": float(np.mean(type_probs)),
                    "detection_rate": float((np.array(type_probs) >= 0.5).mean()),
                    "n_windows": n,
                    "label": 1
                }
                print(f"  {key}: mean_prob={np.mean(type_probs):.3f} "
                      f"detection_rate={(np.array(type_probs)>=0.5).mean():.3f} "
                      f"({n} windows)")

        # Process no-leak
        no_leak_dir = topo_dir / NO_LEAK
        if no_leak_dir.exists():
            pairs = find_paired_files(no_leak_dir)
            nl_probs = []

            for h1_path, h2_path in pairs:
                h1_raw = load_raw_signal(str(h1_path))
                h2_raw = load_raw_signal(str(h2_path))
                h1 = resample_signal(h1_raw, FS_MENDELEY, FS_MODEL)
                h2 = resample_signal(h2_raw, FS_MENDELEY, FS_MODEL)
                windows = slice_to_windows(h1, h2, SIGNAL_LENGTH)
                if len(windows) == 0:
                    continue
                probs = run_inference(model, windows, device)
                nl_probs.extend(probs)
                topo_probs.extend(probs)
                all_probs.extend(probs)

            n = len(nl_probs)
            all_labels.extend([0] * n)
            topo_labels.extend([0] * n)

            key = f"{topology}/No-leak"
            results_by_type[key] = {
                "mean_prob": float(np.mean(nl_probs)),
                "false_alarm_rate": float((np.array(nl_probs) >= 0.5).mean()),
                "n_windows": n,
                "label": 0
            }
            print(f"  {key}: mean_prob={np.mean(nl_probs):.3f} "
                  f"false_alarm={(np.array(nl_probs)>=0.5).mean():.3f} "
                  f"({n} windows)")

        # Per topology metrics
        if topo_probs and len(set(topo_labels)) > 1:
            topo_arr  = np.array(topo_probs)
            topo_true = np.array(topo_labels)
            auroc = roc_auc_score(topo_true, topo_arr)
            f1    = f1_score(topo_true, (topo_arr >= 0.5).astype(int))
            acc   = accuracy_score(topo_true, (topo_arr >= 0.5).astype(int))
            results_by_topology[topology] = {
                "auroc": round(auroc, 4),
                "f1":    round(f1,    4),
                "acc":   round(acc,   4),
                "n":     len(topo_probs)
            }

    # Overall metrics
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\n{'='*55}")
    print(f"  MENDELEY SIM-TO-REAL TRANSFER RESULTS")
    print(f"{'='*55}")
    print(f"  Total windows: {len(all_probs):,}")
    print(f"  Leak windows:  {all_labels.sum():,}")
    print(f"  Base windows:  {(all_labels==0).sum():,}")

    if len(set(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        f1    = f1_score(all_labels, preds)
        acc   = accuracy_score(all_labels, preds)

        print(f"\n  Overall AUROC    : {auroc:.4f}")
        print(f"  Overall F1       : {f1:.4f}")
        print(f"  Overall Accuracy : {acc:.4f}")
        print(f"\n{classification_report(all_labels, preds, target_names=['No-Leak','Leak'])}")

        print(f"\n  Per topology:")
        for topo, m in results_by_topology.items():
            print(f"    {topo}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f} "
                  f"Acc={m['acc']:.4f}")

        # Plot score distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(all_probs[all_labels==0], bins=40, alpha=0.6,
                     color="steelblue", label="No-Leak", density=True)
        axes[0].hist(all_probs[all_labels==1], bins=40, alpha=0.6,
                     color="crimson", label="Leak", density=True)
        axes[0].axvline(0.5, color="black", linestyle="--", label="Threshold")
        axes[0].set_title(f"Mendeley Sim-to-Real Transfer\nAUROC={auroc:.4f}")
        axes[0].set_xlabel("Detection Probability")
        axes[0].legend()

        # Per leak type detection rates
        leak_types_plot = [k for k, v in results_by_type.items() if v["label"] == 1]
        detection_rates = [results_by_type[k]["detection_rate"] for k in leak_types_plot]
        axes[1].barh(leak_types_plot, detection_rates, color="crimson", alpha=0.7)
        axes[1].axvline(0.5, color="black", linestyle="--")
        axes[1].set_xlabel("Detection Rate")
        axes[1].set_title("Detection Rate by Leak Type")
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mendeley_validation.png", dpi=150)
        print(f"\nPlot saved: {PLOTS_DIR}/mendeley_validation.png")

        # Save results
        results = {
            "overall": {
                "auroc": round(float(auroc), 4),
                "f1":    round(float(f1),    4),
                "acc":   round(float(acc),   4),
                "n_total":      int(len(all_probs)),
                "n_leak":       int(all_labels.sum()),
                "n_no_leak":    int((all_labels==0).sum()),
            },
            "by_topology": results_by_topology,
            "by_type":     results_by_type,
        }
        with open(RESULTS_DIR / "mendeley_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {RESULTS_DIR}/mendeley_results.json")

        return auroc
    else:
        print("Not enough label diversity for AUROC calculation")
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading best model...")
    ckpt  = torch.load(MODELS_DIR / "best_model_v2.pt", map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = AcousticLeakNet(
        signal_length=SIGNAL_LENGTH, n_scalars=11,
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded — trained epoch {ckpt['epoch']} | "
          f"val AUROC={ckpt['val_auroc']:.4f}")

    print(f"\nRunning on Mendeley hydrophone recordings...")
    print(f"Root: {MENDELEY_ROOT}")
    print("(Model trained on EPANET simulation only — zero real data in training)\n")

    auroc = evaluate(model, device)

    if auroc is not None:
        print(f"\n{'='*55}")
        if auroc >= 0.90:
            print(f"  ✓ STRONG sim-to-real transfer: AUROC={auroc:.4f}")
            print(f"  Model trained on simulation generalises to real recordings")
        elif auroc >= 0.75:
            print(f"  ~ MODERATE sim-to-real transfer: AUROC={auroc:.4f}")
            print(f"  Partial generalisation — sim-to-real gap identified")
        else:
            print(f"  ✗ WEAK sim-to-real transfer: AUROC={auroc:.4f}")
            print(f"  Significant sim-to-real gap — correlated noise needed")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()