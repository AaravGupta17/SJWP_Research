"""
mendeley_eval_accelerometer.py — Mendeley Accelerometer Validation
===================================================================
Tests existing model on Mendeley ACCELEROMETER data (A1, A2 CSV files)
Auto-detects sampling rate from CSV timestamps.
Saves mendeley_accelerometer_results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from math import gcd
from scipy.signal import resample_poly

from model import AcousticLeakNet

# ── Paths ──────────────────────────────────────────────────────────────────
MENDELEY_ROOT = Path("../datasets/Accelerometer/Accelerometer/")
MODELS_DIR    = Path("../models")
RESULTS_DIR   = Path("../results")
PLOTS_DIR     = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_NAME     = "best_model_mend.pt"
SIGNAL_LENGTH = 2000
FS_MODEL      = 5000

LEAK_TYPES = ["Orifice Leak", "Longitudinal Crack",
              "Circumferential Crack", "Gasket Leak"]
NO_LEAK    = "No-leak"
TOPOLOGIES = ["Looped", "Branched"]

OLD_HYDROPHONE_AUROC = 0.40


# ── Signal loading ──────────────────────────────────────────────────────────
def load_csv_signal(path: str):
    """
    Load Mendeley accelerometer CSV.
    Returns (signal, fs) where fs is auto-detected from time column.
    """
    df = pd.read_csv(path, header=0, low_memory=False)

    if not hasattr(load_csv_signal, "_printed"):
        print(f"\n  [CSV DEBUG] File: {Path(path).name}")
        print(f"  [CSV DEBUG] Shape: {df.shape}")
        print(f"  [CSV DEBUG] Columns: {list(df.columns)}")
        print(f"  [CSV DEBUG] First 3 rows:\n{df.head(3)}\n")
        load_csv_signal._printed = True

    # Auto-detect sampling rate from time column (first column)
    time_col = df.iloc[:, 0].values.astype(np.float64)
    dt       = np.median(np.diff(time_col[:1000]))   # use median of first 1000 diffs for robustness
    fs       = int(round(1.0 / dt))

    if not hasattr(load_csv_signal, "_fs_printed"):
        print(f"  [CSV DEBUG] Detected dt={dt:.8f}s  →  fs={fs} Hz\n")
        load_csv_signal._fs_printed = True

    # Signal is last column (Value)
    sig = df.iloc[:, -1].values.astype(np.float32)
    mx  = np.abs(sig).max()
    sig = sig / mx if mx > 0 else sig

    return sig, fs


def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return sig
    g    = gcd(target_fs, orig_fs)
    up   = target_fs // g
    down = orig_fs   // g
    return resample_poly(sig, up, down).astype(np.float32)


def normalize_joint(h1: np.ndarray, h2: np.ndarray):
    joint = np.concatenate([h1, h2])
    mu    = np.mean(joint)
    std   = np.std(joint) + 1e-8
    h1n   = np.clip((h1 - mu) / std, -5.0, 5.0)
    h2n   = np.clip((h2 - mu) / std, -5.0, 5.0)
    return h1n, h2n


def slice_to_windows(h1: np.ndarray, h2: np.ndarray, window_len: int) -> np.ndarray:
    min_len   = min(len(h1), len(h2))
    n_windows = min_len // window_len
    windows   = []
    for i in range(n_windows):
        s      = i * window_len
        e      = s + window_len
        c1, c2 = normalize_joint(h1[s:e], h2[s:e])
        windows.append(np.stack([c1, c2]))
    return np.array(windows, dtype=np.float32)


def find_paired_files(folder: Path):
    files    = list(folder.glob("*.csv"))
    a1_files = sorted([f for f in files if "A1" in f.name])
    pairs    = []
    for a1 in a1_files:
        a2 = folder / a1.name.replace("A1", "A2")
        if a2.exists():
            pairs.append((a1, a2))
    if not pairs:
        print(f"  WARNING: No A1/A2 CSV pairs found in {folder}")
        all_files = list(folder.glob("*"))
        print(f"  Files present: {[f.name for f in all_files[:10]]}")
    return pairs


# ── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, windows: np.ndarray, device) -> np.ndarray:
    probs = []
    for i in range(0, len(windows), 64):
        batch      = torch.from_numpy(windows[i:i+64]).to(device)
        scalars    = torch.zeros(batch.size(0), 11, device=device)
        pred, _, _ = model(batch, scalars)
        probs.extend(torch.sigmoid(pred).cpu().numpy().flatten())
    return np.array(probs)


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, device):
    all_probs, all_labels = [], []
    results_by_type       = {}
    results_by_topology   = {}

    for topology in TOPOLOGIES:
        topo_dir = MENDELEY_ROOT / topology
        if not topo_dir.exists():
            print(f"  Not found: {topo_dir}")
            continue

        topo_probs, topo_labels = [], []
        print(f"\n{'─'*50}")
        print(f"  Topology: {topology}")
        print(f"{'─'*50}")

        # ── Leak types ──────────────────────────────────────────────────────
        for leak_type in LEAK_TYPES:
            leak_dir = topo_dir / leak_type
            if not leak_dir.exists():
                print(f"  Skipping {leak_type} — folder not found")
                continue

            pairs      = find_paired_files(leak_dir)
            type_probs = []

            for a1_path, a2_path in pairs:
                try:
                    a1_raw, fs_acc = load_csv_signal(str(a1_path))
                    a2_raw, _      = load_csv_signal(str(a2_path))
                except Exception as e:
                    print(f"  ERROR loading {a1_path.name}: {e}")
                    continue

                a1      = resample_signal(a1_raw, fs_acc, FS_MODEL)
                a2      = resample_signal(a2_raw, fs_acc, FS_MODEL)
                windows = slice_to_windows(a1, a2, SIGNAL_LENGTH)

                if len(windows) == 0:
                    print(f"  WARNING: 0 windows from {a1_path.name}")
                    continue

                p = run_inference(model, windows, device)
                type_probs.extend(p)
                topo_probs.extend(p)
                all_probs.extend(p)

            n = len(type_probs)
            all_labels.extend([1] * n)
            topo_labels.extend([1] * n)

            key = f"{topology}/{leak_type}"
            if type_probs:
                arr      = np.array(type_probs)
                det_rate = (arr >= 0.5).mean()
                results_by_type[key] = {
                    "mean_prob":      float(arr.mean()),
                    "detection_rate": float(det_rate),
                    "n_windows":      n,
                    "label":          1
                }
                print(f"  {leak_type:25s}: mean={arr.mean():.3f}  "
                      f"det={det_rate:.3f}  ({n} windows)")

        # ── No-leak ─────────────────────────────────────────────────────────
        nl_dir = topo_dir / NO_LEAK
        if nl_dir.exists():
            pairs    = find_paired_files(nl_dir)
            nl_probs = []

            for a1_path, a2_path in pairs:
                try:
                    a1_raw, fs_acc = load_csv_signal(str(a1_path))
                    a2_raw, _      = load_csv_signal(str(a2_path))
                except Exception as e:
                    print(f"  ERROR loading {a1_path.name}: {e}")
                    continue

                a1      = resample_signal(a1_raw, fs_acc, FS_MODEL)
                a2      = resample_signal(a2_raw, fs_acc, FS_MODEL)
                windows = slice_to_windows(a1, a2, SIGNAL_LENGTH)

                if len(windows) == 0:
                    continue

                p = run_inference(model, windows, device)
                nl_probs.extend(p)
                topo_probs.extend(p)
                all_probs.extend(p)

            n = len(nl_probs)
            all_labels.extend([0] * n)
            topo_labels.extend([0] * n)

            arr = np.array(nl_probs)
            far = (arr >= 0.5).mean()
            key = f"{topology}/No-leak"
            results_by_type[key] = {
                "mean_prob":        float(arr.mean()),
                "false_alarm_rate": float(far),
                "n_windows":        n,
                "label":            0
            }
            print(f"  {'No-leak':25s}: mean={arr.mean():.3f}  "
                  f"FAR={far:.3f}  ({n} windows)")

        # ── Per topology metrics ─────────────────────────────────────────────
        if topo_probs and len(set(topo_labels)) > 1:
            ta = np.array(topo_probs)
            tl = np.array(topo_labels)
            results_by_topology[topology] = {
                "auroc": round(float(roc_auc_score(tl, ta)), 4),
                "f1":    round(float(f1_score(tl, (ta >= 0.5).astype(int))), 4),
                "n":     len(topo_probs)
            }

    # ── Overall metrics ──────────────────────────────────────────────────────
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\n{'='*55}")
    print(f"  ACCELEROMETER RESULTS — MENDELEY SIM-TO-REAL")
    print(f"{'='*55}")

    auroc = None
    if len(set(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        f1    = f1_score(all_labels, preds)
        acc   = accuracy_score(all_labels, preds)

        print(f"  Overall AUROC : {auroc:.4f}")
        print(f"  Overall F1    : {f1:.4f}")
        print(f"  Overall Acc   : {acc:.4f}")
        print(f"\n  Per topology:")
        for topo, m in results_by_topology.items():
            print(f"    {topo}: AUROC={m['auroc']:.4f}  "
                  f"F1={m['f1']:.4f}  n={m['n']}")

        print(f"\n  Per leak type:")
        for key, v in results_by_type.items():
            if v["label"] == 1:
                print(f"    {key:40s}: det={v['detection_rate']:.3f}")
            else:
                print(f"    {key:40s}: FAR={v['false_alarm_rate']:.3f}")

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(all_probs[all_labels == 0], bins=40, alpha=0.6,
                     color="steelblue", label="No-Leak", density=True)
        axes[0].hist(all_probs[all_labels == 1], bins=40, alpha=0.6,
                     color="crimson", label="Leak", density=True)
        axes[0].axvline(0.5, color="black", linestyle="--", label="Threshold")
        axes[0].set_title(
            f"Accelerometer Sim-to-Real Validation\nAUROC={auroc:.4f}")
        axes[0].set_xlabel("Detection Probability")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        leak_keys = [k for k, v in results_by_type.items() if v["label"] == 1]
        det_rates = [results_by_type[k]["detection_rate"] for k in leak_keys]
        colors    = ["crimson" if r >= 0.5 else "orange" for r in det_rates]
        axes[1].barh(leak_keys, det_rates, color=colors, alpha=0.7)
        axes[1].axvline(0.5, color="black", linestyle="--")
        axes[1].set_xlabel("Detection Rate")
        axes[1].set_title("Detection Rate by Leak Type and Topology")
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        plot_path = PLOTS_DIR / "mendeley_accelerometer_results.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {plot_path}")

        # ── Save JSON ────────────────────────────────────────────────────────
        results = {
            "model":       "C_seed42_accelerometer_eval",
            "sensor_type": "accelerometer",
            "fs_model":    FS_MODEL,
            "fs_detected": "auto from CSV timestamps",
            "note":        "Correct sensor modality — A1/A2 CSV files, fs auto-detected",
            "overall": {
                "auroc":     round(float(auroc), 4),
                "f1":        round(float(f1),    4),
                "acc":       round(float(acc),   4),
                "n_total":   int(len(all_probs)),
                "n_leak":    int(all_labels.sum()),
                "n_no_leak": int((all_labels == 0).sum()),
            },
            "by_topology": results_by_topology,
            "by_type":     results_by_type,
        }
        json_path = RESULTS_DIR / "mendeley_accelerometer_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {json_path}")

        # ── Comparison summary ───────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*55}")
        print(f"  Original Mendeley AUROC (hydrophone)    : "
              f"{OLD_HYDROPHONE_AUROC:.4f}")
        print(f"  Accelerometer AUROC (correct modality)  : "
              f"{auroc:.4f}")
        delta = auroc - OLD_HYDROPHONE_AUROC
        print(f"  Delta                                   : "
              f"{delta:+.4f}")
        print(f"{'='*55}")

        if auroc >= 0.80:
            print(f"\n  RESULT: Model works well on real accelerometer data")
            print(f"  CONCLUSION: Gap was sensor modality mismatch")
        elif auroc >= 0.60:
            print(f"\n  RESULT: Partial improvement")
            print(f"  CONCLUSION: Modality + flow scale both contribute")
        elif auroc >= 0.50:
            print(f"\n  RESULT: Slight improvement")
            print(f"  CONCLUSION: Flow scale mismatch is dominant")
        else:
            print(f"\n  RESULT: No improvement — check CSV DEBUG output above")
            print(f"  CONCLUSION: Data loading issue likely")

    else:
        print(f"  ERROR: Could not compute AUROC")
        print(f"  all_probs length  : {len(all_probs)}")
        print(f"  all_labels unique : {set(all_labels)}")

    return auroc


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Mendeley root: {MENDELEY_ROOT.resolve()}")
    print(f"Checkpoint   : {CKPT_NAME}")

    ckpt_path = MODELS_DIR / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = AcousticLeakNet(
        signal_length=SIGNAL_LENGTH,
        n_scalars=11,
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} | val AUROC={ckpt['val_auroc']:.4f}")

    evaluate(model, device)


if __name__ == "__main__":
    main()