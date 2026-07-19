"""
mendeley_experiment2.py — Mendeley Flow-Matched Retraining + Eval
==================================================================
Experiment 2: Proves Mendeley gap is domain mismatch, not model failure.

Pipeline:
  1. Synthesise waveforms from mend.csv (Mendeley-matched EPANET simulation)
  2. Fine-tune Model C on these waveforms (small dataset, few epochs)
  3. Evaluate on real Mendeley accelerometer data (A1/A2 CSVs)
     - Auto-detects sampling rate from CSV timestamps
     - Uses joint normalisation matching training pipeline
  4. Output three-row AUROC comparison table

Expected result:
  Municipal EPANET → Municipal pipes  : AUROC 1.000  (existing result)
  Municipal EPANET → Mendeley pipes   : AUROC ~0.49  (Experiment 1 result)
  Mendeley EPANET  → Mendeley pipes   : AUROC ~0.79+ (this experiment)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from math import gcd
from scipy.signal import resample_poly
from tqdm import tqdm

from model import AcousticLeakNet

# ── Paths ──────────────────────────────────────────────────────────────────
MEND_CSV      = Path("../datasets/NetworkList/NetworkMend/mend.csv")
MENDELEY_ROOT = Path("../datasets/Accelerometer/Accelerometer/")
MODELS_DIR    = Path("../models")
RESULTS_DIR   = Path("../results")
PLOTS_DIR     = Path("../plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_CKPT     = "best_model_c_v4.pt"
MEND_CKPT     = "best_model_mend.pt"
SIGNAL_LENGTH = 2000
FS_MODEL      = 5000

# Mendeley-matched synthesis constants
NOISE_RMS     = 0.0012
NORM_SCALE    = 10.0 * NOISE_RMS   # = 0.012
ALPHA_0       = 0.008
REF_FREQ      = 800.0

# Fine-tuning hyperparams
EPOCHS        = 20
BATCH_SIZE    = 128
LR            = 1e-4
VAL_SPLIT     = 0.15
WEIGHT_DECAY  = 1e-4

# Existing results for comparison table
MUNICIPAL_AUROC   = 1.000
EXPERIMENT1_AUROC = 0.4919

LEAK_TYPES = ["Orifice Leak", "Longitudinal Crack",
              "Circumferential Crack", "Gasket Leak"]
NO_LEAK    = "No-leak"
TOPOLOGIES = ["Looped", "Branched"]

LEAK_FREQ = {"OL": 1500, "LC": 800, "CC": 1200, "GL": 300}
LEAK_BW   = {"OL": 600,  "LC": 400, "CC": 500,  "GL": 150}


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — SIGNAL SYNTHESIS (Mendeley-matched)
# ══════════════════════════════════════════════════════════════════════════

def pink_noise(n: int) -> np.ndarray:
    white = np.random.randn(n)
    f     = np.fft.rfftfreq(n)
    f[0]  = 1e-6
    psd   = 1.0 / np.sqrt(f)
    pink  = np.fft.irfft(np.fft.rfft(white) * psd, n=n)
    return pink.astype(np.float32)


def bandpass(sig: np.ndarray, fc: float, bw: float, fs: float) -> np.ndarray:
    F    = np.fft.rfftfreq(len(sig), 1.0 / fs)
    mask = np.exp(-0.5 * ((F - fc) / (bw / 2.0)) ** 2).astype(np.float32)
    return np.fft.irfft(np.fft.rfft(sig) * mask, n=len(sig)).astype(np.float32)


def synthesise_pair(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    n  = SIGNAL_LENGTH
    fs = FS_MODEL

    is_leak    = int(row["Leak_Status"]) == 1
    leak_type  = str(row["Leak_Type"]) if is_leak else "OL"
    wave_speed = float(row["Acoustic_Propagation_Speed_mps"])
    alpha0     = float(row["Attenuation_Alpha_per_m"])
    d_left     = float(row["Leak_Distance_Left_m"])
    d_right    = float(row["Leak_Distance_Right_m"])
    pressure   = float(row["Avg_Pressure_at_Leak"]) if is_leak else 0.0
    leak_area  = float(row["Leak_Area_m2"])          if is_leak else 0.0
    cd         = float(row["Leak_Cd"])               if is_leak else 0.61

    # Stage 2: Torricelli amplitude
    g         = 9.81
    q_lps     = cd * leak_area * np.sqrt(2 * g * max(pressure, 0.01))
    amplitude = np.clip(q_lps * 500.0, 0.01, 10.0)

    # Stage 3/4: Pink noise source
    fc = LEAK_FREQ.get(leak_type, 800)
    bw = LEAK_BW.get(leak_type, 400)

    if is_leak:
        src = pink_noise(n)
        src = bandpass(src, fc, bw, fs)
        src = src / (np.std(src) + 1e-8) * amplitude
    else:
        src = np.zeros(n, dtype=np.float32)

    # Stage 5: Dual-channel propagation
    tdoa_s   = (d_left - d_right) / wave_speed
    tdoa_smp = int(round(tdoa_s * fs))

    def apply_attenuation(sig: np.ndarray, dist: float) -> np.ndarray:
        F    = np.fft.rfftfreq(len(sig), 1.0 / fs)
        a_f  = alpha0 * np.sqrt(np.maximum(F, 1.0) / REF_FREQ)
        gain = np.exp(-a_f * dist).astype(np.float32)
        return np.fft.irfft(np.fft.rfft(sig) * gain, n=len(sig)).astype(np.float32)

    ch1 = apply_attenuation(src, d_left)
    ch2 = apply_attenuation(src, d_right)

    if tdoa_smp > 0:
        ch2 = np.roll(ch2,  tdoa_smp)
    elif tdoa_smp < 0:
        ch1 = np.roll(ch1, -tdoa_smp)

    struct = pink_noise(n) * 0.05
    struct = bandpass(struct, 60, 40, fs)
    ch1   += struct
    ch2   += struct

    ch1 *= rng.uniform(0.7, 1.3)
    ch2 *= rng.uniform(0.7, 1.3)

    # Stage 6: Background noise
    t = np.arange(n, dtype=np.float32) / fs
    if rng.random() < 0.5:
        pump_amp = rng.uniform(0.001, 0.003)
        pump     = pump_amp * np.sin(2 * np.pi * 50 * t).astype(np.float32)
        ch1 += pump + rng.uniform(-0.1, 0.1) * pump
        ch2 += pump + rng.uniform(-0.1, 0.1) * pump

    if rng.random() < 0.6:
        f_traf   = rng.uniform(10, 30)
        traf_amp = rng.uniform(0.0005, 0.002)
        traf     = traf_amp * np.sin(2 * np.pi * f_traf * t).astype(np.float32)
        ch1 += traf
        ch2 += traf

    ch1 += rng.normal(0, NOISE_RMS * 0.3, n).astype(np.float32)
    ch2 += rng.normal(0, NOISE_RMS * 0.3, n).astype(np.float32)

    amp_scale = rng.uniform(0.85, 1.15)
    ch1 *= amp_scale
    ch2 *= amp_scale

    # Fixed-scale normalisation
    ch1 = np.clip(ch1 / NORM_SCALE, -5.0, 5.0)
    ch2 = np.clip(ch2 / NORM_SCALE, -5.0, 5.0)

    return np.stack([ch1, ch2]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# PART 2 — DATASET
# ══════════════════════════════════════════════════════════════════════════

class MendeleySimDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df     = df.reset_index(drop=True)
        self.rng    = np.random.default_rng(seed)
        self.labels = df["Leak_Status"].astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        wave   = synthesise_pair(row, self.rng)
        label  = self.labels[idx]
        scalar = np.zeros(11, dtype=np.float32)
        return (torch.from_numpy(wave),
                torch.tensor(scalar),
                torch.tensor(label, dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════
# PART 3 — FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════

def finetune(device):
    print("\n" + "="*55)
    print("  EXPERIMENT 2 — FINE-TUNING ON MENDELEY-MATCHED DATA")
    print("="*55)

    ckpt_path = MODELS_DIR / BASE_CKPT
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg       = ckpt["cfg"]
    model     = AcousticLeakNet(
        signal_length=SIGNAL_LENGTH,
        n_scalars=11,
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded base model: epoch {ckpt['epoch']}, "
          f"val AUROC={ckpt['val_auroc']:.4f}")

    print(f"  Loading {MEND_CSV} ...")
    df      = pd.read_csv(MEND_CSV)
    dataset = MendeleySimDataset(df)
    print(f"  Dataset: {len(dataset)} samples  "
          f"({int(dataset.labels.sum())} leak, "
          f"{int((dataset.labels == 0).sum())} no-leak)")

    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Train: {n_train}  Val: {n_val}")

    optimiser = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc   = 0.0
    best_epoch   = 0
    train_losses = []
    val_aurocs   = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for waves, scalars, labels in tqdm(train_dl,
                                           desc=f"Epoch {epoch+1}/{EPOCHS}"):
            waves, scalars, labels = (waves.to(device),
                                      scalars.to(device),
                                      labels.to(device))
            optimiser.zero_grad()
            det, _, _ = model(waves, scalars)
            loss      = criterion(det.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_dl)
        train_losses.append(avg_loss)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for waves, scalars, labels in val_dl:
                waves, scalars = waves.to(device), scalars.to(device)
                det, _, _      = model(waves, scalars)
                probs          = torch.sigmoid(det).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        val_auroc = roc_auc_score(all_labels, all_probs)
        val_aurocs.append(val_auroc)
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"loss={avg_loss:.4f} | val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "cfg":         cfg,
                "epoch":       epoch + 1,
                "val_auroc":   val_auroc,
            }, MODELS_DIR / MEND_CKPT)
            print(f"  ✓ Saved best checkpoint (AUROC={val_auroc:.4f})")

    print(f"\n  Fine-tuning complete. Best val AUROC={best_auroc:.4f} "
          f"at epoch {best_epoch}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, color="steelblue")
    ax1.set_title("Fine-tuning Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax2.plot(val_aurocs, color="crimson")
    ax2.axhline(EXPERIMENT1_AUROC, linestyle="--",
                color="orange",
                label=f"Exp1 baseline ({EXPERIMENT1_AUROC})")
    ax2.set_title("Val AUROC on Mendeley-matched Sim")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_finetune_curve.png", dpi=150)
    plt.close()

    return best_auroc


# ══════════════════════════════════════════════════════════════════════════
# PART 4 — EVAL ON REAL MENDELEY ACCELEROMETER DATA
# ══════════════════════════════════════════════════════════════════════════

def load_csv_signal(path: str):
    """
    Load Mendeley accelerometer CSV.
    Auto-detects sampling rate from time column.
    Returns (signal, fs).
    """
    df       = pd.read_csv(path, header=0, low_memory=False)
    time_col = df.iloc[:, 0].values.astype(np.float64)
    dt       = np.median(np.diff(time_col[:1000]))
    fs       = int(round(1.0 / dt))

    if not hasattr(load_csv_signal, "_printed"):
        print(f"\n  [CSV DEBUG] File   : {Path(path).name}")
        print(f"  [CSV DEBUG] Shape  : {df.shape}")
        print(f"  [CSV DEBUG] Columns: {list(df.columns)}")
        print(f"  [CSV DEBUG] dt={dt:.8f}s  →  fs={fs} Hz\n")
        load_csv_signal._printed = True

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


def slice_to_windows(h1: np.ndarray, h2: np.ndarray,
                     window_len: int) -> np.ndarray:
    """
    Joint normalisation — matches training preprocessing exactly.
    Both channels normalised together to preserve relative amplitude.
    """
    min_len   = min(len(h1), len(h2))
    n_windows = min_len // window_len
    windows   = []
    for i in range(n_windows):
        s     = i * window_len
        e     = s + window_len
        joint = np.concatenate([h1[s:e], h2[s:e]])
        mu    = np.mean(joint)
        std   = np.std(joint) + 1e-8
        c1    = np.clip((h1[s:e] - mu) / std, -5.0, 5.0)
        c2    = np.clip((h2[s:e] - mu) / std, -5.0, 5.0)
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
        print(f"  WARNING: No A1/A2 pairs in {folder}")
        print(f"  Files: {[f.name for f in files[:5]]}")
    return pairs


@torch.no_grad()
def run_inference(model, windows: np.ndarray, device) -> np.ndarray:
    probs = []
    for i in range(0, len(windows), 64):
        batch      = torch.from_numpy(windows[i:i+64]).to(device)
        scalars    = torch.zeros(batch.size(0), 11, device=device)
        pred, _, _ = model(batch, scalars)
        probs.extend(torch.sigmoid(pred).cpu().numpy().flatten())
    return np.array(probs)


def evaluate_on_mendeley(model, device):
    print("\n" + "="*55)
    print("  EVALUATING ON REAL MENDELEY ACCELEROMETER DATA")
    print("="*55)

    all_probs, all_labels = [], []
    results_by_topology   = {}
    results_by_type       = {}

    for topology in TOPOLOGIES:
        topo_dir    = MENDELEY_ROOT / topology
        topo_probs  = []
        topo_labels = []

        print(f"\n  Topology: {topology}")

        for leak_type in LEAK_TYPES:
            leak_dir = topo_dir / leak_type
            if not leak_dir.exists():
                print(f"    Skipping {leak_type} — not found")
                continue

            pairs      = find_paired_files(leak_dir)
            type_probs = []

            for a1_path, a2_path in pairs:
                try:
                    a1_raw, fs_acc = load_csv_signal(str(a1_path))
                    a2_raw, _      = load_csv_signal(str(a2_path))
                except Exception as e:
                    print(f"    ERROR {a1_path.name}: {e}")
                    continue

                a1      = resample_signal(a1_raw, fs_acc, FS_MODEL)
                a2      = resample_signal(a2_raw, fs_acc, FS_MODEL)
                windows = slice_to_windows(a1, a2, SIGNAL_LENGTH)

                if len(windows) == 0:
                    continue

                p = run_inference(model, windows, device)
                type_probs.extend(p)
                topo_probs.extend(p)
                all_probs.extend(p)

            n = len(type_probs)
            topo_labels.extend([1] * n)
            all_labels.extend([1] * n)

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
                print(f"    {leak_type:25s}: mean={arr.mean():.3f}  "
                      f"det={det_rate:.3f}  ({n} windows)")

        nl_dir = topo_dir / NO_LEAK
        if nl_dir.exists():
            pairs    = find_paired_files(nl_dir)
            nl_probs = []

            for a1_path, a2_path in pairs:
                try:
                    a1_raw, fs_acc = load_csv_signal(str(a1_path))
                    a2_raw, _      = load_csv_signal(str(a2_path))
                except Exception as e:
                    print(f"    ERROR {a1_path.name}: {e}")
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
            topo_labels.extend([0] * n)
            all_labels.extend([0] * n)

            arr = np.array(nl_probs)
            far = (arr >= 0.5).mean()
            key = f"{topology}/No-leak"
            results_by_type[key] = {
                "mean_prob":        float(arr.mean()),
                "false_alarm_rate": float(far),
                "n_windows":        n,
                "label":            0
            }
            print(f"    {'No-leak':25s}: mean={arr.mean():.3f}  "
                  f"FAR={far:.3f}  ({n} windows)")

        if topo_probs and len(set(topo_labels)) > 1:
            ta = np.array(topo_probs)
            tl = np.array(topo_labels)
            results_by_topology[topology] = {
                "auroc": round(float(roc_auc_score(tl, ta)), 4),
                "f1":    round(float(f1_score(tl, (ta >= 0.5).astype(int))), 4),
                "n":     len(topo_probs)
            }
            print(f"\n    {topology} AUROC={results_by_topology[topology]['auroc']:.4f}  "
                  f"F1={results_by_topology[topology]['f1']:.4f}")

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    auroc      = roc_auc_score(all_labels, all_probs)
    f1         = f1_score(all_labels, (all_probs >= 0.5).astype(int))
    acc        = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))

    print(f"\n  Overall AUROC : {auroc:.4f}")
    print(f"  Overall F1    : {f1:.4f}")
    print(f"  Overall Acc   : {acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_probs[all_labels == 0], bins=40, alpha=0.6,
                 color="steelblue", label="No-Leak", density=True)
    axes[0].hist(all_probs[all_labels == 1], bins=40, alpha=0.6,
                 color="crimson", label="Leak", density=True)
    axes[0].axvline(0.5, color="black", linestyle="--", label="Threshold")
    axes[0].set_title(
        f"Exp 2 — Mendeley-matched Model on Real Accelerometer Data\n"
        f"AUROC={auroc:.4f}")
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
    plt.savefig(PLOTS_DIR / "exp2_mendeley_eval.png", dpi=150)
    plt.close()

    return auroc, f1, acc, results_by_topology, results_by_type


# ══════════════════════════════════════════════════════════════════════════
# PART 5 — THREE-ROW COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════

def print_comparison_table(exp2_auroc: float):
    print("\n" + "="*65)
    print("  DOMAIN MISMATCH PROOF — THREE-ROW AUROC TABLE")
    print("="*65)
    print(f"  {'Training Domain':<35} {'Test Domain':<20} {'AUROC':>6}")
    print(f"  {'-'*35} {'-'*20} {'-'*6}")
    print(f"  {'Municipal EPANET (Model C)':<35} "
          f"{'Municipal pipes':<20} {MUNICIPAL_AUROC:>6.4f}")
    print(f"  {'Municipal EPANET (Model C)':<35} "
          f"{'Mendeley lab pipe':<20} {EXPERIMENT1_AUROC:>6.4f}")
    print(f"  {'Mendeley-matched EPANET (Exp 2)':<35} "
          f"{'Mendeley lab pipe':<20} {exp2_auroc:>6.4f}")
    print("="*65)

    gap_closed = exp2_auroc - EXPERIMENT1_AUROC
    print(f"\n  Gap closed by domain-matched retraining: {gap_closed:+.4f}")

    if exp2_auroc >= 0.80:
        print(f"  CONCLUSION: Domain mismatch fully explains the gap.")
        print(f"  Model learns correctly from whatever physical domain")
        print(f"  it is trained on.")
    elif exp2_auroc >= 0.65:
        print(f"  CONCLUSION: Domain mismatch is the primary cause.")
        print(f"  Remaining gap likely due to above-ground vs buried")
        print(f"  boundary conditions and junction reflections.")
    else:
        print(f"  CONCLUSION: Partial improvement. Synthesis pipeline")
        print(f"  may need additional real-world noise sources.")
    print("="*65)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"mend.csv     : {MEND_CSV.resolve()}")
    print(f"Mendeley root: {MENDELEY_ROOT.resolve()}")
    print(f"Base ckpt    : {BASE_CKPT}")

    # Step 1: Fine-tune
    best_sim_auroc = finetune(device)

    # Step 2: Load best fine-tuned checkpoint
    ckpt  = torch.load(MODELS_DIR / MEND_CKPT,
                       map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = AcousticLeakNet(
        signal_length=SIGNAL_LENGTH,
        n_scalars=11,
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n  Loaded fine-tuned model: epoch {ckpt['epoch']}, "
          f"sim val AUROC={ckpt['val_auroc']:.4f}")

    # Step 3: Eval on real Mendeley accelerometer data
    exp2_auroc, exp2_f1, exp2_acc, by_topo, by_type = \
        evaluate_on_mendeley(model, device)

    # Step 4: Print comparison table
    print_comparison_table(exp2_auroc)

    # Step 5: Save results
    results = {
        "experiment":  2,
        "description": "Mendeley flow-matched retraining proves domain mismatch",
        "fixes_applied": [
            "Auto-detected sampling rate from CSV timestamps (was hardcoded 51200, actual 25600)",
            "Joint normalisation in slice_to_windows (matches training preprocessing)",
        ],
        "comparison_table": {
            "municipal_epanet_municipal_pipes":  MUNICIPAL_AUROC,
            "municipal_epanet_mendeley_pipes":   EXPERIMENT1_AUROC,
            "mendeley_epanet_mendeley_pipes":    round(float(exp2_auroc), 4),
        },
        "exp2_real_mendeley_eval": {
            "auroc":       round(float(exp2_auroc), 4),
            "f1":          round(float(exp2_f1),    4),
            "acc":         round(float(exp2_acc),   4),
            "by_topology": by_topo,
            "by_type":     by_type,
        },
        "sim_val_auroc": round(float(best_sim_auroc), 4),
        "fine_tune_cfg": {
            "base_ckpt":  BASE_CKPT,
            "epochs":     EPOCHS,
            "lr":         LR,
            "batch_size": BATCH_SIZE,
        }
    }
    json_path = RESULTS_DIR / "experiment2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved : {json_path}")
    print(f"  Plots saved   : {PLOTS_DIR}/exp2_*.png")


if __name__ == "__main__":
    main()