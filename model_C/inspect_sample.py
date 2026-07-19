"""
inspect_sample.py — Interactive single-sample inference for Model C
====================================================================
Run from Model_C directory:
    python inspect_sample.py

Shows you exactly what the model predicts for individual samples.
You can see the signal, the true label, and the model's prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ── Load model ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import AcousticLeakNet

MODELS_DIR = Path("../models")
CACHE_ROOT = Path("../cache_c")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(str(MODELS_DIR / "best_model_c_v4.pt"),
                  map_location=device, weights_only=False)
model = AcousticLeakNet(signal_length=2000, n_scalars=11,
                        base_channels=64, dropout=0.3).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded Model C epoch {ckpt['epoch']}")
print(f"Val metrics: AUROC={ckpt['val_metrics']['auroc']:.4f} "
      f"SevR2={ckpt['val_metrics']['sev_r2']:.4f} "
      f"PosMAE={ckpt['val_metrics']['pos_mae']:.4f}")

# ── Load test caches ───────────────────────────────────────────────────────────
NETWORKS = {
    "1": ("L-TOWN",   "test_network_3"),
    "2": ("KY15",     "test_network_6"),
    "3": ("Richmond", "test_network_8"),
}

print("\nAvailable networks:")
for k, (name, _) in NETWORKS.items():
    print(f"  {k}: {name}")

MEAN_PIPE_LENGTH = {
    "test_network_3": 47.8,
    "test_network_6": 681.9,
    "test_network_8": 515.6,
}

def predict_sample(signal, labels_row):
    """Run model on one sample and return predictions."""
    sig_tensor = torch.tensor(signal[None], dtype=torch.float32).to(device)
    scalars    = torch.zeros(1, 11, device=device)
    with torch.no_grad():
        det, pos, sev = model(sig_tensor, scalars)
        prob     = torch.sigmoid(det).item()
        pos_pred = pos.item()
        sev_pred = sev.item()
    return prob, pos_pred, sev_pred

def describe_sample(labels_row, split):
    true_label  = int(labels_row[0])
    true_pos    = labels_row[1]
    true_flow   = labels_row[2]
    pos_valid   = int(labels_row[3])
    return true_label, true_pos, true_flow, pos_valid

def run_inspection(split_name, network_name):
    signals = np.load(str(CACHE_ROOT / split_name / "signals.npy"), mmap_mode="r")
    labels  = np.load(str(CACHE_ROOT / split_name / "labels.npy"),  mmap_mode="r")
    mean_pipe_len = MEAN_PIPE_LENGTH[split_name]

    leak_idx   = np.where(labels[:, 0] == 1)[0]
    noleak_idx = np.where(labels[:, 0] == 0)[0]
    dual_idx   = np.where((labels[:, 0] == 1) & (labels[:, 3] == 1))[0]

    print(f"\n{'='*60}")
    print(f"  {network_name} — {len(signals):,} total samples")
    print(f"  Leak: {len(leak_idx):,} | No-leak: {len(noleak_idx):,} | Dual-sensor leak: {len(dual_idx):,}")
    print(f"{'='*60}")

    correct = 0
    total   = 0

    while True:
        print("\nOptions:")
        print("  l  — random LEAK sample")
        print("  n  — random NO-LEAK sample")
        print("  d  — random DUAL-SENSOR leak (localisation + severity)")
        print("  b  — run BATCH of 20 random samples and show accuracy")
        print("  q  — quit / change network")
        print("  NUMBER — specific sample index")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            break
        elif choice == "l":
            idx = int(random.choice(leak_idx))
        elif choice == "n":
            idx = int(random.choice(noleak_idx))
        elif choice == "d":
            if len(dual_idx) == 0:
                print("  No dual-sensor samples in this network.")
                continue
            idx = int(random.choice(dual_idx))
        elif choice == "b":
            # Batch test
            sample_indices = np.concatenate([
                np.random.choice(leak_idx,   min(10, len(leak_idx)),   replace=False),
                np.random.choice(noleak_idx, min(10, len(noleak_idx)), replace=False),
            ])
            np.random.shuffle(sample_indices)
            batch_correct = 0
            print(f"\n  {'Idx':>7}  {'True':>8}  {'Pred Prob':>10}  {'Decision':>10}  {'Correct':>8}")
            print(f"  {'─'*55}")
            for i in sample_indices:
                sig = signals[i].copy().astype(np.float32)
                lab = labels[i]
                prob, _, _ = predict_sample(sig, lab)
                true_label = int(lab[0])
                decision   = 1 if prob >= 0.5 else 0
                correct_flag = "✓" if decision == true_label else "✗"
                if decision == true_label:
                    batch_correct += 1
                true_str = "LEAK" if true_label == 1 else "NO-LEAK"
                dec_str  = "LEAK" if decision   == 1 else "NO-LEAK"
                print(f"  {i:>7}  {true_str:>8}  {prob:>10.4f}  {dec_str:>10}  {correct_flag:>8}")
            print(f"\n  Batch accuracy: {batch_correct}/{len(sample_indices)} = {batch_correct/len(sample_indices)*100:.1f}%")
            continue
        else:
            try:
                idx = int(choice)
                if idx < 0 or idx >= len(signals):
                    print(f"  Index out of range (0 to {len(signals)-1})")
                    continue
            except ValueError:
                print("  Invalid input.")
                continue

        # ── Run inference on selected sample ──────────────────────────────────
        sig = signals[idx].copy().astype(np.float32)
        lab = labels[idx]
        true_label, true_pos, true_flow, pos_valid = describe_sample(lab, split_name)
        prob, pos_pred, sev_pred = predict_sample(sig, lab)

        decision = 1 if prob >= 0.5 else 0
        correct_flag = "✓ CORRECT" if decision == true_label else "✗ WRONG"

        print(f"\n  {'─'*55}")
        print(f"  Sample index     : {idx}")
        print(f"  True label       : {'LEAK' if true_label == 1 else 'NO-LEAK'}")
        print(f"  Detection prob   : {prob:.4f}")
        print(f"  Decision (≥0.5)  : {'LEAK' if decision == 1 else 'NO-LEAK'}  {correct_flag}")

        if true_label == 1:
            print(f"\n  ── Leak details ──────────────────────────────────")
            print(f"  True flow rate   : {true_flow:.4f} L/s")
            print(f"  Predicted flow   : {sev_pred:.4f} L/s")
            sev_err = abs(sev_pred - true_flow)
            print(f"  Severity error   : {sev_err:.4f} L/s ({sev_err/max(true_flow,1e-6)*100:.1f}%)")

            if pos_valid == 1:
                print(f"\n  True position    : {true_pos:.4f} (fraction of pipe)")
                print(f"  Predicted pos    : {pos_pred:.4f} (fraction of pipe)")
                pos_err_norm = abs(pos_pred - true_pos)
                pos_err_m    = pos_err_norm * mean_pipe_len
                print(f"  Position error   : {pos_err_norm:.4f} = {pos_err_m:.2f}m physical")
            else:
                print(f"\n  Single-sensor pipe — localisation not evaluated")

        # ── Plot signal ────────────────────────────────────────────────────────
        plot = input("\n  Plot signal? (y/n): ").strip().lower()
        if plot == "y":
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            t = np.arange(2000) / 5000.0 * 1000  # ms
            axes[0].plot(t, sig[0], color="steelblue", linewidth=0.8)
            axes[0].set_ylabel("Sensor 1 (Left)")
            axes[0].set_title(
                f"{network_name} | Sample {idx} | "
                f"True: {'LEAK' if true_label==1 else 'NO-LEAK'} | "
                f"Pred: {prob:.3f} | {correct_flag}")
            axes[1].plot(t, sig[1], color="coral", linewidth=0.8)
            axes[1].set_ylabel("Sensor 2 (Right)")
            axes[1].set_xlabel("Time (ms)")
            plt.tight_layout()
            plt.savefig(f"../plots/inspect_{network_name}_{idx}.png", dpi=150)
            plt.show()
            print(f"  Saved: ../plots/inspect_{network_name}_{idx}.png")

        total   += 1
        correct += (decision == true_label)
        if total > 0:
            print(f"\n  Session accuracy: {correct}/{total} = {correct/total*100:.1f}%")


# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    print("\n" + "="*60)
    print("  Select network:")
    for k, (name, _) in NETWORKS.items():
        print(f"    {k}: {name}")
    print("    q: quit")

    net_choice = input("\nNetwork: ").strip().lower()
    if net_choice == "q":
        print("Done.")
        break
    if net_choice not in NETWORKS:
        print("Invalid choice.")
        continue

    network_name, split_name = NETWORKS[net_choice]
    run_inspection(split_name, network_name)