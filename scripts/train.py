"""
train_c.py — AcousticLeakNet Model C Training
==============================================
Reads from ../cache_c/
Saves best_model_c_v3.pt

Run after: python pregenerate_c.py --split train
           python pregenerate_c.py --split val
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import AcousticLeakNet, UncertaintyLoss, count_parameters

CACHE_ROOT  = Path("../cache_c")
MODELS_DIR  = Path("../models")
PLOTS_DIR   = Path("../plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_NAME = "best_model_c_v3.pt"

CFG = {
    "batch_size":    256,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "n_epochs":      25,
    "base_channels": 64,
    "dropout":       0.3,
    "seed":          42,
}


class CachedDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        cache_dir    = CACHE_ROOT / split
        self.signals = np.load(str(cache_dir / "signals.npy"), mmap_mode="r")
        self.labels  = np.load(str(cache_dir / "labels.npy"),  mmap_mode="r")
        self.augment = augment
        print(f"CachedDataset [{split}]: {len(self.signals):,} samples | augment={augment}")

    def __len__(self):
        return len(self.signals)

    def _augment(self, sig):
        sig = sig * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.25:
            sig += np.random.normal(0, np.random.uniform(0.01, 0.05), sig.shape)
        return sig

    def __getitem__(self, idx):
        sig = self.signals[idx].copy().astype(np.float32)
        lab = self.labels[idx]
        if self.augment:
            sig = self._augment(sig)
        return (torch.from_numpy(sig),
                torch.tensor(lab[0], dtype=torch.float32),   # leak_status
                torch.tensor(lab[1], dtype=torch.float32),   # leak_pos
                torch.tensor(lab[2], dtype=torch.float32),   # leak_flow (severity)
                torch.tensor(lab[3], dtype=torch.float32))   # pos_valid


def make_loader(split, batch_size, shuffle, augment=False):
    ds = CachedDataset(split, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def safe_r2(y_true: list, y_pred: list) -> float:
    """
    R² with safety checks.
    Returns nan if fewer than 2 samples or all true values identical.
    Prevents numerical explosion when std(y_true) ~ 0.
    """
    if len(y_true) < 2:
        return float("nan")
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    if yt.std() < 1e-6:
        return float("nan")
    return float(r2_score(yt, yp))


def train_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs):
    model.train()
    losses, det_l, pos_l, sev_l = [], [], [], []
    pbar = tqdm(loader, desc=f"Ep {epoch:3d}/{n_epochs} [train]",
                unit="batch", leave=False, dynamic_ncols=True)

    for sig, true_det, true_pos, true_sev, pos_valid in pbar:
        sig       = sig.to(device)
        true_det  = true_det.to(device)
        true_pos  = true_pos.to(device)
        true_sev  = true_sev.to(device)
        pos_valid = pos_valid.to(device)
        scalars   = torch.zeros(sig.size(0), 11, device=device)

        optimizer.zero_grad()
        pred_det, pred_pos, pred_sev = model(sig, scalars)
        loss, l_d, l_p, l_s = criterion(pred_det, pred_pos, pred_sev,
                                         true_det, true_pos, true_sev, pos_valid)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()), 1.0)
        optimizer.step()

        losses.append(loss.item())
        det_l.append(l_d); pos_l.append(l_p); sev_l.append(l_s)
        pbar.set_postfix(loss=f"{loss.item():.4f}", det=f"{l_d:.3f}")

    pbar.close()
    return {"loss": np.mean(losses), "det": np.mean(det_l),
            "pos":  np.mean(pos_l),  "sev": np.mean(sev_l)}


@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch, n_epochs):
    model.eval()
    losses = []
    all_prob, all_true   = [], []
    all_pos_p, all_pos_t = [], []
    all_sev_p, all_sev_t = [], []

    pbar = tqdm(loader, desc=f"Ep {epoch:3d}/{n_epochs} [val]  ",
                unit="batch", leave=False, dynamic_ncols=True)

    for sig, true_det, true_pos, true_sev, pos_valid in pbar:
        sig         = sig.to(device)
        scalars     = torch.zeros(sig.size(0), 11, device=device)
        true_det_d  = true_det.to(device)
        true_pos_d  = true_pos.to(device)
        true_sev_d  = true_sev.to(device)
        pos_valid_d = pos_valid.to(device)

        pred_det, pred_pos, pred_sev = model(sig, scalars)
        loss, *_ = criterion(pred_det, pred_pos, pred_sev,
                             true_det_d, true_pos_d, true_sev_d, pos_valid_d)
        losses.append(loss.item())

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

    auroc   = roc_auc_score(all_true, all_prob)
    f1      = f1_score(all_true, (np.array(all_prob) >= 0.5).astype(int))
    pos_mae = float(mean_absolute_error(all_pos_t, all_pos_p)) if all_pos_t else float("nan")
    sev_r2  = safe_r2(all_sev_t, all_sev_p)

    # Epoch 1 diagnostic — confirms severity evaluation is working
    if epoch == 1:
        sev_arr = np.array(all_sev_t)
        print(f"  [diag] sev_samples={len(all_sev_t)} "
              f"pos_samples={len(all_pos_t)} "
              f"sev_true_std={sev_arr.std():.4f} "
              f"sev_pred_std={np.std(all_sev_p):.4f}")

    return {"loss": np.mean(losses), "auroc": auroc, "f1": f1,
            "pos_mae": pos_mae, "sev_r2": sev_r2}


def train():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_loader = make_loader("train", CFG["batch_size"], shuffle=True,  augment=True)
    val_loader   = make_loader("val",   CFG["batch_size"], shuffle=False, augment=False)

    model     = AcousticLeakNet(signal_length=2000, n_scalars=11,
                                base_channels=CFG["base_channels"],
                                dropout=CFG["dropout"]).to(device)
    criterion = UncertaintyLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["n_epochs"], eta_min=1e-6)

    print(f"Model parameters: {count_parameters(model):,}")

    history     = {"train": [], "val": []}
    best_auroc  = 0.0
    start_epoch = 1

    ckpt_path = MODELS_DIR / CKPT_NAME
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        criterion.load_state_dict(ckpt["criterion_state"])
        best_auroc  = ckpt["val_auroc"]
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"  Resumed epoch {ckpt['epoch']} | best AUROC={best_auroc:.4f}")

    for epoch in range(start_epoch, CFG["n_epochs"] + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer,
                         device, epoch, CFG["n_epochs"])
        vl = val_epoch(model, val_loader, criterion,
                       device, epoch, CFG["n_epochs"])
        scheduler.step()
        history["train"].append(tr)
        history["val"].append(vl)

        sev_str = f"{vl['sev_r2']:.4f}" if not np.isnan(vl['sev_r2']) else "  nan"
        pos_str = f"{vl['pos_mae']:.4f}" if not np.isnan(vl['pos_mae']) else "  nan"

        print(f"Ep {epoch:3d}/{CFG['n_epochs']} | "
              f"train={tr['loss']:.4f} "
              f"(det={tr['det']:.3f} pos={tr['pos']:.3f} sev={tr['sev']:.3f}) | "
              f"val AUROC={vl['auroc']:.4f} F1={vl['f1']:.4f} "
              f"PosMAE={pos_str} SevR2={sev_str}")

        if vl["auroc"] > best_auroc:
            best_auroc = vl["auroc"]
            torch.save({
                "model_state":     model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "cfg":             CFG,
                "epoch":           epoch,
                "val_auroc":       vl["auroc"],
                "val_metrics":     vl,
                "model_variant":   "C",
                "changes":         ["SNR range calibration",
                                    "Pink noise source",
                                    "Correlated channel leak signal",
                                    "Joint normalisation"],
            }, MODELS_DIR / CKPT_NAME)
            print(f"  Saved best Model C (AUROC={best_auroc:.4f})")

    ep = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(ep, [h["loss"]   for h in history["train"]], label="Train")
    axes[0].plot(ep, [h["loss"]   for h in history["val"]],   label="Val")
    axes[0].set_title("Model C — Loss"); axes[0].legend()
    axes[1].plot(ep, [h["auroc"]  for h in history["val"]], color="green")
    axes[1].set_title("Model C — Val AUROC")
    axes[2].plot(ep, [h["pos_mae"] if not np.isnan(h["pos_mae"]) else 0
                      for h in history["val"]], color="orange")
    axes[2].set_title("Model C — Val Pos MAE")
    axes[3].plot(ep, [h["sev_r2"] if not np.isnan(h["sev_r2"]) else 0
                      for h in history["val"]], color="purple")
    axes[3].set_title("Model C — Val Severity R2")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves_c_v3.png", dpi=150)

    with open(MODELS_DIR / "history_c_v3.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Model C val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {MODELS_DIR}/{CKPT_NAME}")
    print(f"Run: python evaluate_c.py")


if __name__ == "__main__":
    train()