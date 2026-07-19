import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # Remove unsupported flag (Windows)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import LeakDataset, collate_fn
from model import AcousticLeakNet, UncertaintyLoss, count_parameters

# ─────────────────────────────────────────────────────────────
# Performance flags (RTX optimization)
# ─────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODELS_DIR = Path("../models")
PLOTS_DIR  = Path("../plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "batch_size":    128,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "n_epochs":      30,
    "val_split":     0.2,
    "base_channels": 64,
    "dropout":       0.3,
    "num_workers":   0,
    "seed":          42,
}

# ─────────────────────────────────────────────────────────────
# Dataset split
# ─────────────────────────────────────────────────────────────
def build_splits(index_csv: str, val_split: float, seed: int):
    import pandas as pd
    df = pd.read_csv(index_csv)
    labels = (df["file_type"] == "leak").astype(int).values
    idx = np.arange(len(df))
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_split,
        stratify=labels, random_state=seed
    )
    train_ds = Subset(LeakDataset(index_csv, augment=True), idx_tr)
    val_ds   = Subset(LeakDataset(index_csv, augment=False), idx_val)
    return train_ds, val_ds


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        persistent_workers=False,
    )

# ─────────────────────────────────────────────────────────────
# Training Epoch (AMP enabled)
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    losses = []

    pbar = tqdm(loader, desc=f"[Train {epoch}]", leave=False)

    for batch in pbar:
        if batch is None:
            continue

        signal, scalars, true_det, true_pos, true_sev, pos_valid = [
            x.to(device, non_blocking=True) for x in batch
        ]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            pred_det, pred_pos, pred_sev = model(signal, scalars)
            loss, *_ = criterion(
                pred_det, pred_pos, pred_sev,
                true_det, true_pos, true_sev, pos_valid
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(losses)


# ─────────────────────────────────────────────────────────────
# Validation Epoch (Optimized accumulation)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    losses = []

    det_probs = []
    det_true  = []
    pos_pred, pos_true = [], []
    sev_pred, sev_true = [], []

    pbar = tqdm(loader, desc=f"[Val {epoch}]", leave=False)

    for batch in pbar:
        if batch is None:
            continue

        signal, scalars, true_det, true_pos, true_sev, pos_valid = [
            x.to(device, non_blocking=True) for x in batch
        ]

        pred_det, pred_pos, pred_sev = model(signal, scalars)
        loss, *_ = criterion(
            pred_det, pred_pos, pred_sev,
            true_det, true_pos, true_sev, pos_valid
        )

        losses.append(loss.item())

        probs = torch.sigmoid(pred_det).cpu()
        det_probs.append(probs)
        det_true.append(true_det.cpu())

        leak_mask = true_det.bool()

        if leak_mask.any():
            sev_pred.append(pred_sev[leak_mask].cpu())
            sev_true.append(true_sev[leak_mask].cpu())

        pos_mask = leak_mask & pos_valid.bool()
        if pos_mask.any():
            pos_pred.append(pred_pos[pos_mask].cpu())
            pos_true.append(true_pos[pos_mask].cpu())

    # Concatenate once (more memory efficient)
    det_probs = torch.cat(det_probs).numpy()
    det_true  = torch.cat(det_true).numpy()

    auroc = roc_auc_score(det_true, det_probs)
    f1    = f1_score(det_true, (det_probs >= 0.5).astype(int))

    pos_mae = (
        mean_absolute_error(torch.cat(pos_true), torch.cat(pos_pred))
        if pos_true else float("nan")
    )

    sev_r2 = (
        r2_score(torch.cat(sev_true), torch.cat(sev_pred))
        if len(sev_true) > 1 else float("nan")
    )

    return np.mean(losses), auroc, f1, pos_mae, sev_r2


# ─────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────
def train(index_csv: str = "../csv/train_index_sampled.csv"):
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds = build_splits(index_csv, CFG["val_split"], CFG["seed"])
    train_loader = make_loader(train_ds, CFG["batch_size"], True)
    val_loader   = make_loader(val_ds, CFG["batch_size"], False)

    model = AcousticLeakNet(
        signal_length=8000,
        n_scalars=9,
        base_channels=CFG["base_channels"],
        dropout=CFG["dropout"]
    ).to(device)

    # PyTorch 2.x compile (huge speedup)
    model = torch.compile(model)

    criterion = UncertaintyLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["n_epochs"], eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    print("Parameters:", count_parameters(model))

    best_auroc = 0.0

    for epoch in range(1, CFG["n_epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device, epoch
        )

        val_loss, auroc, f1, pos_mae, sev_r2 = val_epoch(
            model, val_loader, criterion,
            device, epoch
        )

        scheduler.step()

        print(
            f"Ep {epoch:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val AUROC {auroc:.4f} F1 {f1:.4f} "
            f"PosMAE {pos_mae:.4f} SevR² {sev_r2:.4f}"
        )

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), MODELS_DIR / "acousticleaknet_best.pt")
            print("✓ Saved best model")

    print("Best AUROC:", best_auroc)


if __name__ == "__main__":
    train()