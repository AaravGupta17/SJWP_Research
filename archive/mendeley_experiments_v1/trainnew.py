    """
    train.py — AcousticLeakNet Training
    =====================================
    Train: Networks 1, 2, 4, 5  (../cache/train/)
    Val:   Network 7             (../cache/val/)
    
    Run after: python pregenerate.py --split train
               python pregenerate.py --split val
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

    CACHE_ROOT = Path("../cache_v2")
    MODELS_DIR = Path("../models")
    PLOTS_DIR  = Path("../plots")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
            print(f"CachedDataset [{split}]: {len(self.signals):,} samples | "
                  f"augment={augment}")

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
                    torch.tensor(lab[0], dtype=torch.float32),
                    torch.tensor(lab[1], dtype=torch.float32),
                    torch.tensor(lab[2], dtype=torch.float32),
                    torch.tensor(lab[3], dtype=torch.float32))


    def make_loader(split, batch_size, shuffle, augment=False):
        ds = CachedDataset(split, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=torch.cuda.is_available())


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
        all_prob, all_true = [], []
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

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        auroc   = roc_auc_score(all_true, all_prob)
        f1      = f1_score(all_true, (np.array(all_prob) >= 0.5).astype(int))
        pos_mae = mean_absolute_error(all_pos_t, all_pos_p) if all_pos_t else float("nan")
        sev_r2  = r2_score(all_sev_t, all_sev_p) if len(all_sev_t) > 1 else float("nan")

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

        train_loader = make_loader("train", CFG["batch_size"],
                                   shuffle=True,  augment=True)
        val_loader   = make_loader("val",   CFG["batch_size"],
                                   shuffle=False, augment=False)

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

        history    = {"train": [], "val": []}
        best_auroc  = 0.0
        start_epoch = 1

        # Resume from checkpoint if exists
        ckpt_path = MODELS_DIR / "best_model_v2.pt"
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
            print(f"  Continuing from epoch {start_epoch} to {CFG['n_epochs']}")

        for epoch in range(start_epoch, CFG["n_epochs"] + 1):
            tr = train_epoch(model, train_loader, criterion, optimizer,
                             device, epoch, CFG["n_epochs"])
            vl = val_epoch(model, val_loader, criterion,
                           device, epoch, CFG["n_epochs"])
            scheduler.step()
            history["train"].append(tr)
            history["val"].append(vl)

            print(f"Ep {epoch:3d}/{CFG['n_epochs']} | "
                  f"train={tr['loss']:.4f} "
                  f"(det={tr['det']:.3f} pos={tr['pos']:.3f} sev={tr['sev']:.3f}) | "
                  f"val AUROC={vl['auroc']:.4f} F1={vl['f1']:.4f} "
                  f"PosMAE={vl['pos_mae']:.4f} SevR²={vl['sev_r2']:.4f}")

            if vl["auroc"] > best_auroc:
                best_auroc = vl["auroc"]
                torch.save({
                    "model_state":     model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "cfg":             CFG,
                    "epoch":           epoch,
                    "val_auroc":       vl["auroc"],
                    "val_metrics":     vl,
                }, MODELS_DIR / "best_model_v2.pt")
                print(f"  ✓ Saved best (AUROC={best_auroc:.4f})")

        # Training curves
        ep = range(1, len(history["train"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ep, [h["loss"]  for h in history["train"]], label="Train")
        axes[0].plot(ep, [h["loss"]  for h in history["val"]],   label="Val")
        axes[0].set_title("Total Loss"); axes[0].legend()
        axes[1].plot(ep, [h["auroc"]   for h in history["val"]], color="green")
        axes[1].set_title("Val AUROC (Network 7 — unseen topology)")
        axes[2].plot(ep, [h["pos_mae"] for h in history["val"]], color="orange")
        axes[2].set_title("Val Position MAE (sensor-relative)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150)

        with open(MODELS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Best val AUROC: {best_auroc:.4f} (Network 7)")
        print(f"✓ Model: {MODELS_DIR}/best_model.pt")
        print(f"Now run: python evaluate.py")


    if __name__ == "__main__":
        train()