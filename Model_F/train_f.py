"""
train_f.py — Model F training: fresh mixtures every step, real + synthetic
==========================================================================
Architecture = AcousticLeakNet (model_C/model.py), so every experiment can
load the checkpoint. What changes is the data and the objective:

  synthetic rows  pre-generated leak component (pregen_f.py) + a fresh
                  background [F1] + interferers, band-limited [F3]
  real rows       labelled one-sensor windows from the TRAINING sources
                  (Hong Kong / Dongguan), made two-channel with the same
                  pairing as real backgrounds [F5]. No position/flow label.
  every row       random EQ [F4], joint z-score [F2]. Each batch has 50%
                  leak in both the synthetic and the real part, so "real vs
                  synthetic" says nothing about the label.

  loss            BCE on detection (label smoothing 0.05) + fixed-weight
                  Huber on position (x0.5) and flow (x0.2), synthetic leak
                  rows only. Models C-E used learned uncertainty weights:
                  once detection is easy its weight grows without bound,
                  which drove logits past +-17 (INTEGRITY_LOG #12).
  checkpoint      best mean of synthetic-val and real-val detection AUROC
                  (on logits). Models C-E chose by SevR2 - PosMAE, so
                  detection never influenced which model was kept.

--exclude-source removes a source from BOTH the backgrounds and the real
rows, so a model can be tested on a source it never heard (E11).

    python Model_F/train_f.py                                   # main model
    python Model_F/train_f.py --exclude-source hongkong --prefix f_nohk
    python Model_F/train_f.py --real-frac 0 --prefix f_synonly  # ablation
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
import augment_f as A                                      # noqa: E402
from _common import MODELS_DIR, build_model, record_run   # noqa: E402

SOURCES = ("hongkong", "dongguan")


class Bank:
    """Real windows from bank_f.py, filtered by split and excluded sources."""

    def __init__(self, path: Path, val: bool, exclude=()):
        z = np.load(path, allow_pickle=False)
        keep = (z["s_val"] == val) & ~np.isin(z["s_source"], list(exclude))
        self.single = z["single"][keep].astype(np.float32)
        self.y = z["s_y"][keep].astype(int)
        self.group = z["s_group"][keep]
        pk = z["p_val"] == val
        self.pairs = z["pairs"][pk].astype(np.float32)
        # windows of each group, for drawing a second window of one recording
        order = np.argsort(self.group, kind="stable")
        g_sorted = self.group[order]
        starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
        self._members = np.split(order, starts[1:])
        self._group_of = np.empty(len(self.group), int)
        for k, m in enumerate(self._members):
            self._group_of[m] = k
        self.no_leak = np.flatnonzero(self.y == 0)
        self.leak = np.flatnonzero(self.y == 1)

    def partner(self, i: int, rng) -> int:
        """Another window of the same recording (or itself if it has one)."""
        m = self._members[self._group_of[i]]
        return int(m[rng.integers(len(m))])

    def two_channel(self, i: int, rng) -> np.ndarray:
        return A.pair_channels(self.single[i], self.single[self.partner(i, rng)], rng)


def background(bank: Bank, rng) -> np.ndarray:
    """(2, T) unit-RMS background for a synthetic row [F1]."""
    u = rng.random()
    if u < 0.55 and len(bank.no_leak):
        return bank.two_channel(int(rng.choice(bank.no_leak)), rng)
    if u < 0.70 and len(bank.pairs):
        return bank.pairs[rng.integers(len(bank.pairs))].copy()
    if u < 0.90:
        a, b = A.coloured_noise(rng), A.coloured_noise(rng)
        return A.band_limit(A.pair_channels(a, b, rng))
    return A.band_limit(rng.standard_normal((2, A.T)))


class MixDataset(Dataset):
    """Index i only selects the RNG stream; each item is built on the fly."""

    def __init__(self, cache: Path, split: str, bank: Bank, real_frac: float, seed: int,
                 fixed: bool = False):
        self.dir, self.bank, self.real_frac = cache / split, bank, real_frac
        self.seed, self.fixed = seed, fixed
        labels = np.load(self.dir / "labels.npy")
        self.labels = labels
        self.leak_row = np.load(self.dir / "leak_row.npy")
        self.syn_leak = np.flatnonzero(labels[:, 0] == 1)
        self.syn_none = np.flatnonzero(labels[:, 0] == 0)
        if real_frac > 0 and (len(bank.leak) == 0 or len(bank.no_leak) == 0):
            raise ValueError("real rows need both classes in the bank")
        self._leak = None                      # memmap opened lazily in each worker
        self.length = 0
        self.epoch = 0

    def leak_array(self):
        if self._leak is None:
            self._leak = np.load(self.dir / "leak.npy", mmap_mode="r")
        return self._leak

    def __len__(self):
        return self.length

    def rng_for(self, i: int):
        """Reproducible: fixed sets depend on (seed, i); training items on
        (seed, epoch, i), so every epoch draws new mixtures."""
        if self.fixed:
            return np.random.default_rng([self.seed, i])
        return np.random.default_rng([self.seed, self.epoch, i])

    def __getitem__(self, i: int):
        rng = self.rng_for(i)
        want_leak = rng.random() < 0.5
        if rng.random() < self.real_frac:
            pool = self.bank.leak if want_leak else self.bank.no_leak
            x = self.bank.two_channel(int(rng.choice(pool)), rng)
            x = A.finish(x.astype(np.float64), rng)
            lab = np.array([float(want_leak), 0.0, 0.0, 0.0], np.float32)
            return torch.from_numpy(x), torch.from_numpy(lab), torch.tensor(0.0)
        r = int(rng.choice(self.syn_leak if want_leak else self.syn_none))
        x = background(self.bank, rng) * 10 ** (rng.uniform(-3, 3, (2, 1)) / 20)
        extra = A.interferers(rng)
        if self.leak_row[r] >= 0:
            extra += self.leak_array()[self.leak_row[r]].astype(np.float64)
        x = x + A.band_limit(extra)
        x = A.finish(x, rng)
        return torch.from_numpy(x), torch.from_numpy(self.labels[r].copy()), torch.tensor(1.0)


def loss_fn(det, pos, sev, lab, is_syn, smooth=0.05):
    y = lab[:, 0]
    l_det = TF.binary_cross_entropy_with_logits(det, y * (1 - smooth) + 0.5 * smooth)
    leak = (y == 1) & (is_syn == 1)
    pm = leak & (lab[:, 3] == 1)
    l_pos = TF.huber_loss(pos[pm], lab[pm, 1], delta=0.1) if pm.any() else det.sum() * 0
    l_sev = TF.huber_loss(sev[leak], lab[leak, 2], delta=1.0) if leak.any() else det.sum() * 0
    return l_det + 0.5 * l_pos + 0.2 * l_sev, l_det.item()


def fixed_set(ds: MixDataset, n: int):
    xs, ys, ss = zip(*(ds[i] for i in range(n)))
    return torch.stack(xs), torch.stack(ys), torch.stack(ss)


@torch.no_grad()
def logits_of(model, x, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        det, _, _ = model(xb, torch.zeros(len(xb), 11, device=device))
        out.append(det.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_f")
    ap.add_argument("--exclude-source", nargs="*", default=[], choices=SOURCES)
    ap.add_argument("--real-frac", type=float, default=0.3, help="share of rows that are real windows")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=4000, help="synthetic validation samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="f")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / args.cache
    bank_tr = Bank(cache / "bank.npz", val=False, exclude=args.exclude_source)
    bank_va = Bank(cache / "bank.npz", val=True, exclude=args.exclude_source)
    print(f"Device: {device} | excluded sources: {args.exclude_source or 'none'}")
    print(f"Bank train: {len(bank_tr.single)} real windows ({len(bank_tr.no_leak)} no-leak), "
          f"{len(bank_tr.pairs)} Mendeley pairs | val: {len(bank_va.single)} real windows")

    train_ds = MixDataset(cache, "train", bank_tr, args.real_frac, args.seed)
    train_ds.length = args.steps * args.batch
    syn_val = MixDataset(cache, "val", bank_va, 0.0, args.seed + 1, fixed=True)
    xs_val, ys_val, _ = fixed_set(syn_val, args.n_val)
    xr_val = yr_val = None
    if args.real_frac > 0 and len(bank_va.leak) and len(bank_va.no_leak):
        rng = np.random.default_rng(args.seed + 2)
        xr_val = torch.from_numpy(np.stack([A.finish(bank_va.two_channel(i, rng).astype(np.float64),
                                                     rng) for i in range(len(bank_va.single))]))
        yr_val = bank_va.y
    print(f"Validation: {len(xs_val)} synthetic, {0 if xr_val is None else len(xr_val)} real windows")

    # workers are re-created each epoch so they see the new train_ds.epoch
    loader = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    cfg = {"base_channels": args.base_channels, "dropout": args.dropout, "fusion": "cca",
           "input_norm": "zscore", "band_hz": A.BAND_HZ, **vars(args)}
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ckpt_path = MODELS_DIR / f"best_model_{args.prefix}_seed{args.seed}.pt"
    history, best = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.epoch = epoch
        t0, losses, dets = time.time(), [], []
        for x, lab, is_syn in loader:
            x, lab, is_syn = x.to(device), lab.to(device), is_syn.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                det, pos, sev = model(x, torch.zeros(len(x), 11, device=device))
            loss, l_det = loss_fn(det.float(), pos.float(), sev.float(), lab, is_syn)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item()); dets.append(l_det)
        auc_syn = roc_auc_score(ys_val[:, 0].numpy(), logits_of(model, xs_val, device))
        auc_real = (roc_auc_score(yr_val, logits_of(model, xr_val, device))
                    if xr_val is not None else float("nan"))
        score = auc_syn if np.isnan(auc_real) else (auc_syn + auc_real) / 2
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "det_loss": float(np.mean(dets)),
                        "val_auroc_synthetic": auc_syn, "val_auroc_real": auc_real, "score": score})
        print(f"Ep {epoch:3d}/{args.epochs} | loss {np.mean(losses):.4f} (det {np.mean(dets):.4f}) | "
              f"val AUROC synthetic {auc_syn:.4f} real {auc_real:.4f} | {time.time() - t0:.0f}s")
        if score > best:
            best = score
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_score": score, "history": history}, ckpt_path)
            print(f"  saved {ckpt_path.name}")

    (MODELS_DIR / f"history_{args.prefix}_seed{args.seed}.json").write_text(json.dumps(history, indent=1))
    b = max(history, key=lambda h: h["score"])
    record_run(f"train_{args.prefix}", cfg, {"history": history, "best": b, "checkpoint": ckpt_path.name},
               f"best epoch {b['epoch']}: val AUROC synthetic {b['val_auroc_synthetic']:.3f}, "
               f"real {b['val_auroc_real']:.3f} (held-out groups of the training sources)")


if __name__ == "__main__":
    main()
