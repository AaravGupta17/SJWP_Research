"""
cross_dataset.py — E9: does leak detection transfer to real networks it has never seen?
=========================================================================================
Real-world test across independent public datasets (experiments/public_data.py):

  hk_noiselogger, hk_hydrophone   real BURIED networks, Hong Kong (metal + plastic;
                                  in these two sets no site has both a leak and a
                                  no-leak recording)
  dongguan                        outdoor training base (ductile iron, PE, steel, PVC)
  mendeley_acc, mendeley_hyd      lab testbed (PVC)

All windows: one sensor channel, 5 kHz, band-limited to 2 kHz, 0.4 s, and
standardised per window (sensor gains differ between datasets, so absolute
loudness would identify the dataset rather than the leak).

Protocols
  within  5-fold GroupKFold inside each dataset (groups = sites/recordings)
          — how well can a dataset be learned at all?
  loso    leave-one-SOURCE-out (main result): hold out every dataset from one
          source (all of Hong Kong, all of Mendeley, or Dongguan) and train on
          the rest. Needed because datasets from one source can share
          recordings: the Mendeley accelerometer and hydrophone sets are the same
          runs, and the Hong Kong noise loggers and hydrophones may cover the
          same leaks.
  lodo    leave-one-DATASET-out: for completeness only; a sibling dataset from
          the same source can leak into training.

Methods
  logreg      logistic regression on loudness-free spectral/statistical features
  cnn_scratch single-channel CNN (the AcousticLeakNet channel encoder + a new
              detection head), random initialisation
  cnn_pre     same, encoder initialised from the synthetic-trained checkpoint
  probe       synthetic encoder frozen, only the new head trained

Metrics: AUROC on the held-out data, balanced accuracy at threshold 0
(log-odds), 95% CI from a bootstrap over sites/recordings.

    python experiments/cross_dataset.py                       # GPU recommended
    python experiments/cross_dataset.py --methods logreg       # fast, CPU fine
    python experiments/cross_dataset.py --quick
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import public_data as P
from _common import MODELS_DIR, build_model, get_device, record_run
from metrics import detection_report, fmt

ALL_DATASETS = ["hk_noiselogger", "hk_hydrophone", "dongguan", "mendeley_acc", "mendeley_hyd"]
SOURCE = {"hk_noiselogger": "hongkong", "hk_hydrophone": "hongkong", "dongguan": "dongguan",
          "mendeley_acc": "mendeley", "mendeley_hyd": "mendeley"}
METHODS = ("logreg", "cnn_scratch", "cnn_pre", "probe")
BAND_EDGES = (10, 50, 100, 200, 400, 600, 800, 1200, 1600, 2000)
INPUT_SCALE = 0.1        # z-scored windows x 0.1 ~ the synthetic background level


# ── features for the classical baseline ────────────────────────────────────────

def features_1ch(x: np.ndarray, fs: int = P.FS) -> np.ndarray:
    """Loudness-free features of (N, T) windows."""
    x = x.astype(np.float64)
    f, pxx = welch(x, fs=fs, nperseg=256, axis=-1)
    tot = pxx.sum(axis=1, keepdims=True) + 1e-30
    bands = [np.log10(pxx[:, (f >= lo) & (f < hi)].sum(axis=1) / tot[:, 0] + 1e-12)
             for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    centroid = (pxx * f).sum(axis=1) / tot[:, 0]
    m = f <= P.BAND_HZ
    flat = np.exp(np.log(pxx[:, m] + 1e-30).mean(axis=1)) / (pxx[:, m].mean(axis=1) + 1e-30)
    zc = (np.diff(np.sign(x), axis=1) != 0).mean(axis=1)
    crest = np.abs(x).max(axis=1) / (x.std(axis=1) + 1e-12)
    return np.column_stack(bands + [centroid / 1000, flat, kurtosis(x, axis=1), crest, zc])


def standardise(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


# ── single-channel network built from the AcousticLeakNet encoder ─────────────

class SingleChannelNet(nn.Module):
    def __init__(self, encoder: nn.Module, enc_channels: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(2 * enc_channels, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):                      # x: (B, T)
        f = self.encoder(x.unsqueeze(1))       # (B, C, T/16)
        return self.head(torch.cat([f.mean(-1), f.amax(-1)], dim=-1)).squeeze(-1)


def make_net(method: str, ckpt: str, device):
    if method == "cnn_scratch":
        enc = build_model({}).channel_encoder
    else:
        c = torch.load(MODELS_DIR / ckpt, map_location="cpu", weights_only=False)
        full = build_model(c.get("cfg"))
        full.load_state_dict(c["model_state"])
        enc = copy.deepcopy(full.channel_encoder)
    return SingleChannelNet(enc).to(device)


def train_net(net, x, y, steps, batch, lr, seed, device, probe=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if probe:
        for p in net.encoder.parameters():
            p.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    net.train()
    if probe:
        net.encoder.eval()                     # keep synthetic BatchNorm statistics
    for _ in range(steps):
        bi = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
        xb = x[bi] * INPUT_SCALE
        shift = rng.integers(0, x.shape[1], size=len(bi))          # random circular time shift
        xb = np.stack([np.roll(r, s) for r, s in zip(xb, shift)])
        loss = TF.binary_cross_entropy_with_logits(
            net(torch.as_tensor(xb, device=device)),
            torch.as_tensor(y[bi], dtype=torch.float32, device=device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    net.eval()
    return net


@torch.no_grad()
def predict_net(net, x, device, batch=256):
    out = [net(torch.as_tensor(x[i:i + batch] * INPUT_SCALE, device=device)).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros(0)


def fit_predict(method, xtr, ytr, xte, args, seed, device, ftr=None, fte=None):
    if method == "logreg":
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
        clf.fit(ftr, ytr)
        return clf.decision_function(fte)
    net = make_net(method, args.ckpt, device)
    net = train_net(net, xtr, ytr, args.steps, args.batch, args.lr, seed, device,
                    probe=(method == "probe"))
    return predict_net(net, xte, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--protocols", nargs="+", default=["within", "loso"],
                    choices=["within", "loso", "lodo"])
    ap.add_argument("--ckpt", default="best_model_c_v4.pt", help="synthetic checkpoint for cnn_pre/probe")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="200 steps, 300 bootstrap draws")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.n_boot = 200, 300
    device = get_device()
    print(f"Device: {device}\nLoading datasets (5 kHz, band-limited to {P.BAND_HZ:.0f} Hz):")
    W = P.load(args.datasets)
    present = [d for d in args.datasets if (W.dataset == d).any()]
    X = standardise(W.x)
    Fz = features_1ch(X)
    results = {"datasets": {d: {"windows": int((W.dataset == d).sum()),
                                "groups": int(len(np.unique(W.group[W.dataset == d]))),
                                "leak_fraction": float(W.y[W.dataset == d].mean())}
                            for d in present},
               "within": {}, "loso": {}, "lodo": {}}

    if "within" in args.protocols:
        print("\n=== within-dataset, 5-fold grouped CV ===")
        for d in present:
            m = np.flatnonzero(W.dataset == d)
            n_groups = len(np.unique(W.group[m]))
            if n_groups < 5 or len(np.unique(W.y[m])) < 2:
                print(f"  {d}: too few groups/classes, skipped")
                continue
            results["within"][d] = {}
            for method in args.methods:
                t0 = time.time()
                oof = np.full(len(m), np.nan)
                for k, (tr, te) in enumerate(GroupKFold(5).split(m, W.y[m], W.group[m])):
                    if len(np.unique(W.y[m[tr]])) < 2:
                        continue
                    oof[te] = fit_predict(method, X[m[tr]], W.y[m[tr]], X[m[te]], args,
                                          args.seed + k, device, Fz[m[tr]], Fz[m[te]])
                ok = ~np.isnan(oof)
                rep = detection_report(W.y[m][ok], oof[ok], W.group[m][ok], threshold=0.0,
                                       n_boot=args.n_boot, seed=args.seed)
                results["within"][d][method] = rep
                print(f"  {d:16s} {method:12s} AUROC {fmt(rep)} | bal.acc "
                      f"{rep['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)")

    # held-out splits: (protocol, name, test mask)
    src = np.array([SOURCE[d] for d in W.dataset])
    splits = []
    if "loso" in args.protocols and len(set(src)) > 1:
        splits += [("loso", s_, src == s_) for s_ in sorted(set(src))]
    if "lodo" in args.protocols and len(present) > 1:
        splits += [("lodo", d, W.dataset == d) for d in present]

    for proto, name, te_mask in splits:
        tr_mask = ~te_mask
        if len(np.unique(W.y[tr_mask])) < 2:
            continue
        print(f"\n=== {proto}: hold out {name} | train on {sorted(set(W.dataset[tr_mask]))} ===")
        scores = {}
        for method in args.methods:
            t0 = time.time()
            scores[method] = fit_predict(method, X[tr_mask], W.y[tr_mask], X[te_mask], args,
                                         args.seed, device, Fz[tr_mask], Fz[te_mask])
            print(f"  ({method}: {time.time() - t0:.0f}s)")
        test_ds = W.dataset[te_mask]
        for d in sorted(set(test_ds)):                 # report each held-out dataset separately
            m = test_ds == d
            y_d, g_d = W.y[te_mask][m], W.group[te_mask][m]
            if len(np.unique(y_d)) < 2:
                continue
            results[proto].setdefault(d, {})
            for method in args.methods:
                rep = detection_report(y_d, scores[method][m], g_d, threshold=0.0,
                                       n_boot=args.n_boot, seed=args.seed)
                results[proto][d][method] = rep
                print(f"  test {d:16s} {method:12s} AUROC {fmt(rep)} | detect "
                      f"{rep['detection_rate']:.2f} false alarm {rep['false_alarm_rate']:.2f}")

    def best(proto):
        rows = [(d, m, r["auroc"]) for d, mm in results[proto].items() for m, r in mm.items()]
        return ", ".join(f"{d}/{m} {a:.2f}" for d, m, a in rows)
    summary = f"within: {best('within')} | loso: {best('loso')}"
    record_run("e9_cross_dataset", vars(args), results, summary[:900])


if __name__ == "__main__":
    main()
