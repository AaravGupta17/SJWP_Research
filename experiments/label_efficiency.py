"""
label_efficiency.py — E5: does synthetic pretraining reduce the real data needed?
===================================================================================
Research question: given only a small number of labelled REAL recordings,
does starting from the synthetic-pretrained AcousticLeakNet beat training
the same network from scratch, and a simple classical model?

Design
  train topology : Branched (all Branched recordings, incl. the ones used
                   as training noise — fine here, they are training data)
  test topology  : Looped (never seen in any form) — a different pipe
                   layout, so this also tests transfer across topology
  budgets        : --fractions of the Branched windows, class-stratified
  conditions     :
    scratch      AcousticLeakNet, random init, all layers trained
    pretrained   AcousticLeakNet from --ckpt (synthetic), all layers trained
    probe        AcousticLeakNet from --ckpt, body frozen, only the
                 detection head trained (what did synthetic training learn?)
    logreg       logistic regression on log band energies (features.py)
  seeds          : each (condition, fraction) repeated for --seeds; the
                   subset, init and batch order all depend on the seed

Every neural condition gets the same optimiser, the same number of
gradient steps (independent of budget), class-balanced batches, and the
same fixed-scale input normalisation calibrated on Branched no-leak
windows. There is no early stopping on the test set: the model after the
last step is evaluated once.

GPU strongly recommended for the default settings. On CPU use --quick.

    python experiments/label_efficiency.py
    python experiments/label_efficiency.py --quick
"""

import argparse
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import mendeley as M
from _common import (DATASETS, MODELS_DIR, N_SCALARS, PLOTS_DIR, REPO_ROOT,
                     build_model, get_device, load_model, predict_proba,
                     record_run)
from metrics import detection_report

CONDITIONS = ("scratch", "pretrained", "probe", "logreg")


def stratified_subset(y: np.ndarray, fraction: float, seed: int,
                      min_per_class: int = 4) -> np.ndarray:
    """Indices of a class-stratified random subset (deterministic in seed)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1):
        ci = np.flatnonzero(y == c)
        k = int(round(len(ci) * fraction))
        k = min(len(ci), max(k, min_per_class))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def balanced_batches(y: np.ndarray, batch: int, steps: int, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    half = batch // 2
    for _ in range(steps):
        yield np.concatenate([rng.choice(pos, half), rng.choice(neg, batch - half)])


def train_network(model, x, y, steps, batch, lr, seed, device, probe=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if probe:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.detection_head.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # probe: keep BatchNorm statistics frozen at their synthetic values
    model.train(not probe)
    for bi in balanced_batches(y, batch, steps, rng):
        xb = x[bi] * rng.uniform(0.85, 1.15, size=(len(bi), 1, 1)).astype(np.float32)
        xb = torch.as_tensor(xb, device=device)
        yb = torch.as_tensor(y[bi], dtype=torch.float32, device=device)
        det, _, _ = model(xb, torch.zeros(len(bi), N_SCALARS, device=device))
        loss = TF.binary_cross_entropy_with_logits(det, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATASETS / "Accelerometer" / "Accelerometer")
    ap.add_argument("--sensor", choices=["accelerometer", "hydrophone"], default="accelerometer")
    ap.add_argument("--ckpt", default="best_model_c_v4.pt")
    ap.add_argument("--train-topology", default="Branched")
    ap.add_argument("--test-topology", default="Looped")
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--quick", action="store_true",
                    help="CPU-friendly: fractions 0.02 0.1 1.0, seeds 0 1, 150 steps")
    args = ap.parse_args()
    if args.quick:
        args.fractions, args.seeds, args.steps = [0.02, 0.1, 1.0], [0, 1], 150

    device = get_device()
    print(f"Device: {device}")
    ws = M.load_windowset(args.root, args.sensor, cache_dir=REPO_ROOT / "cache_mendeley",
                          verbose=False)
    tr = ws.subset(ws.topology == args.train_topology)
    te = ws.subset(ws.topology == args.test_topology)
    assert not set(tr.group) & set(te.group), "train/test recordings overlap"
    ref = M.calibrate_ref_scale(tr.x[tr.y == 0])
    xtr, xte = M.fixed_scale_windows(tr.x, ref), M.fixed_scale_windows(te.x, ref)
    print(f"train {args.train_topology}: {len(tr)} windows / {len(set(tr.group))} recordings | "
          f"test {args.test_topology}: {len(te)} windows / {len(set(te.group))} recordings")

    base_model, base_ckpt = load_model(args.ckpt, device)
    be_tr, be_te = F.band_energy(tr.x), F.band_energy(te.x)

    runs = []
    # zero-shot reference point (no real labels at all)
    # scores are logits (threshold 0); see predict_proba on saturated sigmoids
    p0 = predict_proba(base_model, xte, logits=True)
    zs = detection_report(te.y, p0, te.group, threshold=0.0, n_boot=args.n_boot)
    runs.append({"condition": "pretrained", "fraction": 0.0, "seed": None,
                 "n_train_windows": 0, **zs})
    print(f"zero-shot pretrained: AUROC {zs['auroc']:.3f}")

    for frac in args.fractions:
        for seed in args.seeds:
            sub = stratified_subset(tr.y, frac, seed)
            for cond in args.conditions:
                t0 = time.time()
                if cond == "logreg":
                    clf = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, class_weight="balanced"))
                    clf.fit(be_tr[sub], tr.y[sub])
                    p = clf.decision_function(be_te)              # log-odds, threshold 0
                else:
                    if cond == "scratch":
                        torch.manual_seed(seed)
                        model = build_model(base_ckpt.get("cfg")).to(device)
                    else:
                        model = copy.deepcopy(base_model)
                    model = train_network(model, xtr[sub], tr.y[sub], args.steps, args.batch,
                                          args.lr, seed, device, probe=(cond == "probe"))
                    p = predict_proba(model, xte, logits=True)
                rep = detection_report(te.y, p, te.group, threshold=0.0,
                                       n_boot=args.n_boot, seed=seed)
                runs.append({"condition": cond, "fraction": frac, "seed": seed,
                             "n_train_windows": int(len(sub)), **rep})
                print(f"frac {frac:<5g} seed {seed} {cond:10s} n={len(sub):5d} "
                      f"AUROC {rep['auroc']:.3f}  bal.acc {rep['balanced_accuracy']:.3f}  "
                      f"({time.time() - t0:.0f}s)")

    # ── aggregate over seeds ──────────────────────────────────────────────
    agg = {}
    for cond in args.conditions:
        agg[cond] = []
        for frac in args.fractions:
            a = [r["auroc"] for r in runs if r["condition"] == cond and r["fraction"] == frac]
            n = [r["n_train_windows"] for r in runs if r["condition"] == cond and r["fraction"] == frac]
            agg[cond].append({"fraction": frac, "n_train_windows": int(np.mean(n)),
                              "auroc_mean": float(np.mean(a)), "auroc_std": float(np.std(a)),
                              "n_seeds": len(a)})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, rows in agg.items():
        xs = [r["n_train_windows"] for r in rows]
        m = np.array([r["auroc_mean"] for r in rows])
        s = np.array([r["auroc_std"] for r in rows])
        ax.plot(xs, m, "o-", label=cond)
        ax.fill_between(xs, m - s, m + s, alpha=0.15)
    ax.axhline(zs["auroc"], color="grey", ls="--", lw=1, label="pretrained, zero-shot")
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"Labelled real windows used ({args.train_topology})")
    ax.set_ylabel(f"AUROC on {args.test_topology} (mean ± std over seeds)")
    ax.set_title("Label efficiency on real recordings")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / "e5_label_efficiency.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    lo = args.fractions[0]
    summary = "; ".join(f"{c} @{lo:g}: {agg[c][0]['auroc_mean']:.3f}, @1: "
                        f"{agg[c][-1]['auroc_mean']:.3f}" for c in agg)
    record_run("e5_label_efficiency",
               {**vars(args), "root": str(args.root), "ref_scale": ref},
               {"aggregate": agg, "zero_shot": zs, "runs": runs}, summary)


if __name__ == "__main__":
    main()
