"""
data_overview.py — E6: what separates leak from no-leak in the Mendeley data?
===============================================================================
E4 found that loudness ranks the clean Looped recordings the WRONG way
round (RMS AUROC 0.21) and that a band-energy classifier trained on
Branched is inverted on Looped (AUROC 0.10). This script describes the
dataset before any model is involved:

1. Per-recording table: topology, leak type, flow condition (0.18 LPS,
   0.47 LPS, ND = no demand, Transient), windows, median RMS.
2. Confounding check. Each topology has ONE no-leak recording per flow
   condition. If loudness mostly tracks flow condition, pooling
   conditions can flip the leak/no-leak relationship. We report RMS AUROC
   pooled and WITHIN each flow condition (1 no-leak vs 4 leak recordings).
3. Within-topology generalisation to an unseen operating condition:
   band-energy logistic regression trained on 3 flow conditions, tested on
   the 4th (leave-one-condition-out), out-of-fold scores pooled. This is a
   simple upper reference for "can leaks be detected in this data at all
   when the operating condition changes?"
4. Figures: per-recording RMS by flow condition, and mean spectra.

Works for either sensor; nothing here uses the synthetic data or models.

    python experiments/data_overview.py
    python experiments/data_overview.py --sensor hydrophone --root datasets/Hydrophone/Hydrophone
"""

import argparse
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import mendeley as M
from _common import DATASETS, PLOTS_DIR, REPO_ROOT, record_run
from metrics import detection_report, fmt, point_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATASETS / "Accelerometer" / "Accelerometer")
    ap.add_argument("--sensor", choices=["accelerometer", "hydrophone"], default="accelerometer")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    ws = M.load_windowset(args.root, args.sensor, cache_dir=REPO_ROOT / "cache_mendeley",
                          verbose=False)
    flow = np.array([M.flow_condition(g) for g in ws.group])
    rms = F.rms(ws.x)
    be = F.band_energy(ws.x)
    results = {"sensor": args.sensor, "recordings": [], "rms_auroc": {}, "loco_logreg": {}}

    # ── 1. per-recording table ─────────────────────────────────────────────
    print(f"\n{'recording':52s} {'flow':>10s} {'label':>6s} {'win':>4s} {'median RMS':>11s}")
    for g in sorted(np.unique(ws.group)):
        m = ws.group == g
        row = {"recording": g, "topology": ws.topology[m][0], "condition": ws.condition[m][0],
               "flow": flow[m][0], "label": int(ws.y[m][0]), "n_windows": int(m.sum()),
               "median_rms": float(np.median(rms[m]))}
        results["recordings"].append(row)
        print(f"{g:52s} {row['flow']:>10s} {row['label']:>6d} {row['n_windows']:>4d} {row['median_rms']:>11.4g}")

    # ── 2. RMS AUROC pooled vs within flow condition ──────────────────────
    print("\nRMS as a leak score (AUROC; <0.5 = no-leak is louder)")
    for topo in M.TOPOLOGIES:
        t = ws.topology == topo
        if not t.any():
            continue
        pooled = detection_report(ws.y[t], rms[t], ws.group[t], threshold=np.inf, n_boot=args.n_boot)
        within = {}
        for fc in sorted(np.unique(flow[t])):
            m = t & (flow == fc)
            within[fc] = point_metrics(ws.y[m], rms[m])["auroc"]
        results["rms_auroc"][topo] = {"pooled": pooled["auroc"],
                                     "pooled_ci95": pooled.get("ci95", {}).get("auroc"),
                                     "within_flow_condition": within}
        print(f"  {topo:9s} pooled {fmt(pooled)} | within condition: " +
              ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} n/a" for k, v in within.items()))

    # ── 3. leave-one-flow-condition-out, within topology ───────────────────
    print("\nBand-energy logistic regression, leave-one-flow-condition-out (within topology)")
    for topo in M.TOPOLOGIES:
        t = np.flatnonzero(ws.topology == topo)
        if len(t) == 0:
            continue
        scores = np.full(len(ws), np.nan)
        per_fold = {}
        for fc in sorted(np.unique(flow[t])):
            te = t[flow[t] == fc]
            tr = t[flow[t] != fc]
            if len(np.unique(ws.y[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(be[tr], ws.y[tr])
            scores[te] = clf.decision_function(be[te])
            per_fold[fc] = point_metrics(ws.y[te], scores[te], threshold=0.0)["auroc"]
        ok = ~np.isnan(scores) & (ws.topology == topo)
        rep = detection_report(ws.y[ok], scores[ok], ws.group[ok], threshold=0.0, n_boot=args.n_boot)
        results["loco_logreg"][topo] = {"pooled_out_of_fold": rep, "per_held_out_condition": per_fold}
        print(f"  {topo:9s} pooled out-of-fold AUROC {fmt(rep)} | detect {fmt(rep, 'detection_rate')}"
              f" | false alarm {fmt(rep, 'false_alarm_rate')}")
        print(f"  {'':9s} per held-out condition: " +
              ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} n/a" for k, v in per_fold.items()))

    # ── 4. figures ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    conds = sorted(np.unique(flow))
    for j, topo in enumerate(M.TOPOLOGIES):
        recs = [r for r in results["recordings"] if r["topology"] == topo]
        for r in recs:
            x = conds.index(r["flow"]) + (0.15 if r["label"] else -0.15)
            x += np.random.default_rng(zlib.crc32(r["recording"].encode())).uniform(-0.06, 0.06)
            ax[0, j].scatter(x, r["median_rms"], s=60 if r["label"] == 0 else 30,
                             marker="s" if r["label"] == 0 else "o",
                             color="tab:blue" if r["label"] == 0 else "tab:red")
        ax[0, j].set_yscale("log")
        ax[0, j].set_xticks(range(len(conds)))
        ax[0, j].set_xticklabels(conds)
        ax[0, j].set_title(f"{topo}: median window RMS per recording")
        ax[0, j].scatter([], [], marker="s", color="tab:blue", label="no-leak")
        ax[0, j].scatter([], [], marker="o", color="tab:red", label="leak")
        ax[0, j].legend(fontsize=8)
        ax[0, j].grid(alpha=0.3)

        t = ws.topology == topo
        for lab, col in ((0, "tab:blue"), (1, "tab:red")):
            m = t & (ws.y == lab)
            if not m.any():
                continue
            f, p = welch(ws.x[m].astype(np.float64), fs=5000, nperseg=256, axis=-1)
            ax[1, j].semilogy(f, p.mean(axis=(0, 1)), color=col,
                              label="no-leak" if lab == 0 else "leak")
        ax[1, j].set_xlabel("Frequency (Hz)")
        ax[1, j].set_title(f"{topo}: mean spectrum (raw amplitude)")
        ax[1, j].legend(fontsize=8)
        ax[1, j].grid(alpha=0.3)
    plt.suptitle(f"Mendeley {args.sensor} recordings", fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"e6_data_overview_{args.sensor}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    summary = "; ".join(
        f"{t}: RMS pooled {r['pooled']:.2f}, within-cond "
        f"{np.nanmean([v for v in r['within_flow_condition'].values() if v is not None]):.2f}"
        for t, r in results["rms_auroc"].items())
    summary += "; LOCO logreg " + ", ".join(
        f"{t} {r['pooled_out_of_fold']['auroc']:.2f}" for t, r in results["loco_logreg"].items())
    record_run(f"e6_data_overview_{args.sensor}", {**vars(args), "root": str(args.root)},
               results, summary)


if __name__ == "__main__":
    main()
