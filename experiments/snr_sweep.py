"""
snr_sweep.py — E3: where does detection/localisation break as leaks get quieter?
=================================================================================
The original Model C test sets use leak SNRs of 0.5–12 dB above the
background noise, where every method may look perfect. This script
regenerates a class-balanced subsample of each held-out network with the
leak SNR FIXED at each level in --snr (dB), using the unchanged Model C
synthesiser (model_C/dataset_c.py, snr_override_db hook), and evaluates:

  model     AcousticLeakNet detection AUROC and localisation MAE
  rms       loudness-only detector AUROC
  gcc       GCC-PHAT peak AUROC and GCC-PHAT localisation MAE

--no-dc additionally regenerates every level WITHOUT the pressure DC
offset, to test whether the model's detection depends on it.

Evaluation only — nothing is retrained. Needs the raw EPANET CSVs
(datasets/NetworkList) and the Mendeley hydrophone noise bank
(datasets/Hydrophone), exactly as model_C/pregen_c.py does.

    python experiments/snr_sweep.py
    python experiments/snr_sweep.py --n 1000 --snr -20 -10 0 10 --no-dc
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _common import (PLOTS_DIR, REPO_ROOT, load_model, predict_proba,
                     record_run)
from metrics import point_metrics

SPLITS = {"L-TOWN (Network 3)": "test_Network_3.csv",
          "KY15 (Network 6)":   "test_Network_6.csv",
          "Richmond (Network 8)": "test_Network_8.csv"}
DEFAULT_SNR = [-20, -15, -10, -5, 0, 5, 10]


def sample_index(csv_path, n: int, seed: int, out_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parts = [g.sample(n=min(len(g), n // 2), random_state=seed)
             for _, g in df.groupby(df["file_type"] == "leak")]
    sub = pd.concat(parts).sort_index()
    sub.to_csv(out_path, index=False)
    return sub


def generate(ds, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    sig, det, pos, pv = [], [], [], []
    for i in range(len(ds)):
        s, _, d, p, _, v = ds[i]
        if d.item() < 0:          # failed row
            continue
        sig.append(s.numpy()); det.append(d.item()); pos.append(p.item()); pv.append(v.item())
    return (np.stack(sig).astype(np.float32), np.array(det, int),
            np.array(pos, np.float32), np.array(pv, np.float32))


def evaluate(model, sig, det, pos, pv) -> dict:
    import features as F
    from gccphat import score_sample
    p, p_pos = predict_proba(model, sig, return_pos=True, logits=True)
    pm = point_metrics(det, p, threshold=0.0)          # logit 0 == P 0.5
    gcc = np.array([score_sample(w.astype(np.float64)) for w in sig])
    m = (det == 1) & (pv == 1)
    return {
        "model_auroc": pm["auroc"],
        "model_detection_rate": pm["detection_rate"],
        "model_false_alarm_rate": pm["false_alarm_rate"],
        "rms_auroc": point_metrics(det, F.rms(sig))["auroc"],
        "gcc_auroc": point_metrics(det, gcc[:, 0])["auroc"],
        "model_pos_mae": float(np.abs(p_pos[m] - pos[m]).mean()) if m.any() else None,
        "gcc_pos_mae": float(np.abs(gcc[m, 1] - pos[m]).mean()) if m.any() else None,
        "midpoint_pos_mae": float(np.abs(0.5 - pos[m]).mean()) if m.any() else None,
        "n": int(len(det)), "n_leak": int(det.sum()), "n_pos_valid": int(m.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model_c_v4.pt")
    ap.add_argument("--snr", type=float, nargs="+", default=DEFAULT_SNR)
    ap.add_argument("--n", type=int, default=2000, help="rows per network (class-balanced)")
    ap.add_argument("--no-dc", action="store_true", help="also run without pressure DC offset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_model(args.ckpt)
    work = REPO_ROOT / "cache_sweep"
    work.mkdir(exist_ok=True)

    # dataset_c resolves ../datasets/... relative to model_C/, like pregen_c.py
    os.chdir(REPO_ROOT / "model_C")
    from dataset_c import LeakDataset

    variants = [True, False] if args.no_dc else [True]
    levels = [None] + list(args.snr)          # None = original SNR distribution
    results = {}
    for name, csv_name in SPLITS.items():
        sub_csv = work / f"sweep_{csv_name}"
        sample_index(REPO_ROOT / "data" / "csv" / csv_name, args.n, args.seed, sub_csv)
        ds = LeakDataset(str(sub_csv), augment=False)
        no_src = sum(1 for i in ds._valid_idx
                     if ds._cache[i]["leak_status"] == 1
                     and ds._cache[i]["d_left"] <= 0 and ds._cache[i]["d_right"] <= 0)
        results[name] = {"leak_rows_without_injected_source": no_src, "levels": {}}
        for dc in variants:
            ds.include_pressure_dc = dc
            for lv in levels:
                ds.snr_override_db = lv
                key = f"{'native' if lv is None else f'{lv:g}dB'}{'' if dc else '_noDC'}"
                r = evaluate(model, *generate(ds, args.seed))
                results[name]["levels"][key] = r
                print(f"{name:22s} {key:12s} model AUROC {r['model_auroc']:.3f} | "
                      f"rms {r['rms_auroc']:.3f} | gcc {r['gcc_auroc']:.3f} | "
                      f"posMAE model {r['model_pos_mae'] or float('nan'):.3f} "
                      f"gcc {r['gcc_pos_mae'] or float('nan'):.3f}")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 8), squeeze=False)
    for j, (name, res) in enumerate(results.items()):
        for dc in variants:
            sfx = "" if dc else "_noDC"
            xs = list(args.snr)
            L = [res["levels"][f"{s:g}dB{sfx}"] for s in xs]
            ls = "-" if dc else "--"
            tag = "" if dc else " (no DC)"
            ax[0, j].plot(xs, [l["model_auroc"] for l in L], "o" + ls, label="AcousticLeakNet" + tag)
            if dc:
                ax[0, j].plot(xs, [l["rms_auroc"] for l in L], "s-", label="RMS energy")
                ax[0, j].plot(xs, [l["gcc_auroc"] for l in L], "^-", label="GCC-PHAT peak")
                ax[1, j].plot(xs, [l["model_pos_mae"] for l in L], "o-", label="AcousticLeakNet")
                ax[1, j].plot(xs, [l["gcc_pos_mae"] for l in L], "^-", label="GCC-PHAT")
                ax[1, j].plot(xs, [l["midpoint_pos_mae"] for l in L], "k:", label="always 0.5")
        ax[0, j].axvspan(0.5, 12, color="grey", alpha=0.12, label="training SNR range")
        ax[0, j].set_title(name)
        ax[0, j].set_ylabel("Detection AUROC")
        ax[0, j].set_ylim(0.0, 1.02)
        ax[1, j].set_xlabel("Leak SNR (dB)")
        ax[1, j].set_ylabel("Position MAE (fraction of span)")
        for a in ax[:, j]:
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    plt.tight_layout()
    out = PLOTS_DIR / "e3_snr_sweep.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    def at(name, key):
        v = results[name]["levels"].get(key, {}).get("model_auroc")
        return f"{v:.3f}" if v is not None else "n/a"
    lo = f"{min(args.snr):g}dB"
    summary = "; ".join(f"{n.split(' ')[0]}: native {at(n, 'native')}, {lo} {at(n, lo)}"
                        for n in results)
    record_run("e3_snr_sweep", vars(args), results, summary)


if __name__ == "__main__":
    main()
