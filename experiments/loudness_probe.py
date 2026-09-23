"""
loudness_probe.py — E1: does the detector just threshold loudness?
===================================================================
Question: what does each trained checkpoint output for inputs that contain
NO leak at all, as their loudness (RMS) and DC offset change?

Inputs are pure noise (AR(1)-coloured, like flow noise, and white), so the
correct answer is always "no leak". If P(leak) jumps to ~1 once RMS passes
some level, the network is acting as a loudness threshold. The real-data
evaluation fed it per-window z-scored inputs (RMS = 1.0), which is why it
flagged almost every real no-leak window.

Second panel: fixed quiet noise plus a constant DC offset. Model C's
synthesiser adds a pressure-proportional DC offset to every window
(dataset_c.py, p * 3e-4), so the network may also have learned to use DC.

Needs no dataset: only the checkpoints in models/.

    python experiments/loudness_probe.py
    python experiments/loudness_probe.py --ckpts best_model_c_v4.pt --n 256
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import lfilter

from _common import (MODELS_DIR, PLOTS_DIR, SIGNAL_LENGTH, load_model,
                     predict_proba, record_run)

DEFAULT_CKPTS = ["best_model_c_v4.pt", "best_model_c_seed42.pt",
                 "best_model_c_seed123.pt", "best_model_d.pt", "best_model_mend.pt"]
RMS_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0]
DC_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
QUIET_RMS = 0.02


def make_noise(n: int, rms: float, kind: str, rng) -> np.ndarray:
    x = rng.standard_normal((n, 2, SIGNAL_LENGTH))
    if kind == "coloured":
        x = lfilter([1.0], [1.0, -0.95], x, axis=-1)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / x.std(axis=-1, keepdims=True) * rms
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--n", type=int, default=128, help="noise windows per level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    results = {}
    for ck in args.ckpts:
        if not (MODELS_DIR / ck).exists():
            print(f"skip {ck}: not found")
            continue
        model, _ = load_model(ck)
        rng = np.random.default_rng(args.seed)
        res = {"rms_sweep": {}, "dc_sweep": {}}
        for kind in ("coloured", "white"):
            res["rms_sweep"][kind] = [
                float(predict_proba(model, make_noise(args.n, r, kind, rng)).mean())
                for r in RMS_LEVELS]
        for dc in DC_LEVELS:
            x = make_noise(args.n, QUIET_RMS, "coloured", rng) + np.float32(dc)
            res["dc_sweep"][str(dc)] = float(predict_proba(model, x).mean())
        results[ck] = res
        c = res["rms_sweep"]["coloured"]
        flip = next((r for r, p in zip(RMS_LEVELS, c) if p >= 0.5), None)
        res["rms_at_first_p_over_0.5"] = flip
        print(f"{ck:28s} coloured-noise P(leak): " +
              "  ".join(f"{r:g}:{p:.2f}" for r, p in zip(RMS_LEVELS, c)))
        print(f"{'':28s} DC sweep (noise rms {QUIET_RMS}): " +
              "  ".join(f"{d}:{p:.2f}" for d, p in res["dc_sweep"].items()))

    if not results:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for ck, res in results.items():
        ax[0].plot(RMS_LEVELS, res["rms_sweep"]["coloured"], marker="o", label=ck)
        ax[1].plot(DC_LEVELS, list(res["dc_sweep"].values()), marker="o", label=ck)
    ax[0].axvline(1.0, color="grey", ls=":", lw=1)
    ax[0].text(1.0, 0.5, " z-scored input\n (RMS = 1)", fontsize=8, color="grey")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Input RMS (leak-free coloured noise)")
    ax[0].set_ylabel("Mean P(leak)")
    ax[0].set_title("Leak-free noise: P(leak) vs loudness")
    ax[1].set_xlabel(f"Added DC offset (noise RMS {QUIET_RMS})")
    ax[1].set_title("Leak-free noise: P(leak) vs DC offset")
    for a in ax:
        a.set_ylim(-0.03, 1.03)
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    out = PLOTS_DIR / "e1_loudness_probe.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    summary = "; ".join(f"{ck.replace('best_model_', '').replace('.pt', '')}: "
                        f"P>=0.5 from RMS {r['rms_at_first_p_over_0.5']}"
                        for ck, r in results.items())
    record_run("e1_loudness_probe",
               {"ckpts": list(results), "n": args.n, "seed": args.seed,
                "rms_levels": RMS_LEVELS, "dc_levels": DC_LEVELS, "quiet_rms": QUIET_RMS},
               results, summary)


if __name__ == "__main__":
    main()
