"""
texture_probe.py — E7: does the model call every unfamiliar noise a leak?
===========================================================================
Hypothesis (from E1 + E4): during synthetic training the no-leak class
was ONE noise texture — windows of the Mendeley Branched no-leak
hydrophone recordings. The network may therefore have learned "sounds
like that noise bank = no leak; anything else = leak", which would flag
every real recording made with a different sensor or on a different pipe.

Test: feed each checkpoint leak-free inputs from several sources, every
window rescaled to the SAME RMS, so loudness cannot explain differences:

  bank            the training noise bank itself (hydrophone, Branched)
  bank_phase      same windows, phases randomised identically in both
                  channels: spectrum and inter-channel phase kept, waveform
                  structure (transients, non-Gaussianity) destroyed
  bank_phase_ind  phases randomised independently per channel: spectrum
                  kept, inter-channel coherence destroyed too
  hyd_looped      hydrophone, Looped no-leak (same sensor, other pipe)
  acc_branched    accelerometer, Branched no-leak (other sensor, same runs)
  acc_looped      accelerometer, Looped no-leak (other sensor, other pipe)
  gaussian        AR(1)-coloured Gaussian noise (as in E1)

Sanity row: "bank_training_scale" applies the exact training
preprocessing (per-file max normalisation, divide by 10 x noise RMS, no
per-window rescale). It should score ~0; if not, this reproduction of the
training pipeline is wrong and the other rows can't be trusted.

Reading: if bank ~0 but hyd_looped / acc_* / gaussian ~1 at the same RMS,
the model keys on noise texture, and the fix is a diverse no-leak class
(Model E), not input scaling.

    python experiments/texture_probe.py
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

import mendeley as M
from _common import (DATASETS, MODELS_DIR, REPO_ROOT, load_model, predict_proba,
                     record_run)

DEFAULT_CKPTS = ["best_model_c_v4.pt", "best_model_c_seed42.pt", "best_model_d.pt"]
TARGET_RMS = [0.05, 0.1, 0.2]


def rescale_rms(w: np.ndarray, target: float) -> np.ndarray:
    r = np.sqrt((w.astype(np.float64) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    return (w / r * target).astype(np.float32)


def phase_randomise(w: np.ndarray, rng, shared: bool) -> np.ndarray:
    X = np.fft.rfft(w.astype(np.float64), axis=-1)
    shape = (X.shape[0], 1, X.shape[2]) if shared else X.shape
    ph = np.exp(2j * np.pi * rng.random(shape))
    ph[..., 0] = 1.0                                   # DC and Nyquist bins must stay real
    if w.shape[-1] % 2 == 0:
        ph[..., -1] = 1.0
    return np.fft.irfft(X * ph, n=w.shape[-1], axis=-1).astype(np.float32)


def pick(w: np.ndarray, n: int, rng) -> np.ndarray:
    return w if len(w) <= n else w[np.sort(rng.choice(len(w), n, replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyd-root", type=Path, default=DATASETS / "Hydrophone" / "Hydrophone")
    ap.add_argument("--acc-root", type=Path, default=DATASETS / "Accelerometer" / "Accelerometer")
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--n", type=int, default=300, help="windows per source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cache = REPO_ROOT / "cache_mendeley"

    # hydrophones with the training preprocessing (per-file max normalisation)
    hyd = M.load_windowset(args.hyd_root, "hydrophone", file_norm="max", cache_dir=cache,
                           verbose=False)
    bank = hyd.x[(hyd.topology == "Branched") & (hyd.y == 0)]
    noise_rms = float(np.sqrt((bank.astype(np.float64) ** 2).mean(axis=(1, 2))).mean())
    ref = M.TRAIN_REF_MULTIPLIER * noise_rms            # dataset_c: noise_rms * 10
    print(f"noise bank: {len(bank)} windows, noise RMS {noise_rms:.5g} (training ref {ref:.5g})")

    sources = {
        "bank": pick(bank, args.n, rng),
        "hyd_looped": pick(hyd.x[(hyd.topology == "Looped") & (hyd.y == 0)], args.n, rng),
    }
    sources["bank_phase"] = phase_randomise(sources["bank"], rng, shared=True)
    sources["bank_phase_ind"] = phase_randomise(sources["bank"], rng, shared=False)
    if args.acc_root.exists():
        acc = M.load_windowset(args.acc_root, "accelerometer", cache_dir=cache, verbose=False)
        sources["acc_branched"] = pick(acc.x[(acc.topology == "Branched") & (acc.y == 0)], args.n, rng)
        sources["acc_looped"] = pick(acc.x[(acc.topology == "Looped") & (acc.y == 0)], args.n, rng)
    g = lfilter([1.0], [1.0, -0.95], rng.standard_normal((args.n, 2, 2000)), axis=-1)
    sources["gaussian"] = g.astype(np.float32)

    results = {"noise_rms": noise_rms, "ref": ref, "target_rms": TARGET_RMS, "models": {}}
    for ck in args.ckpts:
        if not (MODELS_DIR / ck).exists():
            print(f"skip {ck}: not found")
            continue
        model, _ = load_model(ck)
        res = {}
        x = M.fixed_scale_windows(sources["bank"], ref)
        p = predict_proba(model, x, logits=True)
        res["bank_training_scale"] = {"model_rms": float(np.sqrt((x ** 2).mean())),
                                      "flag_rate": float((p >= 0).mean())}
        for name, w in sources.items():
            res[name] = {}
            for t in TARGET_RMS:
                p = predict_proba(model, rescale_rms(w, t), logits=True)
                res[name][str(t)] = float((p >= 0).mean())     # fraction called "leak"
        results["models"][ck] = res

        s = res["bank_training_scale"]
        print(f"\n{ck}  sanity: training-scale bank (RMS {s['model_rms']:.3f}) flagged "
              f"{s['flag_rate']:.0%}  (should be ~0%)")
        print(f"  {'source':16s} " + " ".join(f"RMS {t:<5g}" for t in TARGET_RMS) + "   (fraction flagged as leak)")
        for name in sources:
            print(f"  {name:16s} " + " ".join(f"{res[name][str(t)]:>8.0%} " for t in TARGET_RMS))

    ck0 = next(iter(results["models"]), None)
    summary = ""
    if ck0:
        r = results["models"][ck0]
        summary = (f"{ck0}: sanity {r['bank_training_scale']['flag_rate']:.0%}; flagged @RMS0.1 " +
                   ", ".join(f"{k} {v['0.1']:.0%}" for k, v in r.items() if k != "bank_training_scale"))
    record_run("e7_texture_probe", {**vars(args), "hyd_root": str(args.hyd_root),
                                    "acc_root": str(args.acc_root)}, results, summary)


if __name__ == "__main__":
    main()
