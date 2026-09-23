"""
sheffield_calibration.py — E10: measure plastic-pipe leak acoustics to calibrate the synthesiser
==================================================================================================
Uses the University of Sheffield CID-lab data (Shekofteh 2026, CC BY 4.0;
scripts/download_public_data.py sheffield): a 63 mm MDPE pipe at 2.8–4.2 bar,
calibrated accelerometers (m/s^2) at 8192 Hz.

A. Attenuation (All_Data). Each leak test has an accelerometer at the leak
   (Acc0) and at 1, 2, ... 50 m (AccD = D metres). The matching no-leak
   background recording gives the noise at each position. For each frequency
   band, the leak's EXCESS power (leak PSD minus background PSD) is fitted
   against distance -> attenuation in dB per metre. Only points with excess
   >= 3 dB above background are used.

B. Leak spectrum at the leak: shape of the excess PSD at 0 m (peak and
   centroid frequency, -10 dB band).

C. Wave speed (Coherence folder). Simultaneous recordings: accelerometer 1 at
   the leak, accelerometer 2 at k metres (test#k). The delay between them is
   estimated by cross-correlation restricted to the coherent band; the wave
   speed is the slope of distance vs delay.

These are the quantities the Model C/D synthesiser sets by hand
(MATERIAL_ACOUSTIC centre frequency / bandwidth / damping, EPANET attenuation,
wave speed). The script prints Model C's PVC values next to the
measurements.

    python experiments/sheffield_calibration.py
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, correlate, sosfiltfilt, welch

from _common import DATASETS, PLOTS_DIR, REPO_ROOT, record_run

ROOT = DATASETS / "public" / "sheffield" / "extracted" / "Acoustic Data (Leakage Experiments)"
FS = 8192
BANDS = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200)]
MIN_EXCESS_DB = 3.0

# test folder -> background file (ReadME: NoLeak_2.8bar "is used for all tests")
BACKGROUND = {
    "(test#1) 1mmValve_4.2bar": "(test#1) 1mmValve_4.2bar/NoLeak_4.2bar.csv",
    "(test#2) 1mmValve_2.8bar": "(test#2) 1mmValve_2.8bar/NoLeak_2.8bar.csv",
    "(test#3) 2mmValve_2.8bar": "(test#3) 2mmValve_2.8bar/noLeak_2.8bar_newer.csv",
    "(test#4) 3mmValve_2.8bar": "(test#4) 3mmValve_2.8bar/noLeak.csv",
}
DEFAULT_BACKGROUND = "(test#5) 1mmDirect_2.8bar/NoLeak_2.8bar.csv"


def psd(x):
    f, p = welch(np.asarray(x, np.float64), fs=FS, nperseg=4096)
    return f, p


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return p[m].sum() * (f[1] - f[0])


def distances(cols):
    return {c: int(c[3:]) for c in cols if re.fullmatch(r"Acc\d+", c)}


def attenuation(all_data: Path):
    rows = []
    bg_cache = {}
    spectra_at_leak = {}
    for test_dir in sorted(all_data.iterdir()):
        if not test_dir.is_dir():
            continue
        bg_rel = BACKGROUND.get(test_dir.name, DEFAULT_BACKGROUND)
        if bg_rel not in bg_cache:
            bg = pd.read_csv(all_data / bg_rel)
            bg_cache[bg_rel] = {d: psd(bg[c]) for c, d in distances(bg.columns).items()}
        bg_psd = bg_cache[bg_rel]
        for leak_file in sorted(test_dir.glob("*.csv")):
            if "noleak" in leak_file.stem.lower():
                continue
            df = pd.read_csv(leak_file)
            for col, d in distances(df.columns).items():
                f, p = psd(df[col])
                # nearest background position (tests 3/4 only have 0 m and 10 m)
                bd = min(bg_psd, key=lambda k: abs(k - d))
                _, pb = bg_psd[bd]
                if d == 0:
                    spectra_at_leak[f"{test_dir.name}/{leak_file.stem}"] = (f, np.maximum(p - pb, 0))
                for lo, hi in BANDS:
                    pl, pn = band_power(f, p, lo, hi), band_power(f, pb, lo, hi)
                    excess_db = 10 * np.log10(max(pl - pn, 1e-30) / pn) if pl > pn else -np.inf
                    rows.append({"test": test_dir.name, "file": leak_file.stem, "distance_m": d,
                                 "band": f"{lo}-{hi}", "leak_db": 10 * np.log10(max(pl - pn, 1e-30)),
                                 "excess_over_noise_db": excess_db})
    df = pd.DataFrame(rows)
    fits = []
    for (test, band), g in df.groupby(["test", "band"]):
        g = g[g["excess_over_noise_db"] >= MIN_EXCESS_DB]
        if g["distance_m"].nunique() >= 3:
            slope, _ = np.polyfit(g["distance_m"], g["leak_db"], 1)
            fits.append({"test": test, "band": band, "db_per_m": -slope,
                         "n_points": len(g), "max_distance_m": int(g["distance_m"].max())})
    return df, pd.DataFrame(fits), spectra_at_leak


def wave_speed(coh_root: Path):
    rows = []
    for leak_dir in sorted(p for p in coh_root.iterdir() if p.is_dir()):
        for tdir in sorted(leak_dir.glob("test#*"), key=lambda p: int(p.name.split("#")[1])):
            k = int(tdir.name.split("#")[1])                 # accelerometer 2 at k metres
            for f in sorted(tdir.glob("*.csv")):
                df = pd.read_csv(f)
                acc = [c for c in df.columns if c.startswith("Accelerometer")]
                a, b = df[acc[0]].to_numpy(float), df[acc[1]].to_numpy(float)
                fc, cxy = coherence(a, b, fs=FS, nperseg=2048)
                band = (fc >= 20) & (fc <= 1000)
                good = fc[band][cxy[band] >= 0.5]
                lo, hi = (good.min(), good.max()) if len(good) >= 3 else (20.0, 400.0)
                hi = max(hi, lo + 20)
                sos = butter(4, [lo / (FS / 2), min(hi, FS / 2 - 1) / (FS / 2)], btype="band", output="sos")
                af, bf = sosfiltfilt(sos, a), sosfiltfilt(sos, b)
                xc = correlate(bf, af, mode="full", method="fft")
                lags = np.arange(-len(a) + 1, len(a))
                win = np.abs(lags) <= int(0.05 * FS)            # |delay| <= 50 ms
                lag = lags[win][np.argmax(np.abs(xc[win]))]
                rows.append({"leak": leak_dir.name, "distance_m": k, "file": f.stem,
                             "delay_ms": 1000 * lag / FS, "coherent_band_hz": [float(lo), float(hi)],
                             "mean_coherence": float(cxy[band].mean())})
    df = pd.DataFrame(rows)
    fits = {}
    for leak, g in df.groupby("leak"):
        g = g[g["delay_ms"] > 0]
        if len(g) >= 3:
            slope = np.linalg.lstsq(g[["delay_ms"]].to_numpy() / 1000, g["distance_m"].to_numpy(),
                                    rcond=None)[0][0]
            resid = g["distance_m"] - slope * g["delay_ms"] / 1000
            fits[leak] = {"wave_speed_mps": float(slope), "n_tests": int(len(g)),
                          "rms_residual_m": float(np.sqrt((resid ** 2).mean()))}
    return df, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    if not args.root.exists():
        sys.exit(f"Sheffield data not found at {args.root}\nRun: python scripts/download_public_data.py sheffield")

    print("A. attenuation along the pipe ...")
    raw, fits, spectra = attenuation(args.root / "All_Data")
    summary_att = fits.groupby("band")["db_per_m"].agg(["median", "min", "max", "count"])
    summary_att = summary_att.reindex([f"{lo}-{hi}" for lo, hi in BANDS]).dropna()
    print(summary_att.round(2).to_string())

    print("\nB. leak spectrum at the leak (excess over background, 0 m)")
    peaks, cents = [], []
    for name, (f, p) in spectra.items():
        m = (f >= 10) & (f <= 3200)
        peaks.append(float(f[m][np.argmax(p[m])]))
        cents.append(float((f[m] * p[m]).sum() / (p[m].sum() + 1e-30)))
    print(f"  peak frequency   median {np.median(peaks):.0f} Hz (range {min(peaks):.0f}–{max(peaks):.0f})")
    print(f"  centroid         median {np.median(cents):.0f} Hz (range {min(cents):.0f}–{max(cents):.0f})")

    print("\nC. wave speed from simultaneous recordings ...")
    coh, speed = wave_speed(args.root / "Coherence (simultaneous recording)")
    for leak, r in speed.items():
        print(f"  {leak:22s} c = {r['wave_speed_mps']:.0f} m/s  ({r['n_tests']} distances, "
              f"rms residual {r['rms_residual_m']:.2f} m)")

    sys.path.insert(0, str(REPO_ROOT / "model_C"))
    import dataset_c as C
    cf, bw, damp = C.MATERIAL_ACOUSTIC["PVC"]
    print(f"\nModel C 'PVC' for comparison: leak band {cf - bw / 2:.0f}–{cf + bw / 2:.0f} Hz "
          f"(centre {cf:.0f}); attenuation = alpha x {damp} x roughness factor per metre "
          f"= {8.686 * 0.001 * damp:.3f} dB/m at alpha=0.001/m (roughness factor 1)")

    # ── figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for band, g in raw.groupby("band"):
        g = g[g["excess_over_noise_db"] >= MIN_EXCESS_DB].groupby("distance_m")["leak_db"].median()
        if len(g) >= 3:
            ax[0].plot(g.index, g.values, "o-", ms=3, label=f"{band} Hz")
    ax[0].set_xlabel("Distance from leak (m)"); ax[0].set_ylabel("Leak power above background (dB)")
    ax[0].set_title("A. Attenuation along MDPE pipe"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    for name, (f, p) in list(spectra.items())[:8]:
        m = (f >= 10) & (f <= 3200)
        ax[1].semilogy(f[m], p[m] / p[m].max(), lw=1, label=name.split("/")[0])
    ax[1].axvspan(cf - bw / 2, cf + bw / 2, color="red", alpha=0.15, label="Model C 'PVC' band")
    ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_title("B. Leak spectrum at the leak (normalised)")
    ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3)
    for leak, g in coh.groupby("leak"):
        ax[2].plot(g["delay_ms"], g["distance_m"], "o", label=leak)
        if leak in speed:
            t = np.linspace(0, g["delay_ms"].max(), 10)
            ax[2].plot(t, speed[leak]["wave_speed_mps"] * t / 1000, "--")
    ax[2].set_xlabel("Measured delay (ms)"); ax[2].set_ylabel("Sensor distance (m)")
    ax[2].set_title("C. Wave speed from simultaneous sensors"); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "e10_sheffield_calibration.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    results = {
        "attenuation_db_per_m_by_band": summary_att.reset_index().to_dict(orient="records"),
        "attenuation_fits": fits.to_dict(orient="records"),
        "leak_spectrum_peak_hz": peaks, "leak_spectrum_centroid_hz": cents,
        "wave_speed": speed, "coherence_delays": coh.to_dict(orient="records"),
        "model_c_pvc": {"centre_hz": cf, "bandwidth_hz": bw, "damping": damp},
    }
    s = ("atten dB/m (median by band): " +
         ", ".join(f"{b} {v:.2f}" for b, v in summary_att["median"].items()) +
         f"; leak peak {np.median(peaks):.0f} Hz; c " +
         ", ".join(f"{k} {v['wave_speed_mps']:.0f} m/s" for k, v in speed.items()))
    record_run("e10_sheffield_calibration", {"root": str(args.root)}, results, s)


if __name__ == "__main__":
    main()
