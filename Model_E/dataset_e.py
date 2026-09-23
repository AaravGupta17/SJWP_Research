"""
dataset_e.py — Model E: synthesiser fixes targeting the diagnosed failures
===========================================================================
Subclass of Model C's LeakDataset (model_C/dataset_c.py). Hydraulic
inputs, labels, Torricelli amplitude, material table and fixed-scale
normalisation are unchanged, so any difference from Model C comes from
the changes below. Each change is a switch (see REALISM_DEFAULTS), so its
effect can be measured separately.

  [E1] delay        Physical propagation: each channel receives the same
                    source with its own FRACTIONAL, non-circular delay.
                    Removes Model C's zero-delay "shared" component (60% of
                    the leak signal), which made the synthetic time delay
                    unrecoverable (cross-correlation vs true TDOA r = 0.00;
                    r = 1.00 without it). Removes the circular np.roll.
  [E2] background   The no-leak class gets diverse textures, not only the
                    Mendeley Branched hydrophone noise bank: coloured
                    Gaussian noise with random spectral slope and random
                    inter-channel coherence. Target: E7 — the C/D models
                    flag unfamiliar noise textures as leaks.
  [E3] interferers  Non-leak sounds in BOTH classes: short band-limited
                    bursts (water use, valves) arriving with a random
                    delay, and pump-like harmonic hum. A leak is steady and
                    broadband; these are not.
  [E4] sensor       With probability 0.5, an accelerometer-like response:
                    a 2nd-order high-pass tilt (random corner 50–300 Hz)
                    on the whole window, RMS preserved. Training noise is
                    hydrophone data; the real test uses accelerometers.
  [E5] gain         Whole-window gain jitter of ±6 dB in both classes, so
                    absolute loudness is a weaker cue.
  [E6] snr          Leak SNR range extended from 0.5–12 dB to −10–12 dB.
  [E7] no_dc        The pressure-proportional DC offset is not added.
  [E8] labels       Leak rows with no valid leak distance (which Model C
                    labelled "leak" but synthesised as pure noise) are
                    dropped.

Material acoustic parameters are deliberately left as in Model C. They
should be checked against the literature (Hunaidi & Chu 1999; Gao et al.
2004/2005) before being changed, and changed as a separate step.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_C"))
import dataset_c as C                                   # noqa: E402

REALISM_DEFAULTS = dict(delay=True, background=True, interferers=True, sensor=True,
                        gain=True, snr=True, no_dc=True)
SNR_DB_MIN_E = -10.0
SNR_DB_MAX_E = 12.0


def fractional_delay(src: np.ndarray, delay: float, T: int, pad: int) -> np.ndarray:
    """Delay a source buffer of length T + 2*pad by `delay` samples (may be
    fractional) and return the central T samples. The FFT shift is circular
    on the long buffer, but with delay < pad the wrapped part never reaches
    the returned segment, so the result is a true (linear) delay."""
    n = len(src)
    f = np.fft.rfftfreq(n)
    out = np.fft.irfft(np.fft.rfft(src) * np.exp(-2j * np.pi * f * delay), n=n)
    return out[pad:pad + T]


def coloured_noise(T: int, rng=np.random) -> np.ndarray:
    """Two-channel noise with random spectral slope and inter-channel
    coherence, unit RMS per channel."""
    slope = rng.uniform(0.0, 2.0)                       # 0 = white, 1 = pink, 2 = brown
    rho = rng.uniform(0.0, 0.9)                         # inter-channel coherence
    f = np.fft.rfftfreq(T)
    f[0] = f[1]
    shape = f ** (-slope / 2)
    common = np.fft.irfft(np.fft.rfft(rng.standard_normal(T)) * shape, n=T)
    out = []
    for _ in range(2):
        own = np.fft.irfft(np.fft.rfft(rng.standard_normal(T)) * shape, n=T)
        ch = np.sqrt(rho) * common + np.sqrt(1 - rho) * own
        out.append(ch / (ch.std() + 1e-12))
    return np.stack(out)


def hp_tilt(x: np.ndarray, fs: int, corner: float) -> np.ndarray:
    """Accelerometer-like 2nd-order high-pass, RMS of each channel preserved."""
    sos = butter(2, corner / (fs / 2), btype="high", output="sos")
    y = sosfilt(sos, x, axis=-1)
    r_in = np.sqrt((x ** 2).mean(axis=-1, keepdims=True))
    r_out = np.sqrt((y ** 2).mean(axis=-1, keepdims=True)) + 1e-12
    return y * r_in / r_out


class LeakDatasetE(C.LeakDataset):

    def __init__(self, index_csv: str, realism: dict = None, **kw):
        kw.setdefault("include_pressure_dc", False)
        super().__init__(index_csv, **kw)
        self.realism = {**REALISM_DEFAULTS, **(realism or {})}
        if not self.realism["no_dc"]:
            self.include_pressure_dc = True
        # [E8] drop leak rows that have no leak source to synthesise
        before = len(self._valid_idx)
        self._valid_idx = [i for i in self._valid_idx
                           if not (self._cache[i]["leak_status"] == 1
                                   and self._cache[i]["d_left"] <= 0
                                   and self._cache[i]["d_right"] <= 0)]
        print(f"  [E8] dropped {before - len(self._valid_idx)} leak rows without a leak source")

    # ── background ──────────────────────────────────────────────────────────
    def _background(self, c: dict) -> np.ndarray:
        T = self.signal_length
        scale = 1.0 + abs(c["flow_velocity"]) * 0.1 + c["demand"] * 0.05
        use_bank = (not self.realism["background"]) or np.random.rand() < 0.4
        w = self.noise_bank.get_window() if use_bank else None
        if w is None:
            w = coloured_noise(T) * self.noise_bank.noise_rms
        return w.astype(np.float64) * scale

    def _interferers(self, result: np.ndarray, c: dict):
        T, fs = self.signal_length, self.fs
        noise_rms = self.noise_bank.noise_rms
        t = np.arange(T) / fs
        if np.random.rand() < 0.5:                      # transient burst (water use, valve)
            centre = np.random.uniform(100, 2000)
            bw = np.random.uniform(50, 800)
            lo, hi = max(centre - bw / 2, 20), min(centre + bw / 2, fs / 2 - 50)
            sos = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
            dur = int(np.random.uniform(0.02, 0.2) * fs)
            start = np.random.randint(0, T - dur)
            env = np.zeros(T)
            env[start:start + dur] = np.hanning(dur)
            pad = T
            burst = np.zeros(T + 2 * pad)
            burst[pad:pad + T] = sosfilt(sos, np.random.randn(T)) * env
            burst /= np.sqrt((burst ** 2).mean() * (T + 2 * pad) / max(dur, 1)) + 1e-12
            amp = noise_rms * 10 ** (np.random.uniform(-5, 15) / 20)
            span = (c["d_left"] + c["d_right"]) or c["pipe_length"]
            max_lag = min(span / max(c["wave_speed"], 100.0) * fs, pad - 1)
            lag = np.random.uniform(-max_lag, max_lag)
            result[0] += amp * fractional_delay(burst, max(lag, 0), T, pad)
            result[1] += amp * fractional_delay(burst, max(-lag, 0), T, pad)
        if np.random.rand() < 0.3:                      # pump-like harmonic hum
            f0 = np.random.uniform(25, 60)
            hum = sum(np.sin(2 * np.pi * k * f0 * t + np.random.uniform(0, 2 * np.pi)) / k
                      for k in (1, 2, 3))
            hum = hum / (hum.std() + 1e-12) * noise_rms * 10 ** (np.random.uniform(-10, 10) / 20)
            result[0] += hum
            result[1] += np.roll(hum, np.random.randint(0, 5))

    def _leak(self, result: np.ndarray, c: dict):
        T, fs = self.signal_length, self.fs
        centre, bandwidth, damping = C.MATERIAL_ACOUSTIC.get(c["pipe_material"], C.DEFAULT_ACOUSTIC)
        lo_db, hi_db = ((SNR_DB_MIN_E, SNR_DB_MAX_E) if self.realism["snr"]
                        else (C.SNR_DB_MIN, C.SNR_DB_MAX))
        if self.snr_override_db is not None:
            snr_db = self.snr_override_db
        else:
            norm = np.clip(c["torricelli_amp"] * 5000.0, 1e-4, 1.0)
            snr_db = np.clip(lo_db + (hi_db - lo_db) * norm + np.random.uniform(-1.5, 1.5),
                             lo_db, hi_db)
        amp = max(self.noise_bank.noise_rms * 10 ** (snr_db / 20.0), 1e-7)
        eff_speed = c["wave_speed"] * np.random.uniform(0.85, 1.15)
        eff_alpha = c["alpha"] * np.random.uniform(0.9, 1.1)
        rough = np.clip(c["pipe_roughness"] / 100.0, 0.5, 3.0)

        if not self.realism["delay"]:                   # Model C propagation (for ablation)
            src = C.generate_leak_source_pink(centre, bandwidth, amp, T, fs)
            shared = src * 0.6
            for ch, d in ((0, c["d_left"]), (1, c["d_right"])):
                if d > 0:
                    delay = int(np.clip(d / eff_speed * fs, 0, T - 1))
                    att = np.exp(-eff_alpha * damping * rough * d)
                    result[ch] += np.roll(src, delay) * att * 0.4 + shared * att
            return

        pad = T
        src = C.generate_leak_source_pink(centre, bandwidth, amp, T + 2 * pad, fs).astype(np.float64)
        for ch, d in ((0, c["d_left"]), (1, c["d_right"])):
            if d > 0:
                delay = min(d / eff_speed * fs, pad - 1)
                att = np.exp(-eff_alpha * damping * rough * d)
                result[ch] += fractional_delay(src, delay, T, pad) * att

    def generate_signal(self, c: dict) -> np.ndarray:
        result = self._background(c)
        if self.include_pressure_dc:
            result[0] += c["p_left"] * 3e-4
            result[1] += c["p_right"] * 3e-4
        if self.realism["interferers"]:
            self._interferers(result, c)
        if c["leak_status"] == 1 and (c["d_left"] > 0 or c["d_right"] > 0):
            self._leak(result, c)
        if self.realism["sensor"] and np.random.rand() < 0.5:
            result = hp_tilt(result, self.fs, np.random.uniform(50, 300))
        if self.realism["gain"]:
            result = result * 10 ** (np.random.uniform(-6, 6) / 20)
        return self._normalize(result.astype(np.float32))
