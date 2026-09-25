"""
augment_f.py — Model F: build one training window from its parts
==================================================================
Model F targets the failure causes found in E1-E10 (docs/MODEL_F_PREREG.md):

  [F1] backgrounds  Fresh background for every sample, mostly REAL no-leak
                    recordings from several sources (Hong Kong, Dongguan,
                    Mendeley Branched), plus coloured and white noise.
                    Models C-E drew every background from 6 Mendeley files,
                    and E7 shows they flag any other texture as a leak.
  [F2] loudness     Joint per-window z-score (both channels by one factor,
                    so the ratio between sensors survives). Absolute
                    loudness was a shortcut (E1, E2) and sensor gains differ
                    between sources.
  [F3] band         Everything is band-limited to 2 kHz, as the real public
                    data is (experiments/public_data.py), so bandwidth cannot
                    tell synthetic from real rows.
  [F4] colouration  Random EQ (spectral tilt and resonances) on every sample,
                    in both classes: each sensor and pipe colours sound
                    differently (E9: sources do not transfer).
  [F5] coherence    Two-channel backgrounds with random coherence and lag,
                    so "the channels are correlated" is not a leak cue.

All functions are numpy-only and take an explicit Generator, so the same
code builds training samples (random) and the fixed validation set (seeded).
"""

import numpy as np
from scipy.signal import butter, sosfreqz

FS = 5000
T = 2000
BAND_HZ = 2000.0
CLIP = 5.0
_SOS = butter(8, BAND_HZ / (FS / 2), btype="low", output="sos")   # = public_data.to_common
# zero-phase (forward-backward) filtering multiplies the spectrum by |H|^2;
# applied by FFT, which is ~20x faster than sosfiltfilt on short windows
_H2 = np.abs(sosfreqz(_SOS, worN=np.fft.rfftfreq(T, 1.0 / FS), fs=FS)[1]) ** 2


def band_limit(x: np.ndarray) -> np.ndarray:
    """Same zero-phase 2 kHz low-pass response the real public data went through."""
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * _H2, n=x.shape[-1], axis=-1)


def band_pass_noise(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    """White noise restricted to [lo, hi] Hz with raised-cosine edges."""
    f = np.fft.rfftfreq(n, 1.0 / FS)
    edge = max((hi - lo) * 0.2, 10.0)
    mask = np.clip(np.minimum(f - lo + edge, hi + edge - f) / edge, 0, 1)
    mask = 0.5 - 0.5 * np.cos(np.pi * mask)
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * mask, n=n)


def joint_zscore(x: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """(2, T) -> zero mean, unit RMS over both channels, clipped.
    Identical to experiments/mendeley.py zscore_windows (E4 'zscore')."""
    x = x - x.mean()
    return np.clip(x / (x.std() + 1e-8), -clip, clip).astype(np.float32)


def unit_rms(x: np.ndarray) -> np.ndarray:
    return x / (np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-12)


def coloured_noise(rng: np.random.Generator, n: int = T) -> np.ndarray:
    """One channel, random spectral slope (white .. brown), unit RMS."""
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(rng.standard_normal(n)) * f ** (-rng.uniform(0.0, 2.0) / 2)
    return unit_rms(np.fft.irfft(spec, n=n))


def pair_channels(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                  max_lag: int = 100) -> np.ndarray:
    """Two-sensor background from single-sensor windows a, b (unit RMS).
    ch2 shares a delayed copy of ch1 with random coherence rho, so channel
    correlation occurs in no-leak data too [F5]."""
    rho = rng.uniform(0.0, 0.95)
    lag = int(rng.integers(-max_lag, max_lag + 1))
    ch2 = np.sqrt(rho) * np.roll(a, lag) + np.sqrt(1 - rho) * b
    return np.stack([a, unit_rms(ch2)])


def random_eq(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply the spectrum by a random smooth curve [F4]: a tilt of
    -6..+6 dB/octave around 500 Hz and 0-2 resonances of -9..+9 dB, shared
    by both channels, plus a small per-channel tilt (+-1.5 dB/octave)."""
    n = x.shape[-1]
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / FS), 10.0)
    octv = np.log2(f / 500.0)
    db = rng.uniform(-6, 6) * octv
    for _ in range(rng.integers(0, 3)):
        centre = rng.uniform(np.log2(50 / 500), np.log2(BAND_HZ / 500))
        width = rng.uniform(0.2, 1.0)
        db = db + rng.uniform(-9, 9) * np.exp(-0.5 * ((octv - centre) / width) ** 2)
    out = np.empty_like(x, dtype=np.float64)
    for ch in range(x.shape[0]):
        g = 10 ** ((db + rng.uniform(-1.5, 1.5) * octv) / 20)
        out[ch] = np.fft.irfft(np.fft.rfft(x[ch]) * g, n=n)
    return out


def interferers(rng: np.random.Generator, n: int = T) -> np.ndarray:
    """Non-leak sounds for BOTH classes, in background-RMS units (as Model E
    [E3]): a band-limited burst (water use, valve) reaching the sensors with a
    random lag, and pump-like harmonic hum."""
    out = np.zeros((2, n))
    if rng.random() < 0.4:
        centre, bw = rng.uniform(100, 1900), rng.uniform(50, 800)
        lo, hi = max(centre - bw / 2, 20), min(centre + bw / 2, BAND_HZ)
        dur = int(rng.uniform(0.02, 0.2) * FS)
        start = int(rng.integers(0, n - dur))
        burst = np.zeros(n)
        burst[start:start + dur] = band_pass_noise(rng, n, lo, hi)[:dur] * np.hanning(dur)
        burst = burst / (np.sqrt(np.mean(burst[start:start + dur] ** 2)) + 1e-12)
        amp = 10 ** (rng.uniform(-5, 15) / 20)
        lag = int(rng.integers(-100, 101))
        out[0] += amp * burst
        out[1] += amp * np.roll(burst, lag)
    if rng.random() < 0.3:
        t = np.arange(n) / FS
        f0 = rng.uniform(25, 60)
        hum = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3))
        hum = unit_rms(hum) * 10 ** (rng.uniform(-10, 10) / 20)
        out[0] += hum
        out[1] += np.roll(hum, int(rng.integers(0, 5)))
    return out


def finish(x: np.ndarray, rng: np.random.Generator, eq: bool = True) -> np.ndarray:
    """Last steps shared by every sample, synthetic or real: EQ, z-score."""
    if eq:
        x = random_eq(x, rng)
    return joint_zscore(x)
