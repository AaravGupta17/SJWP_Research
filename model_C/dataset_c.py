"""
dataset_c.py — Model C: Sim-to-Real Gap Reduction
==================================================
Identical to dataset.py (Model B) with FOUR changes
targeting the sim-to-real transfer gap identified in Section 5.3.

Changes vs Model B (dataset.py):
  [C1] SNR CALIBRATION — amplitude derived deterministically from the
       Torricelli orifice flow rate, mapped to SNR range [0.5, 12.0] dB
       above the empirical noise floor. Larger leaks (higher pressure,
       larger orifice area) produce proportionally stronger signals.
       Replaces arbitrary x50.0 scalar from Model B. Random SNR was
       avoided as it destroys the amplitude-flow correlation needed for
       severity estimation (verified: corr=-0.002 with random SNR).

  [C2] PINK NOISE SOURCE — leak source generated as bandpass-filtered
       pink noise (power proportional to 1/f). Replaces Gaussian-modulated
       sinusoid. Physically motivated: turbulent flow produces broadband
       pressure fluctuations, not clean sinusoidal pulses.

  [C3] CORRELATED CHANNEL LEAK SIGNAL — shared structural vibration
       component added to both channels before independent TDOA propagation.
       Models physical reality that both sensors on same pipe share
       common-mode vibration from leak source (alpha=0.6).

  [C4] FIXED-SCALE NORMALISATION — both channels divided by a fixed
       reference scale (10x empirical noise floor RMS) rather than
       per-sample z-score. Preserves relative amplitude between samples,
       which encodes attenuation (severity) and SNR (leak size).
       Per-sample z-score in Model B forced every signal to unit variance,
       destroying amplitude information needed for severity estimation.

Everything else is IDENTICAL to Model B:
  - Same Mendeley Branched no-leak noise background
  - Same EPANET scalar extraction
  - Same Torricelli amplitude physics (used as SNR seed, not direct amp)
  - Same wave speed perturbation (+-15%)
  - Same attenuation model
  - Same augmentation
  - Scalars zeroed at training time (deployment realistic)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import lfilter, resample_poly, butter, sosfilt
from pathlib import Path
from math import gcd
from tqdm import tqdm

# ── Mendeley noise configuration ───────────────────────────────────────────────
MENDELEY_ROOT  = Path("../datasets/Hydrophone/Hydrophone")
MENDELEY_FS    = 8000
MODEL_FS       = 5000
SIGNAL_LENGTH  = 2000

TRAIN_NOISE_FILES = [
    "Branched/No-leak/BR_NL_0.18 LPS_N_H1.raw",
    "Branched/No-leak/BR_NL_0.18 LPS_N_H2.raw",
    "Branched/No-leak/BR_NL_0.47 LPS_N_H1.raw",
    "Branched/No-leak/BR_NL_0.47 LPS_N_H2.raw",
    "Branched/No-leak/BR_NL_ND_N_H1.raw",
    "Branched/No-leak/BR_NL_ND_N_H2.raw",
    "Branched/No-leak/BR_NL_ND_NN_H1.raw",
    "Branched/No-leak/BR_NL_ND_NN_H2.raw",
    "Branched/No-leak/BR_NL_Transient_N_H1.raw",
    "Branched/No-leak/BR_NL_Transient_N_H2.raw",
    "Branched/No-leak/BR_NL_Transient_NN_H1.raw",
    "Branched/No-leak/BR_NL_Transient_NN_H2.raw",
]

# Material acoustic properties — UNCHANGED from Model B
MATERIAL_ACOUSTIC = {
    "CI":    (150.0,  80.0,  1.3),
    "DI":    (280.0, 150.0,  0.9),
    "PVC":   (800.0, 300.0,  0.5),
    "STEEL": (500.0, 200.0,  0.7),
}
DEFAULT_ACOUSTIC = (200.0, 100.0, 1.0)

CD = 0.61
G  = 9.81

# ── [C1] SNR calibration range ────────────────────────────────────────────────
SNR_DB_MIN = 0.5    # Mendeley-hard lab conditions
SNR_DB_MAX = 12.0   # Strong municipal field signal


class MendeleyNoiseBank:
    """
    Loads Branched no-leak Mendeley recordings.
    Computes noise_rms for [C1] SNR calibration and [C4] fixed normalisation.
    """

    def __init__(self):
        self.windows   = []
        self.noise_rms = 0.01
        self._load()

    def _load(self):
        print("Loading Mendeley noise bank (Branched no-leak)...")
        h1_files = [f for f in TRAIN_NOISE_FILES if "H1" in f]
        loaded   = 0

        for h1_path in h1_files:
            h2_path = h1_path.replace("H1", "H2")
            try:
                h1_raw = self._load_raw(str(MENDELEY_ROOT / h1_path))
                h2_raw = self._load_raw(str(MENDELEY_ROOT / h2_path))
                h1     = self._resample(h1_raw)
                h2     = self._resample(h2_raw)
                n_windows = min(len(h1), len(h2)) // SIGNAL_LENGTH
                for i in range(n_windows):
                    s = i * SIGNAL_LENGTH
                    e = s + SIGNAL_LENGTH
                    w = np.stack([h1[s:e], h2[s:e]]).astype(np.float32)
                    self.windows.append(w)
                loaded += 1
            except Exception as e:
                print(f"  Warning: could not load {h1_path}: {e}")

        print(f"  Loaded {loaded} file pairs -> {len(self.windows)} noise windows")

        # [C1] + [C4] Empirical noise floor RMS
        if self.windows:
            all_rms = [np.sqrt(np.mean(w ** 2)) for w in self.windows]
            self.noise_rms = float(np.mean(all_rms))
            print(f"  Noise floor RMS (empirical): {self.noise_rms:.6f}")
        else:
            print("  WARNING: No Mendeley noise loaded — using fallback noise_rms=0.01")

    def _load_raw(self, path: str) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.int32).astype(np.float32)
        mx  = np.abs(raw).max()
        return raw / mx if mx > 0 else raw

    def _resample(self, sig: np.ndarray) -> np.ndarray:
        g    = gcd(MODEL_FS, MENDELEY_FS)
        up   = MODEL_FS    // g
        down = MENDELEY_FS // g
        return resample_poly(sig, up, down).astype(np.float32)

    def get_window(self) -> np.ndarray:
        if not self.windows:
            return None
        idx = np.random.randint(len(self.windows))
        return self.windows[idx].copy()

    def __len__(self):
        return len(self.windows)


_NOISE_BANK = None

def get_noise_bank() -> MendeleyNoiseBank:
    global _NOISE_BANK
    if _NOISE_BANK is None:
        _NOISE_BANK = MendeleyNoiseBank()
    return _NOISE_BANK


# ── [C2] Pink noise generator ─────────────────────────────────────────────────
def generate_pink_noise(n: int) -> np.ndarray:
    """Pink noise via frequency-domain shaping. Unit variance output."""
    white       = np.fft.rfft(np.random.randn(n))
    freqs       = np.fft.rfftfreq(n)
    freqs[0]    = freqs[1]
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_filter[0] = 0.0
    pink = np.fft.irfft(white * pink_filter, n=n)
    std  = pink.std()
    return (pink / std) if std > 1e-10 else pink


def generate_leak_source_pink(center_freq: float, bandwidth: float,
                               amplitude: float, n: int, fs: int) -> np.ndarray:
    """[C2] Bandpass-filtered pink noise as leak source signal."""
    low    = max(center_freq - bandwidth / 2.0, 10.0)
    high   = min(center_freq + bandwidth / 2.0, fs / 2.0 - 1.0)
    nyq    = fs / 2.0
    low_n  = np.clip(low  / nyq, 0.01, 0.99)
    high_n = np.clip(high / nyq, 0.01, 0.99)
    if high_n <= low_n:
        high_n = min(low_n + 0.05, 0.99)
    sos    = butter(4, [low_n, high_n], btype="band", output="sos")
    source = sosfilt(sos, generate_pink_noise(n))
    std    = source.std()
    if std > 1e-10:
        source = source / std
    return (source * amplitude).astype(np.float32)


class LeakDataset(Dataset):

    def __init__(self, index_csv: str,
                 signal_length: int = SIGNAL_LENGTH,
                 sampling_frequency: int = MODEL_FS,
                 augment: bool = False):
        self.signal_length = signal_length
        self.fs            = sampling_frequency
        self.augment       = augment
        self.noise_bank    = get_noise_bank()

        print(f"Loading {index_csv}")
        self.index_df   = pd.read_csv(index_csv)
        print(f"Caching {len(self.index_df):,} rows...")
        self._cache     = self._build_cache()
        self._valid_idx = [i for i, c in enumerate(self._cache) if c is not None]
        print(f"Ready: {len(self._valid_idx):,} samples | "
              f"fs={sampling_frequency}Hz | augment={augment}")

    def __len__(self):
        return len(self._valid_idx)

    def _build_cache(self) -> list:
        cache   = [None] * len(self.index_df)
        grouped = self.index_df.groupby("file_path", sort=False)
        with tqdm(total=len(self.index_df), desc="Caching",
                  unit="rows", dynamic_ncols=True) as pbar:
            for file_path, group in grouped:
                try:
                    file_df = pd.read_csv(file_path)
                    file_df.columns = file_df.columns.str.strip()
                    for idx, idx_row in group.iterrows():
                        try:
                            data_row   = file_df.iloc[int(idx_row["row_idx"])]
                            cache[idx] = self._extract(data_row, idx_row)
                        except Exception:
                            pass
                        pbar.update(1)
                except Exception:
                    pbar.update(len(group))
        valid = sum(1 for c in cache if c is not None)
        print(f"  {valid:,}/{len(cache):,} rows cached OK")
        return cache

    def _safe(self, value, default: float = 0.0) -> float:
        try:
            v = float(value)
            return default if (np.isnan(v) or np.isinf(v)) else v
        except Exception:
            return default

    def _parse_list(self, value) -> list:
        try:
            return [float(x.strip()) for x in str(value).split(",") if x.strip()]
        except Exception:
            return []

    def _extract(self, data_row, index_row) -> dict:
        s         = self._safe
        file_type = str(index_row["file_type"])

        pipe_length    = s(data_row.get("Pipe_Length_m"),                   100.0)
        pipe_diameter  = s(data_row.get("Pipe_Diameter_m"),                 0.15)
        pipe_roughness = s(data_row.get("Pipe_Roughness"),                  100.0)
        pipe_material  = str(data_row.get("Pipe_Material", "CI")).strip().upper()
        flow_velocity  = s(data_row.get("Avg_Flow_Velocity_mps"),           0.0)
        flow_rate      = s(data_row.get("Avg_Flow_Rate_lps"),               0.0)
        wave_speed     = s(data_row.get("Acoustic_Propagation_Speed_mps"),  1200.0)
        alpha          = s(data_row.get("Attenuation_Alpha_per_m"),         0.001)
        leak_status    = int(s(data_row.get("Leak_Status",                  0)))
        leak_area      = s(data_row.get("Leak_Area_m2",                     0.0))
        demand         = s(index_row["demand_multiplier"],                   1.0)

        sensor_pos   = self._parse_list(data_row.get("Sensor_Positions_m", ""))
        sensor_left  = sensor_pos[0] if len(sensor_pos) >= 1 else pipe_length / 3
        sensor_right = sensor_pos[1] if len(sensor_pos) >= 2 else 2 * pipe_length / 3

        if file_type == "leak":
            d_left  = s(data_row.get("Leak_Distance_Left_m"),  0.0)
            d_right = s(data_row.get("Leak_Distance_Right_m"), 0.0)
            if d_left == 0:
                d_left  = s(data_row.get("Leak_Distance_2nd_Left_m"),  0.0)
            if d_right == 0:
                d_right = s(data_row.get("Leak_Distance_2nd_Right_m"), 0.0)

            pressure  = s(data_row.get("Avg_Pressure_at_Leak"), 0.0)
            leak_flow = s(data_row.get("Leak_Flow_Lps"),         0.0)
            p_left    = (s(data_row.get("Pipe_Pressure_Left1_m",  0.0)) +
                         s(data_row.get("Pipe_Pressure_Left2_m",  0.0))) / 2.0
            p_right   = (s(data_row.get("Pipe_Pressure_Right1_m", 0.0)) +
                         s(data_row.get("Pipe_Pressure_Right2_m", 0.0))) / 2.0

            torricelli_amp = CD * leak_area * np.sqrt(2 * G * max(pressure, 1e-6))

            if d_left > 0 and d_right > 0:
                leak_pos  = float(np.clip(d_left / (d_left + d_right), 0.0, 1.0))
                pos_valid = 1.0
            else:
                leak_pos  = 0.5
                pos_valid = 0.0
        else:
            d_left = d_right = pressure = leak_flow = leak_area = 0.0
            torricelli_amp = 0.0
            sens_p  = self._parse_list(data_row.get("All_Sensor_Pressures_m", "0,0"))
            p_left  = sens_p[0] if len(sens_p) >= 1 else 0.0
            p_right = sens_p[1] if len(sens_p) >= 2 else 0.0
            leak_pos  = 0.5
            pos_valid = 0.0

        return {
            "leak_status":    leak_status,
            "wave_speed":     wave_speed,
            "alpha":          alpha,
            "pipe_length":    pipe_length,
            "pipe_diameter":  pipe_diameter,
            "pipe_roughness": pipe_roughness,
            "pipe_material":  pipe_material,
            "flow_velocity":  flow_velocity,
            "flow_rate":      flow_rate,
            "p_left":         p_left,
            "p_right":        p_right,
            "d_left":         d_left,
            "d_right":        d_right,
            "leak_flow":      leak_flow,
            "leak_area":      leak_area,
            "torricelli_amp": torricelli_amp,
            "pressure":       pressure,
            "leak_pos":       leak_pos,
            "pos_valid":      pos_valid,
            "demand":         demand,
            "sensor_left":    sensor_left,
            "sensor_right":   sensor_right,
        }

    def _synthetic_noise(self, amplitude: float) -> np.ndarray:
        T = self.signal_length
        shared      = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.7, T))
        indep_left  = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.3, T))
        indep_right = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.3, T))
        return np.stack([
            shared + indep_left  + np.random.normal(0, 0.002, T),
            shared + indep_right + np.random.normal(0, 0.002, T),
        ]).astype(np.float32)

    def _get_noise(self, flow_velocity: float, demand: float) -> np.ndarray:
        noise_window = self.noise_bank.get_window()
        if noise_window is not None:
            scale = 1.0 + abs(flow_velocity) * 0.1 + demand * 0.05
            return noise_window * scale
        else:
            amp = 0.01 + abs(flow_velocity) * 0.005 + demand * 0.003
            return self._synthetic_noise(amp)

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signal(self, c: dict) -> np.ndarray:
        T  = self.signal_length
        fs = self.fs

        mat = c["pipe_material"]
        center_freq, bandwidth, damping = MATERIAL_ACOUSTIC.get(mat, DEFAULT_ACOUSTIC)

        result = self._get_noise(c["flow_velocity"], c["demand"])
        result[0] += c["p_left"]  * 3e-4
        result[1] += c["p_right"] * 3e-4

        if c["leak_status"] != 1 or (c["d_left"] <= 0 and c["d_right"] <= 0):
            return self._normalize(result)

        # ── [C1] SNR-calibrated amplitude (deterministic from Torricelli) ────
        # Torricelli_norm maps orifice flow rate to [0, 1]
        # Scaling: typical torricelli_amp ~ 1e-4 to 4e-4, so *5000 gives 0.05-2.0
        # clipped to [1e-4, 1.0] for valid SNR range
        # SNR is then mapped deterministically: larger leak = higher SNR
        # Small jitter (±1.5 dB) added for augmentation without destroying correlation
        noise_rms       = self.noise_bank.noise_rms
        torricelli_norm = np.clip(c["torricelli_amp"] * 5000.0, 1e-4, 1.0)
        snr_db          = SNR_DB_MIN + (SNR_DB_MAX - SNR_DB_MIN) * torricelli_norm
        snr_db         += np.random.uniform(-1.5, 1.5)   # small jitter only
        snr_db          = np.clip(snr_db, SNR_DB_MIN, SNR_DB_MAX)
        snr_linear      = 10 ** (snr_db / 20.0)
        amp             = max(noise_rms * snr_linear, 1e-7)

        eff_speed        = c["wave_speed"] * np.random.uniform(0.85, 1.15)
        eff_alpha        = c["alpha"]      * np.random.uniform(0.9,  1.1)
        roughness_factor = np.clip(c["pipe_roughness"] / 100.0, 0.5, 3.0)

        # ── [C2] Pink noise source ─────────────────────────────────────────
        source = generate_leak_source_pink(center_freq, bandwidth, amp, T, fs)

        # ── [C3] Correlated channel propagation ───────────────────────────
        CORRELATION_ALPHA = 0.6
        shared = source * CORRELATION_ALPHA

        if c["d_left"] > 0:
            delay = int(np.clip((c["d_left"] / eff_speed) * fs, 0, T - 1))
            atten = np.exp(-eff_alpha * damping * roughness_factor * c["d_left"])
            result[0] += np.roll(source, delay) * atten * (1 - CORRELATION_ALPHA) + shared * atten

        if c["d_right"] > 0:
            delay = int(np.clip((c["d_right"] / eff_speed) * fs, 0, T - 1))
            atten = np.exp(-eff_alpha * damping * roughness_factor * c["d_right"])
            result[1] += np.roll(source, delay) * atten * (1 - CORRELATION_ALPHA) + shared * atten

        return self._normalize(result)

    # ── [C4] Fixed-scale normalisation ────────────────────────────────────────
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Fixed-scale normalisation — preserves relative amplitude between samples.

        Divides by a fixed reference (10x empirical noise floor RMS) so that:
          - Strong leak signals remain larger than weak leak signals
          - No-leak signals cluster near a consistent low amplitude
          - Amplitude differences that encode severity are preserved

        Per-sample z-score (used in Model B) forced every signal to unit
        variance regardless of leak amplitude, destroying the amplitude
        information needed for severity estimation (confirmed: RMS std=0.0000).
        """
        ref_scale = max(self.noise_bank.noise_rms * 10.0, 1e-6)
        return np.clip(signal / ref_scale, -10.0, 10.0).astype(np.float32)

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        signal = signal * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.25:
            signal += np.random.normal(0, np.random.uniform(0.005, 0.03), signal.shape)
        return signal

    def __getitem__(self, idx: int):
        try:
            c      = self._cache[self._valid_idx[idx]]
            signal = self.generate_signal(c)
            if self.augment:
                signal = self._augment(signal)

            scalars = torch.tensor([
                c["wave_speed"]     / 2000.0,
                c["alpha"]          / 0.003,
                c["pipe_length"]    / 1000.0,
                c["pipe_diameter"]  / 0.6,
                c["pipe_roughness"] / 200.0,
                c["flow_velocity"]  / 10.0,
                c["flow_rate"]      / 100.0,
                c["demand"]         / 1.2,
                c["p_left"]         / 100.0,
                c["p_right"]        / 100.0,
                c["leak_area"]      / 2e-5,
            ], dtype=torch.float32)

            return (
                torch.tensor(signal,                  dtype=torch.float32),
                scalars,
                torch.tensor(float(c["leak_status"]), dtype=torch.float32),
                torch.tensor(c["leak_pos"],           dtype=torch.float32),
                torch.tensor(c["leak_flow"],          dtype=torch.float32),
                torch.tensor(c["pos_valid"],          dtype=torch.float32),
            )
        except Exception:
            return (
                torch.zeros(2, self.signal_length, dtype=torch.float32),
                torch.zeros(11, dtype=torch.float32),
                torch.tensor(-1.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )


def collate_fn(batch):
    batch = [b for b in batch if b[2].item() >= 0]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None