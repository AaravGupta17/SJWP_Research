"""
LeakDataset — Physics-Based Waveform Synthesis with Real Noise
==============================================================
Key improvements over previous version:
  1. Real Mendeley no-leak recordings as noise background
     - Branched topology used for training noise
     - Looped topology held out for validation only
  2. Correlated noise between channels (same pipe = shared vibration)
  3. All available columns: Pipe_Roughness, Leak_Area_m2, Pipe_Material
  4. Torricelli amplitude from Leak_Area_m2 (physically correct)
  5. Pipe_Material from CSV directly (not inferred from folder)

Signal generation:
  - Negative samples: real Mendeley no-leak window
  - Positive samples: real Mendeley no-leak + synthetic TDOA pulse

fs = 5000 Hz (model training frequency)
Mendeley raw files at 8000 Hz → resampled to 5000 Hz
signal_length = 2000 samples = 400ms window
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import lfilter, resample_poly
from pathlib import Path
from math import gcd
from tqdm import tqdm

# ── Mendeley noise configuration ───────────────────────────────────────────────
MENDELEY_ROOT  = Path("../datasets/Hydrophone/Hydrophone")
MENDELEY_FS    = 8000   # raw files are 8kHz
MODEL_FS       = 5000   # model expects 5kHz
SIGNAL_LENGTH  = 2000   # 400ms at 5kHz

# Branched no-leak files used as training noise
# Looped held out for Mendeley validation
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

# Material acoustic properties
# Format: (center_freq_hz, bandwidth_hz, damping)
MATERIAL_ACOUSTIC = {
    "CI":    (150.0, 80.0,  1.3),
    "DI":    (280.0, 150.0, 0.9),
    "PVC":   (800.0, 300.0, 0.5),
    "STEEL": (500.0, 200.0, 0.7),
}
DEFAULT_ACOUSTIC = (200.0, 100.0, 1.0)

# Torricelli discharge coefficient
CD = 0.61
G  = 9.81


class MendeleyNoiseBank:
    """
    Loads all Branched no-leak Mendeley recordings into RAM.
    Provides random windows of real pipe noise for signal generation.
    Noise is correlated between channels — same pipe, shared vibration.
    """

    def __init__(self):
        self.windows = []   # list of (2, SIGNAL_LENGTH) float32 arrays
        self._load()

    def _load(self):
        print("Loading Mendeley noise bank (Branched no-leak)...")

        # Load paired H1/H2 files
        h1_files = [f for f in TRAIN_NOISE_FILES if "H1" in f]
        loaded = 0

        for h1_path in h1_files:
            h2_path = h1_path.replace("H1", "H2")
            try:
                h1_raw = self._load_raw(str(MENDELEY_ROOT / h1_path))
                h2_raw = self._load_raw(str(MENDELEY_ROOT / h2_path))

                # Resample from 8kHz to 5kHz
                h1 = self._resample(h1_raw)
                h2 = self._resample(h2_raw)

                # Slice into windows
                n_windows = min(len(h1), len(h2)) // SIGNAL_LENGTH
                for i in range(n_windows):
                    s = i * SIGNAL_LENGTH
                    e = s + SIGNAL_LENGTH
                    w = np.stack([h1[s:e], h2[s:e]]).astype(np.float32)
                    self.windows.append(w)
                loaded += 1
            except Exception as e:
                print(f"  Warning: could not load {h1_path}: {e}")

        print(f"  Loaded {loaded} file pairs → {len(self.windows)} noise windows")
        if len(self.windows) == 0:
            print("  WARNING: No Mendeley noise loaded — falling back to synthetic noise")

    def _load_raw(self, path: str) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.int32).astype(np.float32)
        mx  = np.abs(raw).max()
        return raw / mx if mx > 0 else raw

    def _resample(self, sig: np.ndarray) -> np.ndarray:
        g    = gcd(MODEL_FS, MENDELEY_FS)
        up   = MODEL_FS   // g
        down = MENDELEY_FS // g
        return resample_poly(sig, up, down).astype(np.float32)

    def get_window(self) -> np.ndarray:
        """Return a random (2, SIGNAL_LENGTH) noise window."""
        if not self.windows:
            return None
        idx = np.random.randint(len(self.windows))
        return self.windows[idx].copy()

    def __len__(self):
        return len(self.windows)


# Global noise bank — loaded once, shared across all dataset instances
_NOISE_BANK = None

def get_noise_bank() -> MendeleyNoiseBank:
    global _NOISE_BANK
    if _NOISE_BANK is None:
        _NOISE_BANK = MendeleyNoiseBank()
    return _NOISE_BANK


class LeakDataset(Dataset):
    """
    Caches EPANET scalar values into RAM.
    Generates signals using real Mendeley noise as background.

    Positive samples: real noise + synthetic TDOA leak pulse
    Negative samples: real noise only

    Position target: d_left / (d_left + d_right)
    """

    def __init__(self, index_csv: str,
                 signal_length: int = SIGNAL_LENGTH,
                 sampling_frequency: int = MODEL_FS,
                 augment: bool = False):
        self.signal_length = signal_length
        self.fs            = sampling_frequency
        self.augment       = augment

        # Load noise bank
        self.noise_bank = get_noise_bank()

        print(f"Loading {index_csv}")
        self.index_df = pd.read_csv(index_csv)
        print(f"Caching {len(self.index_df):,} rows...")
        self._cache     = self._build_cache()
        self._valid_idx = [i for i, c in enumerate(self._cache) if c is not None]
        print(f"Ready: {len(self._valid_idx):,} samples | "
              f"fs={sampling_frequency}Hz | augment={augment}")

    def __len__(self):
        return len(self._valid_idx)

    # ── Caching ────────────────────────────────────────────────────────────────

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

        # ── Pipe properties ────────────────────────────────────────────────
        pipe_length   = s(data_row.get("Pipe_Length_m"),                    100.0)
        pipe_diameter = s(data_row.get("Pipe_Diameter_m"),                  0.15)
        pipe_roughness= s(data_row.get("Pipe_Roughness"),                   100.0)
        pipe_material = str(data_row.get("Pipe_Material", "CI")).strip().upper()
        flow_velocity = s(data_row.get("Avg_Flow_Velocity_mps"),            0.0)
        flow_rate     = s(data_row.get("Avg_Flow_Rate_lps"),                0.0)
        wave_speed    = s(data_row.get("Acoustic_Propagation_Speed_mps"),   1200.0)
        alpha         = s(data_row.get("Attenuation_Alpha_per_m"),          0.001)
        leak_status   = int(s(data_row.get("Leak_Status",                   0)))
        leak_area     = s(data_row.get("Leak_Area_m2",                      0.0))
        demand        = s(index_row["demand_multiplier"],                    1.0)

        # ── Sensor positions ───────────────────────────────────────────────
        sensor_pos    = self._parse_list(data_row.get("Sensor_Positions_m", ""))
        sensor_left   = sensor_pos[0] if len(sensor_pos) >= 1 else pipe_length / 3
        sensor_right  = sensor_pos[1] if len(sensor_pos) >= 2 else 2 * pipe_length / 3

        if file_type == "leak":
            d_left  = s(data_row.get("Leak_Distance_Left_m"),  0.0)
            d_right = s(data_row.get("Leak_Distance_Right_m"), 0.0)

            # Fallback to 2nd sensor
            if d_left == 0:
                d_left  = s(data_row.get("Leak_Distance_2nd_Left_m"),  0.0)
            if d_right == 0:
                d_right = s(data_row.get("Leak_Distance_2nd_Right_m"), 0.0)

            pressure  = s(data_row.get("Avg_Pressure_at_Leak"), 0.0)
            leak_flow = s(data_row.get("Leak_Flow_Lps"),         0.0)

            p_left  = (s(data_row.get("Pipe_Pressure_Left1_m",  0.0)) +
                       s(data_row.get("Pipe_Pressure_Left2_m",  0.0))) / 2.0
            p_right = (s(data_row.get("Pipe_Pressure_Right1_m", 0.0)) +
                       s(data_row.get("Pipe_Pressure_Right2_m", 0.0))) / 2.0

            # Torricelli amplitude: Q = Cd * A * sqrt(2 * g * P)
            # pressure in metres head → Pa = rho * g * h
            pressure_pa = pressure * 1000 * G
            torricelli_amp = CD * leak_area * np.sqrt(2 * G * max(pressure, 1e-6))

            if d_left > 0 and d_right > 0:
                leak_pos  = float(np.clip(d_left / (d_left + d_right), 0.0, 1.0))
                pos_valid = 1.0
            else:
                leak_pos  = 0.5
                pos_valid = 0.0

        else:  # base
            d_left = d_right = pressure = leak_flow = leak_area = 0.0
            torricelli_amp = 0.0
            sens_p  = self._parse_list(data_row.get("All_Sensor_Pressures_m", "0,0"))
            p_left  = sens_p[0] if len(sens_p) >= 1 else 0.0
            p_right = sens_p[1] if len(sens_p) >= 2 else 0.0
            leak_pos  = 0.5
            pos_valid = 0.0

        return {
            "leak_status":     leak_status,
            "wave_speed":      wave_speed,
            "alpha":           alpha,
            "pipe_length":     pipe_length,
            "pipe_diameter":   pipe_diameter,
            "pipe_roughness":  pipe_roughness,
            "pipe_material":   pipe_material,
            "flow_velocity":   flow_velocity,
            "flow_rate":       flow_rate,
            "p_left":          p_left,
            "p_right":         p_right,
            "d_left":          d_left,
            "d_right":         d_right,
            "leak_flow":       leak_flow,
            "leak_area":       leak_area,
            "torricelli_amp":  torricelli_amp,
            "pressure":        pressure,
            "leak_pos":        leak_pos,
            "pos_valid":       pos_valid,
            "demand":          demand,
            "sensor_left":     sensor_left,
            "sensor_right":    sensor_right,
        }

    # ── Signal generation ──────────────────────────────────────────────────────

    def _synthetic_noise(self, amplitude: float) -> np.ndarray:
        """
        Correlated synthetic noise as fallback when Mendeley noise unavailable.
        Both channels share common pipe vibration + small independent component.
        """
        T = self.signal_length
        shared      = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.7, T))
        indep_left  = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.3, T))
        indep_right = lfilter([1.0], [1.0, -0.95],
                              np.random.normal(0, amplitude * 0.3, T))
        noise = np.stack([
            shared + indep_left  + np.random.normal(0, 0.002, T),
            shared + indep_right + np.random.normal(0, 0.002, T),
        ])
        return noise.astype(np.float32)

    def _get_noise(self, flow_velocity: float, demand: float) -> np.ndarray:
        """
        Get noise background — real Mendeley if available, synthetic fallback.
        Scale amplitude by flow conditions.
        """
        noise_window = self.noise_bank.get_window()

        if noise_window is not None:
            # Scale real noise by flow conditions
            scale = 1.0 + abs(flow_velocity) * 0.1 + demand * 0.05
            return noise_window * scale
        else:
            # Fallback: correlated synthetic noise
            amp = 0.01 + abs(flow_velocity) * 0.005 + demand * 0.003
            return self._synthetic_noise(amp)

    def generate_signal(self, c: dict) -> np.ndarray:
        T    = self.signal_length
        fs   = self.fs
        time = np.arange(T) / fs

        mat   = c["pipe_material"]
        center_freq, bandwidth, damping = MATERIAL_ACOUSTIC.get(
            mat, DEFAULT_ACOUSTIC)

        # Get real noise background (correlated between channels)
        result = self._get_noise(c["flow_velocity"], c["demand"])

        # Pressure DC offset
        result[0] += c["p_left"]  * 3e-4
        result[1] += c["p_right"] * 3e-4

        # No leak or no sensors → return noise only
        if c["leak_status"] != 1 or (c["d_left"] <= 0 and c["d_right"] <= 0):
            return self._normalize(result)

        # ── Leak source signal ─────────────────────────────────────────────
        # Amplitude from Torricelli — physically grounded
        amp = np.clip(c["torricelli_amp"] * 50.0, 1e-4, 5.0)

        # ±15% wave speed perturbation — cross-network generalisation
        eff_speed = c["wave_speed"] * np.random.uniform(0.85, 1.15)
        eff_alpha = c["alpha"] * np.random.uniform(0.9, 1.1)

        # Pipe roughness affects damping — higher roughness = more attenuation
        roughness_factor = np.clip(c["pipe_roughness"] / 100.0, 0.5, 3.0)

        # Gaussian pulse with material-dependent carrier
        center  = np.random.uniform(0.05, 0.25)
        sigma   = np.clip(1.0 / (2 * np.pi * bandwidth), 0.002, 0.04)
        envelope = amp * np.exp(-((time - center) ** 2) / (2 * sigma ** 2))
        carrier  = (      np.sin(2 * np.pi * center_freq     * time)
                  + 0.3 * np.sin(2 * np.pi * center_freq * 2 * time)
                  + 0.1 * np.sin(2 * np.pi * center_freq * 3 * time))
        source = envelope * carrier

        # Left sensor
        if c["d_left"] > 0:
            delay = int(np.clip((c["d_left"] / eff_speed) * fs, 0, T - 1))
            atten = np.exp(-eff_alpha * damping * roughness_factor * c["d_left"])
            result[0] += np.roll(source, delay) * atten

        # Right sensor
        if c["d_right"] > 0:
            delay = int(np.clip((c["d_right"] / eff_speed) * fs, 0, T - 1))
            atten = np.exp(-eff_alpha * damping * roughness_factor * c["d_right"])
            result[1] += np.roll(source, delay) * atten

        return self._normalize(result)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        for ch in range(signal.shape[0]):
            mu  = np.mean(signal[ch])
            std = np.std(signal[ch]) + 1e-8
            signal[ch] = np.clip((signal[ch] - mu) / std, -5.0, 5.0)
        return signal

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """Conservative augmentation — no time shift (destroys TDOA)."""
        signal = signal * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.25:
            signal += np.random.normal(0, np.random.uniform(0.005, 0.03),
                                       signal.shape)
        return signal

    # ── Scalars ────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        try:
            c      = self._cache[self._valid_idx[idx]]
            signal = self.generate_signal(c)
            if self.augment:
                signal = self._augment(signal)

            # 11 physics features (was 9)
            scalars = torch.tensor([
                c["wave_speed"]     / 2000.0,
                c["alpha"]          / 0.003,
                c["pipe_length"]    / 1000.0,
                c["pipe_diameter"]  / 0.6,
                c["pipe_roughness"] / 200.0,   # NEW
                c["flow_velocity"]  / 10.0,
                c["flow_rate"]      / 100.0,
                c["demand"]         / 1.2,
                c["p_left"]         / 100.0,
                c["p_right"]        / 100.0,
                c["leak_area"]      / 2e-5,    # NEW
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