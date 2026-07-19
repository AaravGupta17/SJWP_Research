"""
LeakDataset — Physics-Based Waveform Synthesis
===============================================
CRITICAL DESIGN DECISIONS (do not change between train and test):
  - fs = 5000 Hz
  - signal_length = 2000 samples (400ms window)
  - Position target = d_left / (d_left + d_right) — sensor-relative, not pipe-relative
  - Position only valid when BOTH sensors present (d_left > 0 AND d_right > 0)
  - Single sensor: detect + severity only, position_valid = 0
  - Wave speed perturbation ±15% for cross-network generalisation
  - All 4 materials: CI, DI, PVC, STEEL with different acoustic properties

Material IDs (from build_index.py material_map — order depends on os.listdir):
  Check material_map output from build_index.py to confirm your mapping.
  Default acoustic properties assigned by material name, not ID.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.signal import lfilter

# Material acoustic properties by NAME (not ID — avoids ID mapping confusion)
# Format: (center_freq_hz, bandwidth_hz, damping, wave_speed_mps)
MATERIAL_ACOUSTIC = {
    "CI":    (150.0, 80.0,  1.3, 1200.0),
    "DI":    (280.0, 150.0, 0.9, 1350.0),
    "PVC":   (800.0, 300.0, 0.5, 400.0),
    "STEEL": (500.0, 200.0, 0.7, 1500.0),
}
DEFAULT_ACOUSTIC = (200.0, 100.0, 1.0, 1200.0)


def get_material_name(index_row) -> str:
    """Extract material name from file path."""
    path = str(index_row.get("file_path", ""))
    for mat in MATERIAL_ACOUSTIC:
        if f"/{mat}/" in path or f"\\{mat}\\" in path:
            return mat
    return "CI"


class LeakDataset(Dataset):
    """
    Caches all scalar values from EPANET CSVs into RAM at init.
    Generates acoustic waveforms on-the-fly during training.

    Position target: d_left / (d_left + d_right)
      - Ranges [0, 1]: 0 = leak at left sensor, 1 = leak at right sensor
      - Only valid when both sensors present
      - Directly tied to TDOA physics — no dependency on True_Leak_Position_m
    """

    def __init__(self, index_csv: str,
                 signal_length: int = 2000,
                 sampling_frequency: int = 5000,
                 augment: bool = False):
        self.signal_length = signal_length
        self.fs            = sampling_frequency
        self.augment       = augment

        print(f"Loading {index_csv}")
        self.index_df = pd.read_csv(index_csv)
        print(f"Caching {len(self.index_df):,} rows...")
        self._cache     = self._build_cache()
        self._valid_idx = [i for i, c in enumerate(self._cache) if c is not None]
        print(f"Ready: {len(self._valid_idx):,} valid samples | "
              f"fs={sampling_frequency}Hz | {signal_length}samples | "
              f"augment={augment}")

    def __len__(self):
        return len(self._valid_idx)

    # ── Caching ────────────────────────────────────────────────────────────────

    def _build_cache(self) -> list:
        from tqdm import tqdm
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
        mat_name  = get_material_name(index_row)

        pipe_length   = s(data_row.get("Pipe_Length_m"),             100.0)
        pipe_diameter = s(data_row.get("Pipe_Diameter_m"),           0.15)
        flow_velocity = s(data_row.get("Avg_Flow_Velocity_mps"),     0.0)
        flow_rate     = s(data_row.get("Avg_Flow_Rate_lps"),         0.0)
        wave_speed    = s(data_row.get("Acoustic_Propagation_Speed_mps"), 1200.0)
        alpha         = s(data_row.get("Attenuation_Alpha_per_m"),   0.001)
        leak_status   = int(s(data_row.get("Leak_Status",            0)))
        demand        = s(index_row["demand_multiplier"],             1.0)

        if file_type == "leak":
            d_left  = s(data_row.get("Leak_Distance_Left_m"),  0.0)
            d_right = s(data_row.get("Leak_Distance_Right_m"), 0.0)

            # Fallback to 2nd sensor distances if primary is 0
            if d_left == 0:
                d_left  = s(data_row.get("Leak_Distance_2nd_Left_m"),  0.0)
            if d_right == 0:
                d_right = s(data_row.get("Leak_Distance_2nd_Right_m"), 0.0)

            leak_flow = s(data_row.get("Leak_Flow_Lps"),        0.0)
            pressure  = s(data_row.get("Avg_Pressure_at_Leak"), 0.0)
            p_left    = (s(data_row.get("Pipe_Pressure_Left1_m",  0.0)) +
                         s(data_row.get("Pipe_Pressure_Left2_m",  0.0))) / 2.0
            p_right   = (s(data_row.get("Pipe_Pressure_Right1_m", 0.0)) +
                         s(data_row.get("Pipe_Pressure_Right2_m", 0.0))) / 2.0

            # Position: d_left / (d_left + d_right) — sensor-relative TDOA ratio
            if d_left > 0 and d_right > 0:
                leak_pos  = float(np.clip(d_left / (d_left + d_right), 0.0, 1.0))
                pos_valid = 1.0
            else:
                leak_pos  = 0.5   # placeholder, masked in loss
                pos_valid = 0.0

        else:  # base
            d_left = d_right = leak_flow = pressure = 0.0
            sens_p  = self._parse_list(data_row.get("All_Sensor_Pressures_m", "0,0"))
            p_left  = sens_p[0] if len(sens_p) >= 1 else 0.0
            p_right = sens_p[1] if len(sens_p) >= 2 else 0.0
            leak_pos  = 0.5
            pos_valid = 0.0

        return {
            "leak_status":   leak_status,
            "wave_speed":    wave_speed,
            "alpha":         alpha,
            "pipe_length":   pipe_length,
            "pipe_diameter": pipe_diameter,
            "flow_velocity": flow_velocity,
            "flow_rate":     flow_rate,
            "p_left":        p_left,
            "p_right":       p_right,
            "d_left":        d_left,
            "d_right":       d_right,
            "leak_flow":     leak_flow,
            "pressure":      pressure,
            "leak_pos":      leak_pos,
            "pos_valid":     pos_valid,
            "demand":        demand,
            "mat_name":      mat_name,
        }

    # ── Signal generation ──────────────────────────────────────────────────────

    def _colored_noise(self, shape: tuple, amplitude: float) -> np.ndarray:
        """Fast colored noise via scipy IIR filter."""
        white = np.random.normal(0, amplitude, shape)
        return lfilter([1.0], [1.0, -0.95], white, axis=-1)

    def generate_signal(self, c: dict) -> np.ndarray:
        T    = self.signal_length
        fs   = self.fs
        time = np.arange(T) / fs

        center_freq, bandwidth, damping, _ = MATERIAL_ACOUSTIC.get(
            c["mat_name"], DEFAULT_ACOUSTIC
        )

        # ±15% wave speed perturbation — critical for cross-network generalisation
        eff_speed = c["wave_speed"] * np.random.uniform(0.85, 1.15)
        eff_alpha = c["alpha"] * np.random.uniform(0.9, 1.1)

        # Noise: flow turbulence (colored) + thermal (white) + ground vibration
        noise_amp = 0.01 + abs(c["flow_velocity"]) * 0.005 + c["demand"] * 0.003
        noise     = self._colored_noise((2, T), noise_amp)
        noise    += np.random.normal(0, 0.002, (2, T))          # thermal
        noise    += self._colored_noise((2, T), 0.003)           # ground vibration

        # Pressure DC offset
        noise[0] += c["p_left"]  * 3e-4
        noise[1] += c["p_right"] * 3e-4

        if c["leak_status"] != 1 or (c["d_left"] <= 0 and c["d_right"] <= 0):
            return self._normalize(noise)

        # Leak source: Gaussian pulse (simpler than Gabor — generalises better)
        amp      = np.clip(c["leak_flow"] * np.sqrt(max(c["pressure"], 0.0)) * 0.3,
                           1e-3, 8.0)
        center   = np.random.uniform(0.05, 0.25)   # random position in window
        sigma    = np.clip(1.0 / (2 * np.pi * bandwidth), 0.002, 0.04)
        envelope = amp * np.exp(-((time - center) ** 2) / (2 * sigma ** 2))

        # Carrier with harmonics
        carrier = (      np.sin(2 * np.pi * center_freq       * time)
                 + 0.3 * np.sin(2 * np.pi * center_freq * 2   * time)
                 + 0.1 * np.sin(2 * np.pi * center_freq * 3   * time))
        source = envelope * carrier

        result = noise.copy()

        # Left sensor
        if c["d_left"] > 0:
            delay   = int(np.clip((c["d_left"] / eff_speed) * fs, 0, T - 1))
            atten   = np.exp(-eff_alpha * damping * c["d_left"])
            result[0] += np.roll(source, delay) * atten

        # Right sensor
        if c["d_right"] > 0:
            delay   = int(np.clip((c["d_right"] / eff_speed) * fs, 0, T - 1))
            atten   = np.exp(-eff_alpha * damping * c["d_right"])
            result[1] += np.roll(source, delay) * atten

        return self._normalize(result)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Per-channel z-score — applied identically train and test."""
        for ch in range(signal.shape[0]):
            mu  = np.mean(signal[ch])
            std = np.std(signal[ch]) + 1e-8
            signal[ch] = np.clip((signal[ch] - mu) / std, -5.0, 5.0)
        return signal

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """Conservative augmentation — no time shift (destroys TDOA)."""
        signal = signal * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.25:
            signal += np.random.normal(0, np.random.uniform(0.01, 0.05), signal.shape)
        return signal

    # ── Main getter ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        try:
            c      = self._cache[self._valid_idx[idx]]
            signal = self.generate_signal(c)
            if self.augment:
                signal = self._augment(signal)

            scalars = torch.tensor([
                c["wave_speed"]    / 2000.0,
                c["alpha"]         / 0.003,
                c["pipe_length"]   / 1000.0,
                c["pipe_diameter"] / 0.6,
                c["flow_velocity"] / 10.0,
                c["flow_rate"]     / 100.0,
                c["demand"]        / 1.2,
                c["p_left"]        / 100.0,
                c["p_right"]       / 100.0,
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
                torch.zeros(9, dtype=torch.float32),
                torch.tensor(-1.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )


def collate_fn(batch):
    batch = [b for b in batch if b[2].item() >= 0]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None