"""
public_data.py — single-channel leak/no-leak windows from several real datasets
=================================================================================
Loads every real dataset into ONE common format so that detectors can be
trained on some datasets and tested on others (experiments/cross_dataset.py).

Common format (identical for every dataset):
  - one sensor channel per window
  - resampled to 5 kHz (the synthetic model's rate) and BAND-LIMITED to
    2 kHz. The Hong Kong noise loggers record at 4096 Hz and contain nothing
    above ~2 kHz; without a common band limit a classifier could identify the
    dataset (and so its label mix) from bandwidth alone.
  - 0.4 s non-overlapping windows (2000 samples), raw amplitude. Per-window
    standardisation happens in the experiment, because sensor gains differ
    wildly between datasets.
  - `group` = the unit that must never be split between train and test:
      Hong Kong : SITE (repeat recordings at one site share a group)
      Dongguan  : recording (consecutive 1 s clips of one recording)
      Mendeley  : recording (both channels of one run)

Datasets (see scripts/download_public_data.py for sources and licences):
  hk_noiselogger, hk_hydrophone   Hong Kong real buried networks
  dongguan                        outdoor training base, Dongguan
  mendeley_acc, mendeley_hyd      Mendeley lab testbed (Branched + Looped)
"""

import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

import mendeley as M
from _common import DATASETS

FS = 5000
BAND_HZ = 2000.0
WIN = 2000
PUBLIC = DATASETS / "public"


@dataclass
class Windows:
    x: np.ndarray          # (N, T) float32
    y: np.ndarray          # (N,) 1 = leak
    group: np.ndarray      # (N,) str
    dataset: np.ndarray    # (N,) str
    meta: np.ndarray       # (N,) str — material / device / condition where known

    def __len__(self):
        return len(self.y)

    def subset(self, m) -> "Windows":
        return Windows(self.x[m], self.y[m], self.group[m], self.dataset[m], self.meta[m])

    @staticmethod
    def concat(parts) -> "Windows":
        parts = [p for p in parts if len(p)]
        return Windows(*(np.concatenate([getattr(p, f) for p in parts])
                         for f in ("x", "y", "group", "dataset", "meta")))


def to_common(x: np.ndarray, fs: int) -> np.ndarray:
    """Resample to 5 kHz, then low-pass at 2 kHz (zero-phase)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    x = x - x.mean()
    if fs != FS:
        g = gcd(int(fs), FS)
        x = resample_poly(x, FS // g, int(fs) // g)
    sos = butter(8, BAND_HZ / (FS / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def windows_of(x: np.ndarray) -> np.ndarray:
    n = len(x) // WIN
    return x[:n * WIN].reshape(n, WIN) if n else np.zeros((0, WIN), np.float32)


def _pack(rows) -> Windows:
    """rows: iterable of (signal @ common rate, label, group, dataset, meta)."""
    xs, ys, gs, ds, ms = [], [], [], [], []
    for sig, lab, grp, dset, meta in rows:
        w = windows_of(sig)
        xs.append(w)
        ys += [lab] * len(w); gs += [grp] * len(w); ds += [dset] * len(w); ms += [meta] * len(w)
    if not xs:
        return Windows(np.zeros((0, WIN), np.float32), *(np.array([]) for _ in range(4)))
    return Windows(np.concatenate(xs).astype(np.float32), np.array(ys, int), np.array(gs),
                   np.array(ds), np.array(ms))


def _read_wav(path: Path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # some HK files have a truncated header
        return wavfile.read(path)


# ── Hong Kong (Mendeley hkn8mxcjyz) ────────────────────────────────────────────

def hk_site(stem: str) -> str:
    """Site id from a Hong Kong file name.
    Noise loggers  '003_05593-20180927'  -> '05593'   (index_site-date)
    Hydrophones    '1.3.02.0345'         -> 'h1.3'    (class.site.repeat.time: repeated
                                                      recordings 15 min apart at one site)
    Accelerometers '00241_20180818_1445' -> '00241'   (site_date_time)"""
    m = re.match(r"^\d+_(\d+)-\d{8}", stem)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)\.(\d+)\.\d+\.\d+$", stem)
    if m:
        return f"h{m.group(1)}.{m.group(2)}"
    return stem.split("_")[0]


def load_hongkong(root: Path = PUBLIC / "hongkong", sensors=("Noise Loggers", "Hydrophones")) -> Windows:
    rows = []
    for sensor in sensors:
        base = Path(root) / sensor
        if not base.is_dir():
            continue
        tag = "hk_noiselogger" if sensor.startswith("Noise") else "hk_hydrophone"
        for f in sorted(base.rglob("*.wav")):
            folder = f.parent.name                          # Leak, No-Leak, Leak-Metal, ...
            label = 0 if folder.lower().replace("-", "").startswith("noleak") else 1
            material = ("metal" if "-Metal" in folder else "nonmetal" if "NonMetal" in folder
                        else "unknown")
            fs, x = _read_wav(f)
            rows.append((to_common(x, fs), label, f"hk:{hk_site(f.stem)}", tag, material))
    return _pack(rows)


# ── Dongguan (Zenodo 18631450) ─────────────────────────────────────────────────

def dongguan_recording(stem: str) -> str:
    """Strip the trailing '<k>-<k+1>' clip index (and an optional '_n' copy
    suffix): 'NA-NA-NA-NA-hydrophone_0-1_1' -> 'NA-NA-NA-NA-hydrophone'.
    Clips whose metadata is all 'NA' collapse into one large group, which is
    conservative (it can only make the test harder, never leak data)."""
    return re.sub(r"[-_]\d+-\d+(_\d+)?$", "", stem)


def load_dongguan(root: Path = PUBLIC / "dongguan" / "extracted", include_env_noise: bool = False) -> Windows:
    rows = []
    folders = [("leak acoustic data", 1), ("no leak acoustic data", 0)]
    if include_env_noise:
        folders.append(("environmental noise", 0))
    for folder, label in folders:
        for f in sorted((Path(root) / folder).glob("*.wav")):
            fs, x = _read_wav(f)
            material = f.stem.split("-")[0] if label or folder != "environmental noise" else "env"
            rows.append((to_common(x, fs), label, f"dg:{dongguan_recording(f.stem)}:{label}",
                         "dongguan", material))
    return _pack(rows)


# ── Mendeley testbed (the data E4/E5 use) ──────────────────────────────────────

def load_mendeley(sensor: str = "accelerometer", root: Path = None) -> Windows:
    root = root or (DATASETS / ("Accelerometer/Accelerometer" if sensor == "accelerometer"
                                else "Hydrophone/Hydrophone"))
    rows = []
    for rec in M.discover(root, sensor):
        x1, x2 = M.load_recording(rec, fs_out=FS)
        for x in (x1, x2):
            rows.append((to_common(x, FS), rec.label, f"md:{rec.rec_id}",
                         "mendeley_acc" if sensor == "accelerometer" else "mendeley_hyd",
                         f"{rec.topology}/{M.flow_condition(rec.rec_id)}"))
    return _pack(rows)


LOADERS = {
    "hk_noiselogger": lambda: load_hongkong(sensors=("Noise Loggers",)),
    "hk_hydrophone": lambda: load_hongkong(sensors=("Hydrophones",)),
    "dongguan": load_dongguan,
    "mendeley_acc": lambda: load_mendeley("accelerometer"),
    "mendeley_hyd": lambda: load_mendeley("hydrophone"),
}


def load(names) -> Windows:
    parts = []
    for n in names:
        try:
            w = LOADERS[n]()
        except FileNotFoundError as e:
            print(f"  skip {n}: {e}")
            continue
        print(f"  {n:16s} {len(w):6d} windows | {len(np.unique(w.group)):4d} groups | "
              f"leak {w.y.mean() if len(w) else 0:.0%}")
        parts.append(w)
    return Windows.concat(parts)
