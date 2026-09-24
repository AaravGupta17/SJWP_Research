"""
_common.py — shared helpers for the experiments/ scripts
=========================================================
- Repo-relative paths, so scripts run from any working directory.
- Model loading that works for every checkpoint in models/.
- record_run(): every experiment writes one JSON record (config, results,
  git commit, timestamp) to results/runs/. These files are the evidence
  trail; scripts/make_data_book.py builds the data book and a (git-ignored)
  results/runs/INDEX.csv from them. No shared file is appended to, so two
  people running experiments never create a merge conflict.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODELS_DIR  = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
# LEAKNET_OUT redirects plots and run records (used for smoke tests, so
# they never mix with real results). Resolved now, because some scripts
# os.chdir() later and a relative path would then point somewhere else.
_OUT        = os.environ.get("LEAKNET_OUT")
_OUT        = Path(_OUT).resolve() if _OUT else None
PLOTS_DIR   = _OUT / "plots" if _OUT else REPO_ROOT / "plots"
RUNS_DIR    = _OUT / "runs" if _OUT else RESULTS_DIR / "runs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS    = REPO_ROOT / "datasets"

MODEL_FS      = 5000
SIGNAL_LENGTH = 2000
N_SCALARS     = 11

# model_C/model.py is the canonical architecture (Model D uses an identical copy)
for _p in (REPO_ROOT / "model_C", REPO_ROOT / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import AcousticLeakNet  # noqa: E402


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict = None) -> AcousticLeakNet:
    cfg = cfg or {}
    return AcousticLeakNet(signal_length=SIGNAL_LENGTH, n_scalars=N_SCALARS,
                           base_channels=cfg.get("base_channels", 64),
                           dropout=cfg.get("dropout", 0.3),
                           fusion=cfg.get("fusion", "cca"))


def load_model(ckpt_name: str, device=None):
    """Returns (model in eval mode, checkpoint dict)."""
    device = device or get_device()
    ckpt = torch.load(MODELS_DIR / ckpt_name, map_location=device, weights_only=False)
    model = build_model(ckpt.get("cfg")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model, windows: np.ndarray, device=None, batch_size: int = 128,
                  return_pos: bool = False, logits: bool = False):
    """windows: (N, 2, T) float array -> P(leak) of shape (N,).
    Scalars are zeroed, exactly as in training/evaluation.

    logits=True returns the raw detection logit instead. USE LOGITS FOR
    AUROC: these models output logits beyond +-17, where float32 sigmoid
    rounds to exactly 0.0 or 1.0, and tied scores push AUROC toward 0.5
    even when the logits rank the windows correctly. Threshold rates are
    unchanged (logit >= 0  <=>  P >= 0.5)."""
    device = device or next(model.parameters()).device
    probs, pos = [], []
    for i in range(0, len(windows), batch_size):
        x = torch.as_tensor(np.ascontiguousarray(windows[i:i + batch_size]),
                            dtype=torch.float32, device=device)
        det, p, _ = model(x, torch.zeros(x.size(0), N_SCALARS, device=device))
        probs.append((det if logits else torch.sigmoid(det)).cpu().numpy())
        pos.append(p.cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros(0)
    if return_pos:
        return probs, (np.concatenate(pos) if pos else np.zeros(0))
    return probs


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _jsonable(o):
    if isinstance(o, str) and o.lower().startswith(str(REPO_ROOT).lower()):
        # store repo-relative paths: no user folder names in committed records
        return "<repo>" + o[len(str(REPO_ROOT)):].replace("\\", "/")
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def record_run(experiment: str, config: dict, results: dict, summary: str = "") -> Path:
    """Save one run record to results/runs/.

    summary: one short human-readable line (e.g. "clean AUROC 0.62 [0.51, 0.74]").
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    record = {
        "experiment": experiment,
        "timestamp": now.isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(get_device()),
        "command": " ".join(sys.argv),
        "config": config,
        "summary": summary,
        "results": results,
    }
    out = RUNS_DIR / f"{now:%Y-%m-%d_%H%M%S}_{experiment}.json"
    out.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")
    print(f"\nRun recorded: {out}")
    return out
