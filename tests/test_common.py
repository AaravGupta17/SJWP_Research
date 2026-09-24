"""Shared experiment helpers."""

import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def test_relative_leaknet_out_survives_chdir(tmp_path):
    # realism_check.py and snr_sweep.py os.chdir() after importing _common,
    # so a relative LEAKNET_OUT must already be absolute by then
    code = (
        f"import os, sys; sys.path.insert(0, {str(EXPERIMENTS)!r}); "
        "import _common; os.chdir(os.path.expanduser('~')); "
        "print(_common.PLOTS_DIR); print(_common.RUNS_DIR)"
    )
    env = {**os.environ, "LEAKNET_OUT": "out"}
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True)
    plots, runs = map(Path, res.stdout.splitlines()[-2:])
    assert plots == (tmp_path / "out" / "plots").resolve()
    assert runs == (tmp_path / "out" / "runs").resolve()
