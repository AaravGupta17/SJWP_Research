# tests24 — experiment run summary

Run on 2026-09-24/25 from the repo root with `LEAKNET_OUT=tests24`, so plots and run records
went here instead of `plots/` and `results/runs/`. Numbers below are copied from the logs and run
records in this folder.

## Commands

| Log | Command | Status |
|---|---|---|
| `texture_probe.log` | `python experiments/texture_probe.py` | OK |
| `data_overview.log` | `python experiments/data_overview.py` | OK |
| `data_overview_hydrophone.log` | `python experiments/data_overview.py --sensor hydrophone --root datasets/Hydrophone/Hydrophone` | OK |
| `mendeley_eval_hydrophone_legacy.log` | `python experiments/mendeley_eval.py --sensor hydrophone --root datasets/Hydrophone/Hydrophone --legacy-file-norm` | OK |
| `realism_check.log` | `python experiments/realism_check.py` | OK (on rerun, see below) |
| — | `python experiments/cross_dataset.py` | Not run: this file is not in the repo |

## Contents

- `*.log`: full console output of each run
- `plots/`: `e6_data_overview_accelerometer.png`, `e6_data_overview_hydrophone.png`, `e8_realism_check.png`
- `runs/`: JSON run record for each experiment, plus `INDEX.csv`

## Results

AUROCs are shown with bootstrap 95% CIs over recordings where the script reports them.

### E7 texture probe (`runs/2026-09-25_000006_e7_texture_probe.json`)

No model flags the Mendeley no-leak noise ("bank") as a leak at training loudness (0%). Every
model flags 99–100% of the accelerometer recordings (Branched and Looped) and pure Gaussian noise
as leaks, at every RMS level tested (0.05, 0.1, 0.2).

### E6 data overview, accelerometer (`runs/2026-09-25_000032_e6_data_overview_accelerometer.json`)

- RMS as a leak score, Looped: pooled AUROC 0.207 [0.031, 0.378] (below 0.5: no-leak recordings are louder)
- RMS as a leak score, Branched: pooled AUROC 0.440 [0.206, 0.691]
- Band-energy logistic regression, leave-one-flow-condition-out, Looped: AUROC 0.866 [0.719, 0.997],
  detection 0.946, false alarm 0.251
- Same, Branched: AUROC 0.617 [0.392, 0.832], detection 0.650, false alarm 0.513

### E6 data overview, hydrophone (`runs/2026-09-25_000101_e6_data_overview_hydrophone.json`)

- RMS as a leak score, Looped: pooled AUROC 0.569 [0.404, 0.732]
- RMS as a leak score, Branched: pooled AUROC 0.683 [0.516, 0.838]
- Band-energy logistic regression, Looped: AUROC 0.767 [0.610, 0.905], detection 0.741, false alarm 0.395
- Same, Branched: AUROC 0.635 [0.405, 0.807], detection 0.708, false alarm 0.473

### E4 Mendeley eval, hydrophone, `--legacy-file-norm` (`runs/2026-09-25_000355_e4_mendeley_eval.json`)

5758 windows from 60 recordings (4561 leak windows). "Clean" means Looped-only, because the
Branched no-leak recordings were used as synthesiser background noise.

| Model | Scaling | Clean AUROC | Detection | False alarm | Saturated |
|---|---|---|---|---|---|
| `best_model_c_v4.pt` | zscore | 0.608 [0.506, 0.707] | 0.961 | 1.000 | 55% |
| `best_model_c_v4.pt` | fixed | 0.519 [0.265, 0.743] | 0.000 | 0.002 | 1% |
| `best_model_c_seed42.pt` | zscore | 0.235 [0.178, 0.296] | 1.000 | 1.000 | 80% |
| `best_model_c_seed42.pt` | fixed | 0.724 [0.481, 0.923] | 0.064 | 0.004 | 66% |
| `best_model_d.pt` | zscore | 0.379 [0.311, 0.446] | 0.851 | 0.927 | 43% |
| `best_model_d.pt` | fixed | 0.887 [0.833, 0.926] | 0.000 | 0.000 | 0% |
| `best_model_mend.pt` | zscore | 0.164 [0.075, 0.280] | 0.000 | 0.000 | 100% |
| `best_model_mend.pt` | fixed | 0.220 [0.110, 0.353] | 0.000 | 0.000 | 100% |
| RMS baseline | — | 0.616 [0.471, 0.755] | | | |
| GCC peak baseline | — | 0.585 [0.526, 0.635] | | | |
| Band-energy logreg (Branched → Looped) | — | 0.182 [0.110, 0.263] | 0.065 | 0.228 | |

The highest clean AUROC is `best_model_d.pt` with fixed scaling (0.887), but at the default
threshold it detects 0% of leaks: it ranks recordings but never calls a leak. The z-scored models
have false-alarm rates of 93–100%.

### E8 realism check (`runs/2026-09-25_000509_e8_realism_check.json`)

A classifier trained to tell real Mendeley windows from synthetic ones (0.5 = indistinguishable):

- Model C synthesiser: no-leak 1.000, leak 1.000
- Model E synthesiser: no-leak 0.994, leak 0.998

The synthetic data is still easy to tell apart from the real recordings.

## Issue during the run

`realism_check.py` crashed the first time with `FileNotFoundError` when saving
`tests24\plots\e8_realism_check.png`. The script calls `os.chdir(model_C)` partway through, so the
relative `LEAKNET_OUT=tests24` no longer resolved. `snr_sweep.py` changes directory the same way.
It was rerun with an absolute `LEAKNET_OUT` path and finished normally; `realism_check.log` is
from the rerun. The permanent fix would be to resolve `LEAKNET_OUT` to an absolute path in
`experiments/_common.py`; this has since been done, so a relative `LEAKNET_OUT` now works.

User-folder paths in the logs and run records were replaced with `<repo>` after the run.
