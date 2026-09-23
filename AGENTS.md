# AGENTS.md

Guidance for coding agents working in this repository.

## Project

AcousticLeakNet: a two-sensor acoustic leak detector for water pipes. It is trained on
synthetic waveforms (EPANET hydraulics + a physics-based acoustic synthesiser) and tested on
held-out synthetic networks and on real Mendeley testbed recordings. This is a student
research / science-fair project, so **evidence integrity matters more than results**. See
`README.md`, `docs/CLAIMS.md` (claim → evidence) and `docs/INTEGRITY_LOG.md` (corrections).

## Environment

- Windows, Python 3.12 venv at `.venv/` (`.venv/Scripts/python.exe`), torch 2.7 + CUDA.
- Run everything from the repo root. Tests: `.venv/Scripts/python.exe -m pytest tests`
  (1–3 min, CPU, no data needed).
- Large data is local only and git-ignored: `datasets/` (Mendeley Accelerometer/Hydrophone,
  EPANET `NetworkList` CSVs), `cache_c/`, `cache_d/`, etc. Checkpoints in `models/` are Git LFS.

## Layout

- `model_C/` main model: `dataset_c.py` (synthesiser), `model.py` (canonical architecture),
  `pregen_c.py`, `train_c.py`, `evaluate_c.py`. `Model_D/` = Model C + extra realism.
  `Model_E/` = synthesiser fixes (`dataset_e.py`, `pregen_e.py`); trained with
  `train_c.py --cache cache_e --prefix e`.
- `experiments/` E1–E8 (loudness probe, shortcut audit, SNR sweep, Mendeley eval, label
  efficiency, data overview, texture probe, realism check). Shared helpers in `experiments/_common.py`; metrics in `experiments/metrics.py`.
- `baselines/` classical detectors (RMS energy, cross-correlation, GCC-PHAT).
- `scripts/` older Model B pipeline and EPANET → CSV index building.
- `results/` metrics JSON; `results/runs/` dated run records + `INDEX.csv`. `plots/` figures.
- `archive/` superseded scripts, history only — do not edit or import.
- `testfies09/` raw text logs of experiment runs.

## Experiment commands

| ID | Command |
|---|---|
| E1 | `python experiments/loudness_probe.py` |
| E2 | `python experiments/shortcut_audit.py` |
| E3 | `python experiments/snr_sweep.py --no-dc` |
| E4 | `python experiments/mendeley_eval.py` |
| E5 | `python experiments/label_efficiency.py` (GPU; `--quick` for a smoke run) |
| E6 | `python experiments/data_overview.py` |
| E7 | `python experiments/texture_probe.py` |
| E8 | `python experiments/realism_check.py` |

Every experiment writes a record via `record_run()` to `results/runs/`. Set `LEAKNET_OUT=<dir>`
to redirect plots and run records for smoke tests so they don't mix with real results.

## Rules that must hold

- Compute AUROC on **logits** (`predict_proba(..., logits=True)`), never on sigmoid
  probabilities — they saturate to exactly 0/1 and ties drag AUROC toward 0.5.
- Mendeley **Branched no-leak** recordings were used as synthesiser background noise, so they
  are contaminated. The clean real-data test is **Looped-only**.
- Real-data metrics: AUROC, detection rate, false-alarm rate, balanced accuracy, with bootstrap
  CIs over **recordings**, not windows. Don't headline F1/accuracy (data is 80% leak).
- Input scaling must match training (fixed reference scale, leak-free RMS ≈ 0.1), not
  per-window z-scoring.
- Never write an expected result into a docstring or doc as if it were measured. Any new number
  in docs must point to a file in `results/`. Log fixed mistakes in `docs/INTEGRITY_LOG.md`.
- No identifying information (school name, city) in committed files — competition rules.
- Don't claim the model generalises to real pipes; the synthetic 1.000 AUROC is in-domain only.

## Git

- Commit author: `ArmaanGuha <armaanguha@gmail.com>`.
- Do not add AI attribution anywhere: no `Co-Authored-By` trailers, session links,
  "Generated with" lines, or tool names in commits, PRs, code or docs.
- Only commit or push when explicitly asked; branch off `main` first.
