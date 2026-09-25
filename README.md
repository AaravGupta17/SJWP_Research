# AcousticLeakNet

A two-sensor acoustic leak detector for water pipes, trained on physics-based synthetic
waveforms generated from EPANET hydraulic simulations, and tested both on held-out
synthetic networks and on real pipe-testbed recordings.

**Research question:** can a leak detector trained only on synthetic acoustics transfer to
real recordings, what causes it to fail when it does not, and does synthetic pretraining
reduce the amount of real labelled data needed?

## Current status of results

All numbers link to run records in `results/runs/` (23 Sep 2026). Real-data AUROCs are computed
on logits, with 95% CIs from a bootstrap over recordings.

**Synthetic data (in-domain)**

| Test | Result | Evidence |
|---|---|---|
| Held-out synthetic networks, Model C | AUROC 1.000 on L-TOWN, KY15, Richmond | `results/test_results_c.json` |
| Trivial features on the same test sets (E2) | Peak amplitude 0.92–0.97, RMS 0.77–0.85; DC offset not used (removing it leaves AUROC 1.000) | `results/runs/2026-09-23_202204_e2_shortcut_audit.json` |
| Leak SNR sweep (E3) | Model AUROC ≥ 0.97 down to −15 dB, 0.84–0.90 at −20 dB; RMS detector ≈ 0.50–0.53 (chance) at ≤ −10 dB | `results/runs/2026-09-23_202425_e3_snr_sweep.json`, `plots/e3_snr_sweep.png` |
| Localisation (E3) | Model position MAE 0.016–0.094 at native SNR; GCC-PHAT ≈ 0.25 = always guessing the midpoint | same |
| Is the synthetic time delay recoverable? | No: cross-correlation lag vs true TDOA r = 0.00 in noise-free Model C leaks (r = 1.00 once the zero-delay shared component is removed). The model's localisation cannot be timing-based | `tests/test_dataset_e.py::test_model_c_synthetic_leak_has_no_recoverable_tdoa` |

**Real recordings (Mendeley accelerometers; clean test = Looped: 16 leak, 4 no-leak recordings)**

| Test | Result | Evidence |
|---|---|---|
| Original evaluation (legacy) | AUROC 0.501, ~99% false alarms | `results/mendeley_accelerometer_results.json` |
| Zero-shot, logit AUROC (E4) | 8 checkpoint × scaling combinations: 0.37–0.77. Only 2 have a CI lower bound above 0.5 (C seed42 fixed 0.73 [0.56, 0.88]; D z-score 0.77 [0.54, 0.94]), and the same architecture with another seed (C v4) scores 0.40. These are the best 2 of 8 and are not consistent, so they are not evidence of transfer. False alarms 88–100% at the default threshold in every case | `results/runs/2026-09-23_202146_e4_mendeley_eval.json` |
| Training-matched input scaling (E4) | False alarms only 100% → 88–91%: scaling is **not** the main cause | same |
| Loudness as a leak score (E4) | Looped RMS AUROC 0.21 (no-leak recordings are louder); band-energy classifier trained on Branched scores 0.10 on Looped | same |
| Label efficiency, Branched → Looped (E5) | No method clearly above chance; synthetic pretraining is **worse** than training from scratch at every budget ≥ 5% (≈0.2–0.3 vs ≈0.4–0.56) | `results/runs/2026-09-23_204135_e5_label_efficiency.json`, `plots/e5_label_efficiency.png` |
| Leak-free noise into the checkpoints (E1) | Model C/D flag pure Gaussian noise as a leak from RMS 0.05 | `results/runs/2026-09-23_185508_e1_loudness_probe.json` |

**Measured plastic-pipe acoustics (E10, Sheffield MDPE)**

| Quantity | Measured | Model C "PVC" | Evidence |
|---|---|---|---|
| Attenuation, 20–400 Hz | 0.64–0.92 dB/m (median over 8 leak tests) | ≈ 0.004 dB/m at α = 0.001/m | `results/runs/2026-09-23_231420_e10_sheffield_calibration.json`, `plots/e10_sheffield_calibration.png` |
| Attenuation, 400–800 Hz | 2.47 dB/m | same | same |
| Above 800 Hz | reaches the noise floor within 1–3 m; slope fits unreliable | same | same |
| Leak spectrum at the leak | peak ~471 Hz, centroid ~770 Hz (median) | 650–950 Hz band | same |
| Wave speed | 247 and 264 m/s (two leaks, 20 sensor distances each) | taken from EPANET per material | same |

How to read these:

- In its own domain, the model learned more than loudness: it detects leaks at SNRs where an
  energy detector is at chance. But the synthetic leak carries no usable time delay, so
  localisation must come from the level difference between the two sensors.
- On real recordings, nothing tested separates leak from no-leak reliably. Fixing the scaling
  mismatch did not help. The current working hypothesis is that the synthetic no-leak class
  contained a single noise texture (one set of hydrophone recordings), so any unfamiliar
  noise looks like a leak. E7 tests this, and Model E is the corresponding fix.
- The clean real test has only 4 no-leak recordings, and loudness is inverted between leak and
  no-leak in Looped. E6 checks whether this comes from confounding by flow condition. The
  dataset is too small to support strong real-world claims either way.

## Repository layout

```
model_C/          Model C: synthesiser (dataset_c.py), model, training, evaluation  ← main model
Model_D/          Model D: Model C + extra realism (leak types, attenuation, noise)
Model_E/          Model E: synthesiser fixes targeting the diagnosed failures (dataset_e.py)
Model_F/          Model F: new training data and objective (real backgrounds, no loudness cue)
scripts/          Model B pipeline + index building from EPANET output
baselines/        Classical detectors on the synthetic caches: RMS energy,
                  cross-correlation, GCC-PHAT
experiments/      E1–E8 (see below); every run is logged to results/runs/
tests/            Unit tests (pytest), no data needed
data/inp, data/csv  EPANET networks and sample index files (Git LFS)
models/           Checkpoints (Git LFS)
results/          Metrics JSON; results/runs/ = one dated JSON record per run
plots/            Figures
docs/             CLAIMS.md (claim → evidence), INTEGRITY_LOG.md (corrections made),
                  DATA_BOOK.md (dated run log; `python scripts/make_data_book.py`)
archive/          Superseded scratch scripts, kept for history only
```

`model_C/model.py` is the canonical architecture (`Model_D/model.py` is an identical copy).

## Data

| Data | Source | Location |
|---|---|---|
| EPANET networks | Public benchmark networks; `L-TOWN.inp` is the BattLeDIM network (based on Limassol, Cyprus; Vrachimis et al., 2022). It is >100 MB and not committed: download it into `data/inp/`. | `data/inp/` |
| Simulation CSVs | Generated by `scripts/inptocsv.py` | `datasets/NetworkList/` (not committed) |
| Real recordings | Mendeley leak-detection testbed dataset (Aghashahi, Sela & Banks, *Data in Brief* 48, 109148, 2023): accelerometers and hydrophones on a looped and a branched 152.4 mm PVC testbed | `datasets/Accelerometer/`, `datasets/Hydrophone/` (not committed) |
| Real buried networks | Hong Kong (Tijani, Tariq, Zayed et al., Mendeley Data doi:10.17632/hkn8mxcjyz.1, 2022, CC BY 4.0): noise loggers (4096 Hz) and hydrophones (4096 Hz) at real leak sites and no-leak sites, metal and plastic pipes. The MEMS accelerometer files are not used: their sampling rate is not stated anywhere we could find | `datasets/public/hongkong/` |
| Outdoor training base | Dongguan (Wang, Mei, Zhan & Chen, Zenodo doi:10.5281/zenodo.18631450, 2026, CC BY 4.0): 1 s leak / no-leak clips, ductile iron, PE, steel, PVC | `datasets/public/dongguan/` |
| Calibrated plastic-pipe acoustics | Sheffield CID lab (Shekofteh, ORDA doi:10.15131/shef.data.32229270.v1, 2026, CC BY 4.0; *Sensors* 2026): 63 mm MDPE, calibrated accelerometers at 0–50 m from the leak, simultaneous sensor pairs | `datasets/public/sheffield/` |

Download the three public datasets with `python scripts/download_public_data.py` (about 750 MB; RAR extraction uses the `tar` built into Windows 10/11).

The Model C/D synthesiser uses the Mendeley **Branched no-leak hydrophone** recordings as
background noise. Every experiment that tests on Mendeley data therefore treats Branched
no-leak as contaminated and reports **Looped-only** as the clean test.

## Setup

```bash
pip install -r requirements.txt
git lfs install && git lfs pull
python -m pytest tests          # 1–3 min on CPU, no data needed
```

## Pipeline (Model C)

```bash
cd model_C
python pregen_c.py --split all            # synthesise waveforms into ../cache_c
python train_c.py --seed 42               # add --fusion concat for the gating ablation
python evaluate_c.py --ckpt best_model_c_v4.pt
```

## Model E (synthesiser fixes)

`Model_E/dataset_e.py` subclasses Model C's synthesiser. It uses physical fractional delays (no
zero-delay shared component, no wrap-around), diverse no-leak noise textures, non-leak
interferers (bursts, pump hum), an accelerometer-like response, ±6 dB gain jitter, a −10 to
12 dB leak SNR range, no DC offset, and it drops leak rows that have no leak source. For plastic
(PVC) pipes it applies the attenuation **measured** in E10 (`Model_E/calibration_mdpe.json`,
0.64–2.47 dB/m by frequency) instead of Model C's ≈0.004 dB/m. Metal pipes keep Model C's
attenuation (no metal measurement available). Each change is a switch, for ablations
(`pregen_e.py --no <switch>`). `--drop-inaudible-db` optionally drops plastic leak rows too
far from both sensors to be heard.

```bash
cd Model_E
python pregen_e.py --split all            # -> ../cache_e (seeded, reproducible)
cd ../model_C
python train_c.py --cache cache_e --prefix e --seed 42
python evaluate_c.py --ckpt best_model_e_seed42.pt --cache cache_e
python evaluate_c.py --ckpt best_model_c_v4.pt  --cache cache_e   # Model C on the harder data
```
Then run E4, E5 and E7 with `--ckpts best_model_e_seed42.pt` / `--ckpt best_model_e_seed42.pt`.

## Model F (new training data and objective)

Same architecture, new data and training, each change aimed at a diagnosed failure (see
`docs/MODEL_F_PREREG.md`, written before any Model F result). Every training sample is mixed on
the fly: a pre-generated Model E leak signal over a fresh background (real no-leak recordings from
Hong Kong, Dongguan and Mendeley Branched, or synthetic noise), plus interferers, random EQ, a
2 kHz band limit and a joint z-score, so absolute loudness is not available. Real labelled windows
from the training sources make up 30% of rows. Checkpoints are chosen by detection AUROC on
held-out groups. No Mendeley label and no Mendeley Looped recording is used.

```bash
python Model_F/bank_f.py                    # real windows -> cache_f/bank.npz
python Model_F/pregen_f.py                  # leak signals -> cache_f/{train,val}
python Model_F/audit_f.py                   # shortcut check of the training data (H0)
python Model_F/train_f.py                   # -> models/best_model_f_seed0.pt
python experiments/model_f_eval.py --ckpt best_model_f_seed0.pt   # E11
```

## Experiments

Run from the repository root. Each script prints its results, saves a plot to `plots/`, and
writes a dated record (config, results, git commit) to `results/runs/`.

| ID | Question | Needs | Command |
|---|---|---|---|
| E1 | Does the network just threshold loudness / DC offset? | checkpoints only | `python experiments/loudness_probe.py` |
| E2 | Can trivial features (RMS, DC, clipping) already solve the synthetic test sets? | `cache_c` | `python experiments/shortcut_audit.py` |
| E3 | How do the model, RMS and GCC-PHAT degrade as leak SNR drops? | EPANET CSVs + noise bank | `python experiments/snr_sweep.py --no-dc` |
| E4 | Zero-shot real-data performance with matched scaling and a clean test set, vs classical baselines | Mendeley accelerometer | `python experiments/mendeley_eval.py` |
| E5 | Does synthetic pretraining reduce the real labelled data needed? | Mendeley accelerometer, GPU | `python experiments/label_efficiency.py` |
| B  | Classical baselines on the synthetic caches | `cache_c` | `python baselines/energy.py`, `gccphat.py`, `crosscorr.py` |
| A  | Does cross-channel gating help? (3 seeds each) | `cache_c`, GPU | `model_C/run_seeds.ps1` with `$fusion = "concat"`, then `evaluate_c.py --ckpt ...` |
| E6 | What separates leak from no-leak in the real data? Confounding by flow condition? | Mendeley | `python experiments/data_overview.py` (add `--sensor hydrophone --root datasets/Hydrophone/Hydrophone` for hydrophones) |
| E7 | Does the model flag every unfamiliar noise texture as a leak? | Mendeley hydrophone + accelerometer | `python experiments/texture_probe.py` |
| E8 | How distinguishable are synthetic windows from real ones (Model C vs E)? | EPANET CSVs + noise bank + Mendeley | `python experiments/realism_check.py` |
| E9 | Does single-channel leak detection transfer to a real network it has never seen? (within-dataset CV and leave-one-source-out; logreg, CNN from scratch, CNN from the synthetic encoder, frozen-encoder probe) | public datasets + Mendeley, GPU | `python experiments/cross_dataset.py` |
| E10 | Measured plastic-pipe attenuation, leak spectrum and wave speed, compared with the synthesiser's values | Sheffield data | `python experiments/sheffield_calibration.py` |
| E11 | Model F on data it never heard: Mendeley Looped (both sensors), held-out public sources, noise sanity; logreg baseline on the same windows | Mendeley + public datasets | `python experiments/model_f_eval.py --ckpt ...` |

Metrics on real data are AUROC, detection rate, false-alarm rate and balanced accuracy, with
95% confidence intervals from a bootstrap over **recordings** (windows from one recording are
not independent). F1 and accuracy are not used as headline metrics because the real data is
80% leak windows.

## Known limitations

- No physical hardware was built or tested; any on-device (ESP32) latency or cost figures are
  design estimates, not measurements.
- The synthesiser has hand-set parameters (material centre frequencies, 0.6 channel
  correlation, SNR range, Model D leak-type mix).
- The cross-channel gating block is time-constant (squeeze-and-excitation style); it does not
  perform time alignment or cross-correlation.
- The real-data test uses one laboratory testbed (PVC, short pipes), which differs from buried
  municipal mains in material, scale and noise.

## License

Not yet chosen.
