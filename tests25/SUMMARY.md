# tests25 — experiment run summary

Run on 2026-09-25 from the repo root on `main` at `d9d9a21`, with `LEAKNET_OUT=tests25`, so plots
and run records went here instead of `plots/` and `results/runs/`. Numbers below are copied from
the logs and run records in this folder.

## Commands (in the order run)

| Log | Command | Status | Time |
|---|---|---|---|
| `download_public_data.log` | `python scripts/download_public_data.py` | OK | ~6 min |
| `realism_check.log` | `python experiments/realism_check.py` | OK | ~30 s |
| `cross_dataset_quick.log` | `python experiments/cross_dataset.py --quick` | OK | ~7 min |
| `cross_dataset.log` | `python experiments/cross_dataset.py` | OK | ~30 min |
| `cross_dataset_model_d.log` | `python experiments/cross_dataset.py --ckpt best_model_d.pt` | OK | ~32 min |

## Contents

- `*.log`: console output of each run. User-folder paths are replaced with `<repo>/`; progress-bar
  updates appear on separate lines. No numbers changed.
- `plots/e8_realism_check.png`
- `runs/`: JSON run record for each experiment
  (`2026-09-25_130813_e8_realism_check.json`, `2026-09-25_131514_e9_cross_dataset.json` (`--quick`),
  `2026-09-25_134512_e9_cross_dataset.json` (full, default checkpoint),
  `2026-09-25_141706_e9_cross_dataset.json` (`--ckpt best_model_d.pt`))

## Public data download

Hong Kong (noise loggers, hydrophones, MEMS accelerometers), Sheffield (one 492.9 MB zip) and
Dongguan (three `.rar` archives) were downloaded to `datasets/public/` (git-ignored).

## E8 realism check

A classifier trained to tell real Mendeley windows from synthetic ones (0.5 = indistinguishable):

- Model C synthesiser: no-leak 1.000, leak 1.000
- Model E synthesiser: no-leak 0.994, leak 0.997

The synthetic data is still easy to tell apart from real recordings. This matches the tests24
run (0.994 / 0.998 for Model E).

## E9 cross-dataset

Five real datasets, resampled to 5 kHz and band-limited to 2000 Hz:

| Dataset | Windows | Groups | Leak share |
|---|---|---|---|
| hk_noiselogger | 2175 | 32 | 46% |
| hk_hydrophone | 2000 | 18 | 50% |
| dongguan | 1772 | 293 | 56% |
| mendeley_acc | 7148 | 40 | 80% |
| mendeley_hyd | 11516 | 60 | 79% |

Methods:

- `logreg`: logistic regression on band energies
- `cnn_scratch`: the CNN trained from scratch on the real data
- `cnn_pre`: the CNN initialised from a synthetic-data checkpoint, then trained on the real data
- `probe`: a linear classifier on the frozen synthetic-pretrained CNN

`cnn_pre` and `probe` use `best_model_c_v4.pt` by default, and `best_model_d.pt` in the `--ckpt`
run. AUROC is shown with a bootstrap 95% CI.

### Within-dataset (5-fold grouped CV), AUROC

| Dataset | Method | Full, C v4 | Full, Model D | Quick, C v4 |
|---|---|---|---|---|
| hk_noiselogger | logreg | 0.585 [0.416, 0.789] | 0.585 [0.416, 0.789] | 0.585 |
| hk_noiselogger | cnn_scratch | 0.425 [0.266, 0.655] | 0.427 [0.268, 0.643] | 0.437 |
| hk_noiselogger | cnn_pre | 0.444 [0.271, 0.635] | 0.364 [0.232, 0.575] | 0.398 |
| hk_noiselogger | probe | 0.460 [0.311, 0.639] | 0.540 [0.378, 0.711] | 0.545 |
| hk_hydrophone | logreg | 0.900 [0.779, 0.993] | 0.900 [0.779, 0.993] | 0.900 |
| hk_hydrophone | cnn_scratch | 0.939 [0.877, 0.987] | 0.944 [0.886, 0.991] | 0.886 |
| hk_hydrophone | cnn_pre | 0.958 [0.906, 0.999] | 0.931 [0.862, 0.988] | 0.913 |
| hk_hydrophone | probe | 0.849 [0.701, 0.962] | 0.745 [0.510, 0.972] | 0.802 |
| dongguan | logreg | 0.891 [0.836, 0.933] | 0.891 [0.836, 0.933] | 0.891 |
| dongguan | cnn_scratch | 0.971 [0.954, 0.989] | 0.971 [0.955, 0.989] | 0.975 |
| dongguan | cnn_pre | 0.968 [0.950, 0.987] | 0.972 [0.955, 0.987] | 0.968 |
| dongguan | probe | 0.965 [0.941, 0.983] | 0.929 [0.906, 0.953] | 0.955 |
| mendeley_acc | logreg | 0.666 [0.532, 0.787] | 0.666 [0.532, 0.787] | 0.666 |
| mendeley_acc | cnn_scratch | 0.847 [0.779, 0.910] | 0.846 [0.776, 0.911] | 0.796 |
| mendeley_acc | cnn_pre | 0.864 [0.800, 0.922] | 0.846 [0.775, 0.909] | 0.842 |
| mendeley_acc | probe | 0.813 [0.758, 0.872] | 0.817 [0.732, 0.892] | 0.759 |
| mendeley_hyd | logreg | 0.632 [0.527, 0.746] | 0.632 [0.527, 0.746] | 0.632 |
| mendeley_hyd | cnn_scratch | 0.779 [0.663, 0.884] | 0.782 [0.666, 0.885] | 0.702 |
| mendeley_hyd | cnn_pre | 0.854 [0.773, 0.925] | 0.815 [0.726, 0.898] | 0.727 |
| mendeley_hyd | probe | 0.690 [0.517, 0.854] | 0.642 [0.455, 0.828] | 0.639 |

### Leave-one-source-out (train on the other sources, test on the held-out one)

AUROC, then detection rate / false-alarm rate at the default threshold.

| Held-out test set | Method | Full, C v4 | Full, Model D |
|---|---|---|---|
| dongguan | logreg | 0.722 [0.651, 0.792] · 0.42 / 0.13 | 0.722 [0.651, 0.792] · 0.42 / 0.13 |
| dongguan | cnn_scratch | 0.627 [0.502, 0.718] · 0.48 / 0.30 | 0.636 [0.513, 0.722] · 0.50 / 0.29 |
| dongguan | cnn_pre | 0.625 [0.543, 0.692] · 0.49 / 0.26 | 0.678 [0.616, 0.746] · 0.33 / 0.10 |
| dongguan | probe | 0.690 [0.640, 0.745] · 0.51 / 0.17 | 0.479 [0.400, 0.540] · 0.45 / 0.42 |
| hk_hydrophone | logreg | 0.193 [0.041, 0.358] · 0.53 / 0.96 | 0.193 [0.041, 0.358] · 0.53 / 0.96 |
| hk_hydrophone | cnn_scratch | 0.883 [0.762, 0.977] · 0.96 / 0.62 | 0.874 [0.750, 0.973] · 0.96 / 0.61 |
| hk_hydrophone | cnn_pre | 0.497 [0.294, 0.736] · 0.47 / 0.44 | 0.874 [0.759, 0.967] · 0.96 / 0.63 |
| hk_hydrophone | probe | 0.340 [0.104, 0.606] · 0.22 / 0.25 | 0.746 [0.505, 0.938] · 0.82 / 0.68 |
| hk_noiselogger | logreg | 0.442 [0.291, 0.661] · 0.29 / 0.43 | 0.442 [0.291, 0.661] · 0.29 / 0.43 |
| hk_noiselogger | cnn_scratch | 0.528 [0.297, 0.666] · 0.81 / 0.85 | 0.531 [0.300, 0.665] · 0.80 / 0.86 |
| hk_noiselogger | cnn_pre | 0.447 [0.194, 0.582] · 0.80 / 0.79 | 0.587 [0.365, 0.719] · 0.74 / 0.69 |
| hk_noiselogger | probe | 0.426 [0.217, 0.568] · 0.65 / 0.81 | 0.611 [0.459, 0.747] · 0.57 / 0.41 |
| mendeley_acc | logreg | 0.512 [0.370, 0.652] · 0.48 / 0.44 | 0.512 [0.370, 0.652] · 0.48 / 0.44 |
| mendeley_acc | cnn_scratch | 0.468 [0.349, 0.590] · 0.35 / 0.42 | 0.467 [0.347, 0.589] · 0.35 / 0.42 |
| mendeley_acc | cnn_pre | 0.537 [0.436, 0.646] · 0.35 / 0.35 | 0.525 [0.402, 0.650] · 0.35 / 0.35 |
| mendeley_acc | probe | 0.606 [0.502, 0.720] · 0.40 / 0.34 | 0.547 [0.428, 0.669] · 0.40 / 0.36 |
| mendeley_hyd | logreg | 0.550 [0.441, 0.650] · 0.59 / 0.52 | 0.550 [0.441, 0.650] · 0.59 / 0.52 |
| mendeley_hyd | cnn_scratch | 0.617 [0.554, 0.678] · 0.38 / 0.19 | 0.621 [0.554, 0.686] · 0.39 / 0.19 |
| mendeley_hyd | cnn_pre | 0.616 [0.539, 0.695] · 0.54 / 0.28 | 0.633 [0.541, 0.724] · 0.05 / 0.02 |
| mendeley_hyd | probe | 0.647 [0.548, 0.739] · 0.60 / 0.28 | 0.493 [0.430, 0.555] · 0.00 / 0.02 |

When Hong Kong is held out, the training set is dongguan + mendeley_acc + mendeley_hyd. When
Mendeley is held out, it is dongguan + hk_hydrophone + hk_noiselogger. The `--quick` run's
leave-one-source-out numbers are in `cross_dataset_quick.log` and show the same pattern.

### What this shows

- Training and testing within one dataset works well for Dongguan (AUROC ~0.97) and the Hong Kong
  hydrophones (0.90–0.96). The Hong Kong noise loggers are near chance for every method.
- Testing on a source not seen in training mostly fails: most AUROCs are 0.45–0.65.
  Exceptions:
  - Dongguan held out: `logreg` reaches 0.722 with a 13% false-alarm rate.
  - Hong Kong hydrophone held out: the CNNs reach ~0.87–0.88, but with ~60% false alarms at
    the default threshold. `logreg` scores 0.193 there, meaning its leak score is inverted on
    that source.
- Model D vs Model C v4 pretraining:
  - Model D helps `cnn_pre` on the Hong Kong hydrophones (0.874 vs 0.497).
  - Model D hurts the `probe` on Dongguan (0.479 vs 0.690).
  - Elsewhere the two are close, and most CIs overlap.
- None of this shows the model generalising to real pipes it was not trained on.
