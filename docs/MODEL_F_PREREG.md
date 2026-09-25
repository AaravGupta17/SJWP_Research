# Model F: pre-registration

Written on 25 Sep 2026, **before** any Model F model was trained or tested. The hypotheses,
tests and pass/fail rules below are fixed. Results are reported whether they pass or fail. If
anything is changed after test results are seen, the changed model gets a new name and the change
is recorded in `docs/INTEGRITY_LOG.md`.

## Why a new model

Diagnosis from E1–E10 (evidence in the files named):

| Finding | Evidence |
|---|---|
| The network is not too small: Model C reaches AUROC 1.000 on held-out synthetic networks, and the same encoder trained on real Dongguan data reaches 0.971 within that dataset | `results/test_results_c.json`; `tests25/runs/2026-09-25_134512_e9_cross_dataset.json` |
| The synthetic task has a shortcut: peak amplitude alone scores 0.92–0.97 | `results/runs/2026-09-23_202204_e2_shortcut_audit.json` |
| Models C/D flag leak-free noise as a leak: Gaussian noise 100%, accelerometer no-leak recordings 99–100% | `tests24/runs/2026-09-25_000006_e7_texture_probe.json` |
| Every synthetic background came from 6 Mendeley recordings, and synthetic data is trivially told apart from real (0.994–1.000) | `tests25/runs/2026-09-25_130813_e8_realism_check.json` |
| Real sources do not transfer to each other (leave-one-source-out AUROC mostly 0.45–0.65) | `tests25/runs/2026-09-25_134512_e9_cross_dataset.json` |
| Checkpoints were chosen by localisation score, never by detection; learned loss weights let logits saturate | `model_C/train_c.py`; INTEGRITY_LOG #12 |

Model F keeps the architecture and changes the data and training (`Model_F/`, [F1]–[F5] in
`Model_F/augment_f.py`): fresh real and synthetic backgrounds for every sample, loudness removed
by a joint z-score, a 2 kHz band limit everywhere, random EQ, randomised channel coherence, real
labelled windows from the training sources, fixed loss weights with label smoothing, and
checkpoint selection by detection AUROC on held-out groups.

## Models to train

| Name | Command | Trained on real data from |
|---|---|---|
| `f` (main) | `python Model_F/train_f.py` | Hong Kong, Dongguan |
| `f_nohk` | `python Model_F/train_f.py --exclude-source hongkong --prefix f_nohk` | Dongguan |
| `f_nodg` | `python Model_F/train_f.py --exclude-source dongguan --prefix f_nodg` | Hong Kong |
| `f_synonly` (ablation) | `python Model_F/train_f.py --real-frac 0 --prefix f_synonly` | none (real backgrounds only) |

No model sees any Mendeley label or any Mendeley Looped recording. Seed 0 for all. Default
hyperparameters in `train_f.py` are used unchanged.

## Hypotheses and pass rules

All tested by E11 (`experiments/model_f_eval.py`); AUROC on logits, 95% CIs from a bootstrap over
recordings/sites.

- **H0, no shortcut in the training data.** In `Model_F/audit_f.py` on the real cache, every
  trivial feature on synthetic rows has AUROC < 0.65. *(Checked first; if it fails, stop and fix the
  data before training.)*
- **H1, the noise shortcut is gone.** `f` flags less than 20% of band-limited white noise and of
  coloured noise as a leak (Models C/D: 100%).
- **H2, a new real network (Mendeley Looped).** For `f`, on both `mendeley_looped_acc` and
  `mendeley_looped_hyd`: the AUROC CI lower bound is above 0.5 **and** the AUROC is above the
  logreg baseline trained on the same real sources. Passing on one sensor only is reported as a
  partial pass.
- **H3, held-out public sources.** `f_nohk` on `hk_hydrophone` and `hk_noiselogger`, and `f_nodg`
  on `dongguan`: Model F's AUROC beats the same-source logreg baseline in at least 2 of the 3
  tests.
- **H4, real labelled data helps (ablation).** `f` has a higher Mendeley Looped AUROC than
  `f_synonly` on both sensors. This is exploratory; no pass rule.

What would count against Model F: H1 failing means the shortcut survived; H2 and H3 failing means
the added realism did not produce transfer. Either is reported as a result.

## Caveats known in advance

- Mendeley has 4–6 no-leak recordings per topology and sensor, so its CIs are wide.
- The public datasets have one sensor. E11 pairs two windows of the same recording into two
  channels, as in training. Those tests measure single-sensor sound recognition, not the
  two-sensor timing idea; only Mendeley tests both sensors.
- One seed per model. GPU non-determinism alone moves AUROC by about 0.01 (INTEGRITY_LOG #22).
