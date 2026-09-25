# Claims → evidence

A check-list for the paper, poster, video and interview. Every number or claim you say out
loud should appear here with its evidence file. Status:

- **OK**: supported by a committed result. Say it with the stated caveat.
- **REWORD**: partly true. Use the narrower version.
- **REMOVE**: not supported by anything in the repo.
- **PENDING**: the experiment exists but hasn't been run yet.

Last reviewed: 23 Sep 2026 (after E1–E5 runs).

## Results

| Claim | Status | Evidence / what's actually true |
|---|---|---|
| AUROC 1.000 on 3 held-out networks | OK, with caveat | `results/test_results_c.json`. Held-out *synthetic* networks. Always show it next to E2: peak amplitude alone reaches 0.92–0.97 on the same data. |
| The model learned more than loudness (on synthetic data) | OK | E3: AUROC ≥ 0.97 at −15 dB, where the RMS detector is at chance (0.50–0.52). DC offset not used (E2 model_noDC = 1.000). |
| "Completely unseen real-world networks" | REMOVE | The pipe geometry comes from real benchmark networks, but no real acoustic data from those networks exists in this project. |
| Generalises because "the same physics operates everywhere" | REWORD | Interpretation, not something the project showed. The test only shows generalisation *within the synthesiser*. |
| Mendeley real-data AUROC ≈ 0.79 | REMOVE | No run produced it. `results/experiment2_results.json` = 0.515. The 0.79 appeared only as a pre-run expectation in a docstring (now corrected). |
| Mendeley F1 0.87 / accuracy 0.78 | REMOVE as headline | True numbers, but they come from 80% leak prevalence while the model flags 99% of no-leak windows. Report the detection rate and false-alarm rate instead. |
| Mendeley zero-shot AUROC 0.501 | REWORD | Use the E4 numbers instead: on the clean Looped test, 8 checkpoint/scaling combinations score 0.37–0.77 with wide CIs, and false alarms are 88–100%. Say "no reliable transfer". Don't quote the best single combination on its own; that's cherry-picking. |
| The real-data failure is caused by a scaling mismatch | REMOVE (tested, rejected) | E4: matched scaling cut false alarms only from 100% to 88–91%. Present it as a hypothesis you tested and rejected; that's a strength in the interview. |
| The model flags any unfamiliar noise texture as a leak | PENDING (E7) | Working hypothesis, from E1 + E4. Run E7 before stating it. |
| Synthetic pretraining reduces the real data needed | REMOVE (tested, opposite result) | E5: pretrained is *worse* than scratch at every budget ≥ 5% (negative transfer). Report that finding. |
| Leaks can be detected in the Mendeley data by simple features | PENDING (E6) | E4: RMS AUROC on Looped is 0.21 (inverted), and the Branched-trained band-energy classifier scores 0.10. E6 tests confounding by flow condition. |
| Only 4 no-leak recordings per topology | OK | E4 JSON `n_groups_no_leak = 4`. This is why every real-data CI is wide. Say it up front. |
| "Closing the gap only needs PVC-specific synthesis tuning; not a fundamental barrier" | REMOVE | Experiment 2 *was* PVC/Mendeley-matched synthesis and scored 0.515. |
| Mendeley testbed is "47 m, 50 mm PVC with flows of 0.18–0.47 L/s" (script.md) | REWORD | Per Aghashahi, Sela & Banks (2023), *Data in Brief* 48, 109148: 47 m of **152.4 mm** schedule-80 PVC. 0.18/0.47 L/s are *demand* flows; leak flows were 0.018–0.075 L/s. Accelerometers were sampled at 51.2 kS/s per the paper (our loader auto-detects the rate from timestamps). |
| Mendeley differs from municipal mains in material, scale and noise | OK | A real domain gap. Present it as *a* cause to test, not *the* proven cause. |
| PosMAE 0.0185 on L-TOWN | OK, with caveat | Synthetic only, dual-sensor leak samples, normalised by span. |
| The model localises using TDOA / "beats GCC-PHAT" | REMOVE | Model C's synthetic leak has no recoverable time delay (cross-correlation vs true TDOA r = 0.00; `tests/test_dataset_e.py`), so GCC-PHAT can't work on it (MAE ≈ 0.25 = always guessing the midpoint). The model must be using the level difference between sensors. Model E fixes the synthesiser; compare again there. |
| Severity R² 0.76 / 0.60 / 0.75 | OK, with caveat | Synthetic only. |

## Method

| Claim | Status | Evidence / what's actually true |
|---|---|---|
| Cross-Channel Attention is "computationally equivalent to cross-correlation / TDOA" | REMOVE | The gate is computed from time-averaged features and is constant over time (`model_C/model.py`; `tests/test_model.py::test_gate_is_constant_over_time`). It cannot see timing. |
| CCA is the key novelty | PENDING (ablation) | Run `--fusion concat` with 3 seeds. If there's no difference, say so. Either way it's a finding. |
| "No prior work uses this" / "first in WDN" / "without precedent" | REWORD | Say "we did not find prior work using…", and cite the closest work you did find (e.g. FiT-WST+, 2025). You didn't do a systematic search. |
| "No free tuning parameters" | REMOVE | Hand-set values: material centre frequencies and bandwidths, 0.6 channel correlation, SNR range 0.5–12 dB, the ×5000 Torricelli scaling, the 40/25/20/15 leak-type mix. |
| Uses a physics-based synthesiser (Torricelli amplitude, wave speed, attenuation) | REWORD | Model C/D: physics-*inspired*; E10 measured plastic attenuation at 0.6–2.5 dB/m, about 100× more than they apply. Model E: plastic attenuation **calibrated to measurements** (MDPE, applied to PVC); metals still uncalibrated. Say exactly that. |
| Real leak noise is measurable between two sensors on a plastic pipe | OK | E10 panel C: delay grows linearly with distance, wave speed 247–264 m/s (Sheffield MDPE). |
| No pipe metadata used at inference | OK | Scalars are zeroed in training and evaluation. |
| 3 test networks span "three continents" | REWORD | L-TOWN is based on Limassol, **Cyprus** (not the Czech Republic). Just name the three networks. |

## Hardware and impact

| Claim | Status | Evidence / what's actually true |
|---|---|---|
| Runs on an ESP32 at ~12 ms with a 9.4 MB model | REMOVE (or "design target") | Never built or measured. 2.47M float32 parameters ≈ 9.9 MB, which doesn't fit comfortably without int8 quantisation. |
| 14–40× cheaper than commercial correlators | REWORD | "Estimated parts cost of a proposed design, vs retail prices of certified products." Not a like-for-like comparison. |
| Sensor spec (SM-24 geophone, GPS timing ±30 ns, …) | REWORD | Proposed design, not built. |
| 126 billion m³/yr lost, US$39 billion/yr | OK | World Bank (Liemberger & Wyatt, 2019). Cite it. |

## Things you should be able to answer in the interview

1. Why is AUROC 1.000 not suspicious? (E2: peak amplitude gets 0.92–0.97. E3: the model still works at −15 dB, where loudness is useless.)
2. How does your model localise? (Not by timing: the Model C synthetic signal has no recoverable delay. By the level difference between sensors. Model E fixes this.)
3. Why did the model fail on real data, and how do you know? (Scaling was tested and rejected in E4. Texture hypothesis: E7. Tiny, confounded test set: E6.)
4. Did the Mendeley noise used in training leak into your test? (Yes for Branched no-leak, so the clean test is Looped only.)
5. Does synthetic pretraining help? (E5: no, it hurt. Why do you think that is?)
6. What exactly did each of you do? (Write your own contribution statement.)
