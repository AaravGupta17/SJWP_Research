# Claims → evidence

A check-list for the paper, poster, video and interview. Every number or claim you say out
loud should appear here with its evidence file. Status:

- **OK**: supported by a committed result. Say it with the stated caveat.
- **REWORD**: partly true. Use the narrower version.
- **REMOVE**: not supported by anything in the repo.
- **PENDING**: the experiment exists but hasn't been run yet.

Last reviewed: 23 Sep 2026.

## Results

| Claim | Status | Evidence / what's actually true |
|---|---|---|
| AUROC 1.000 on 3 held-out networks | OK, with caveat | `results/test_results_c.json`. The networks are held out, but the waveforms come from the training synthesiser. Call it "held-out *synthetic* networks". Pair it with the E2/E3 results. |
| "Completely unseen real-world networks" | REMOVE | The pipe geometry comes from real benchmark networks, but no real acoustic data from those networks exists in this project. |
| Generalises because "the same physics operates everywhere" | REWORD | Interpretation, not something the project showed. The test only shows generalisation *within the synthesiser*. |
| Mendeley real-data AUROC ≈ 0.79 | REMOVE | No run produced it. `results/experiment2_results.json` = 0.515. The 0.79 appeared only as a pre-run expectation in a docstring (now corrected). |
| Mendeley F1 0.87 / accuracy 0.78 | REMOVE as headline | True numbers, but they come from 80% leak prevalence while the model flags 99% of no-leak windows. Report the detection rate and false-alarm rate instead. |
| Mendeley zero-shot AUROC 0.501 | OK, until E4 runs | `results/mendeley_accelerometer_results.json`. Also give the false-alarm rates: 99.7% (Looped) and 99.2% (Branched). The 0.501 was computed on saturated probabilities (INTEGRITY_LOG #12). Replace it with E4's logit-based number. |
| The real-data failure is caused by a scaling mismatch | PENDING (E1 supports it, E4 decides it) | E1: leak-free noise with RMS ≥ 0.05 is scored as a leak (Model C/D). E4 measures how much matched scaling recovers. Don't claim E4's outcome before it runs. |
| "Closing the gap only needs PVC-specific synthesis tuning; not a fundamental barrier" | REMOVE | Experiment 2 *was* PVC/Mendeley-matched synthesis and scored 0.515. |
| Mendeley differs from municipal mains (material, 47 m / 50 mm PVC, 2–5 m head, 0.18–0.47 L/s) | OK | A real and relevant domain gap. Present it as *a* cause to test, not *the* proven cause. |
| PosMAE 0.0185 on L-TOWN | OK, with caveat | Normalised by sensor span, computed on dual-sensor leak samples only, on synthetic data. Compare with GCC-PHAT (E3 / `baselines/gccphat.py`). |
| Severity R² 0.76 / 0.60 / 0.75 | OK, with caveat | Synthetic only. |

## Method

| Claim | Status | Evidence / what's actually true |
|---|---|---|
| Cross-Channel Attention is "computationally equivalent to cross-correlation / TDOA" | REMOVE | The gate is computed from time-averaged features and is constant over time (`model_C/model.py`; `tests/test_model.py::test_gate_is_constant_over_time`). It cannot see timing. |
| CCA is the key novelty | PENDING (ablation) | Run `--fusion concat` with 3 seeds. If there's no difference, say so. Either way it's a finding. |
| "No prior work uses this" / "first in WDN" / "without precedent" | REWORD | Say "we did not find prior work using…", and cite the closest work you did find (e.g. FiT-WST+, 2025). You didn't do a systematic search. |
| "No free tuning parameters" | REMOVE | Hand-set values: material centre frequencies and bandwidths, 0.6 channel correlation, SNR range 0.5–12 dB, the ×5000 Torricelli scaling, the 40/25/20/15 leak-type mix. |
| Uses a physics-based synthesiser (Torricelli amplitude, wave speed, attenuation) | OK | `model_C/dataset_c.py`. |
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

1. Why is AUROC 1.000 not suspicious? (Answer from E2/E3: what does an RMS detector score on the same data?)
2. What does your model see that GCC-PHAT doesn't? (E3 localisation curves)
3. Why did the model fail on real data, and how do you know? (E1 + E4)
4. Did the Mendeley noise used in training leak into your test? (Yes for Branched no-leak, so the clean test is Looped only.)
5. How much real data do you need? (E5)
6. What exactly did each of you do? (Write your own contribution statement.)
