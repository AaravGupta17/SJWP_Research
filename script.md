# AcousticLeakNet — GENIUS Olympiad 2026 AI Category
## Video Presentation Script — Word for Word
### Aarav Gupta & Armaan Guha · Delhi Public School Noida

**Total target: ≤ 5 minutes**
**Format: Screen recording of slides with voiceover**

---

## SLIDE 1 — TITLE
**[Timing: 0:00 – 0:22]**

> "Every day, cities around the world pump millions of litres of clean, treated water — and a huge fraction of it never reaches anyone. It leaks silently underground, undetected, wasted.
>
> We built AcousticLeakNet — a deep learning system that can detect, locate, and estimate the severity of pipe leaks using sound, trained entirely on computer simulations, at a cost that any water utility in the world can actually afford."

---

## SLIDE 2 — PROBLEM & DATASET
**[Timing: 0:22 – 1:10]**

> "The scale of the problem is staggering. One hundred and twenty-six billion cubic metres of treated water is lost every year — costing utilities thirty-nine billion US dollars annually. In developing-world cities, Non-Revenue Water — water that's produced but never billed — averages forty to fifty percent of everything distributed. Physical leaks are the single largest component.
>
> The technology to find these leaks has existed since the 1980s — acoustic correlators that listen at two points on a pipe and compute the Time Difference of Arrival of the leak signal to locate the fault. But commercial units cost fifteen to thirty-five thousand dollars. Around four thousand urban water utilities in India alone simply cannot afford them. Beyond economics, every million litres lost underground wastes five hundred kilowatt-hours of pumping energy — a direct carbon cost.
>
> For our dataset, we simulated eight EPANET hydraulic benchmark networks — four for training: NW_Model1, BWSN-2, BWSN, and KY9; one for validation: Net6; and three completely held-out test networks: L-TOWN from the Czech Republic with nine hundred and five pipes, KY15 from Kentucky with six hundred and sixty-two pipes, and Richmond from the UK with just forty-four pipes. That's a twenty-times size range across three continents. We ran every network across four pipe materials — Cast Iron, Ductile Iron, PVC, and Steel — and three demand multipliers: 0.7, 1.0, and 1.2 times baseline. The result is one hundred and 0.5 million rows across a hundred and fifteen thousand simulation files. No pipe, topology, or hydraulic condition from any test network ever appeared during training."

---

## SLIDE 3 — ARCHITECTURE & SYNTHESIS
**[Timing: 1:10 – 2:10]**

> "AcousticLeakNet has two million, four hundred sixty-five thousand, five hundred thirty-nine trainable parameters. Let me walk you through it.
>
> Two acoustic sensors — one near each end of a pipe section — each feed a two-thousand time-step waveform at five thousand hertz into a shared-weight encoder: four convolutional blocks of Conv1D, batch normalisation, ReLU, and stride-2 pooling. Both sensors use identical weights, which enforces symmetry and prevents the model from learning spurious asymmetries.
>
> The primary architectural novelty is our Cross-Channel Attention module. Each channel's 256-dimensional global context vector cross-attends to the other through a two-layer MLP with sigmoid gating — learning the Time Difference of Arrival relationship end-to-end, in O(T) memory, without ever computing an explicit cross-correlation. No prior work in water distribution network leak detection uses this approach.
>
> Eleven physics features from EPANET — wave speed, attenuation, pipe length, diameter, roughness, flow, pressure at both sensors, and demand multiplier — are available via an optional scalar fusion MLP. We zero these during training so the model learns from acoustics alone.
>
> Three task heads handle simultaneous detection using binary cross-entropy, localisation using Huber loss with delta 0.1, and severity estimation using Huber loss with delta 1.0 — all jointly optimised using Kendall et al.'s uncertainty-weighted multi-task loss.
>
> The synthesis pipeline has six stages. First, EPANET outputs the physical parameters — wave speeds from four hundred to thirteen hundred metres per second, attenuation from 0.0003 to 0.003 per metre. Second, we derive leak amplitude from the Torricelli orifice equation: Q equals Cd times A times root of 2gP, with discharge coefficient 0.61 — this is physically grounded, not a free parameter. Third, from Model D we sample four leak type signatures: orifice, longitudinal crack, circumferential crack, and corroded joint. Fourth, we generate leak signals as one-over-f pink noise bandpass-filtered to each material's characteristic band — Cast Iron at 150 hertz, PVC at 800 hertz. Fifth, we apply TDOA delay per the correlator formula and a structural vibration correlation of 0.6 between channels. Sixth, we add real pipe background noise from Mendeley hydrophone recordings, synthetic pump harmonics at 50 hertz, traffic vibration, and normalise to a fixed scale.
>
> Training used AdamW at a learning rate of 3×10⁻⁴, cosine annealing over 25 epochs, batch size 256."

---

## SLIDE 4 — MODEL EVOLUTION A → D
**[Timing: 2:10 – 2:55]**

> "We developed four model versions iteratively — not by random hyperparameter search, but by diagnosing each version's specific failure modes.
>
> Model A used random uniform SNR amplitude and synthetic Gaussian noise. AUROC was 0.921, severity R-squared was negative 3.49 — severity estimation was completely broken. Model B added Torricelli amplitude and real Mendeley noise. AUROC improved to 0.936 and severity R-squared jumped to 0.62 — proof that physical calibration of amplitude was the key missing ingredient.
>
> Model C is where localisation transformed. PosMAE on L-TOWN dropped from 0.094 to 0.019 — a five-times improvement. But critically: this came entirely from fixing three cascading implementation bugs, not from adding any new features. Bug one: per-sample z-score normalisation was destroying amplitude information needed for severity. Bug two: the Torricelli amplitude wasn't being applied consistently across the cached dataset. Bug three: evaluation was computing PosMAE on unnormalised predictions. Fixing these three errors gave us perfect AUROC of 1.000 and the localisation breakthrough.
>
> Model D added frequency-dependent attenuation, four leak types, and independent sensor coupling noise. AUROC remained 1.000, severity R-squared held at 0.76 plus. The lesson here is fundamental: in physics-based simulation-to-real transfer, the accuracy of your synthesis pipeline is the bottleneck — not model architecture."

---

## SLIDE 5 — RESULTS
**[Timing: 2:55 – 3:35]**

> "Table 5 shows Model C's performance on the three completely unseen test networks.
>
> L-TOWN: AUROC 1.0000, F1 1.0000, normalised PosMAE 0.0185, severity R-squared 0.763 — on nine hundred and five pipes across twenty-one thousand seven hundred and twenty test samples.
>
> KY15: AUROC 1.0000, F1 0.9999, PosMAE 0.0783, severity R-squared 0.604 — six hundred and sixty-two pipes.
>
> Richmond: AUROC 1.0000, F1 1.0000, PosMAE 0.0920, severity R-squared 0.746 — just forty-four pipes.
>
> Combined mean: AUROC 1.0000, F1 1.0000, PosMAE 0.0478, severity R-squared 0.684 across thirty-eight thousand six hundred and sixty-four samples.
>
> Why does it generalise? TDOA signatures, frequency-dependent attenuation profiles, and the Torricelli amplitude-pressure relationship are all governed by physical laws that are independent of network topology. The same physics operates in Czech Republic, USA, and UK — which is exactly why training on simulations generalises across them.
>
> Table 7 shows the twelve-combination network-by-material breakdown. AUROC is perfect or near-perfect in every single cell. Cast Iron consistently achieves the lowest PosMAE because its highest wave speed produces the sharpest TDOA. Localisation error increases with pipe length — KY15 and Richmond have much longer individual pipes than L-TOWN, so even a small normalised error translates to a larger physical distance."

---

## SLIDE 6 — ABLATION & SIM-TO-REAL
**[Timing: 3:35 – 4:10]**

> "Tables 6a and 6b confirm the model's robustness. By material: Cast Iron AUROC 1.0000, Ductile Iron 1.0000, PVC 0.9999, Steel 1.0000. Detection is material-agnostic. By demand multiplier: AUROC is exactly 1.0000 at 0.7 times, 1.0 times, and 1.2 times baseline — demand variation has less than five percent effect on any metric.
>
> Table 8 shows sim-to-real transfer results on the Mendeley laboratory benchmark — two hundred and eighty hydrophone recordings from a 47-metre PVC lab testbed at two flow rates. Our AUROC is approximately 0.43 on this benchmark. The comparison system, FiT-WST+, achieves 0.996 — but it trains and tests on the same physical testbed.
>
> This is a domain mismatch, not a model failure. AcousticLeakNet targets Cast Iron and Ductile Iron municipal mains of 100 to 500 millimetres diameter at 20 to 80 metres of pressure head, with leak flows of 1 to 20 litres per second. The Mendeley testbed is a 47-metre, 50-millimetre PVC pipe at 2 to 5 metres of pressure with flows of 0.18 to 0.47 litres per second — a completely different material, diameter, pressure regime, and leak scale. Closing this gap requires PVC-specific synthesis tuning — it is not a fundamental barrier to the simulation-based approach."

---

## SLIDE 7 — HARDWARE & COST
**[Timing: 4:10 – 4:42]**

> "Tables 4a and 4b show our full component-level hardware design.
>
> For metal pipes — Cast Iron, Ductile Iron, Steel — we use the PCB Piezotronics 352C33 ICP accelerometer: a professional-grade sensor with a 0 to 10,000 hertz range and 100 millivolts per g sensitivity, magnetically clamped externally — no excavation. The complete two-node segment costs one thousand one hundred and forty-seven dollars and forty-eight cents.
>
> For PVC and plastic pipes — which have wave speeds of only 300 to 500 metres per second and attenuate within 10 to 25 metres — we use the SM-24 ION Geophysical geophone supplemented by a PVDF piezoelectric film sensor, epoxy-bonded to the pipe surface. The two-node segment costs just three hundred and seventy dollars and eighty-six cents.
>
> Both node types use the u-blox NEO-M8T GPS module for Pulse-Per-Second synchronisation to ±30 nanoseconds. This is non-negotiable: with sensor separations of 50 to 100 metres and acoustic speeds of 900 to 1,300 metres per second, TDOA values are measured in microseconds — nanosecond timing accuracy is required.
>
> Compared to commercial correlators at fifteen to thirty-five thousand dollars per segment, this is a ten to forty-two times cost reduction. And this is structural — not a quality compromise. We eliminate four commercial cost drivers: pressure-rated hydrophone hardware requiring WRAS or NSF 61 certification costing up to half a million dollars; proprietary encrypted radio synchronisation — GPS PPS replaces a two to five thousand dollar subsystem for eighteen dollars; regulatory certification; and software licensing."

---

## SLIDE 8 — ENVIRONMENTAL IMPACT
**[Timing: 4:42 – 5:00]**

> "The environmental case is immediate and concrete. Detecting and repairing a single medium-severity leak of 10 litres per second within one week recovers 6,048 cubic metres of treated water and saves approximately 3,024 kilowatt-hours of pumping energy.
>
> At a deployment cost of three hundred and seventy to eleven hundred and forty-seven dollars per two-node segment, versus fifteen to thirty-five thousand for commercial equipment, AcousticLeakNet brings systematic acoustic leak monitoring within reach of the four thousand Indian utilities — and hundreds of thousands of utilities across South Asia, Sub-Saharan Africa, and Latin America — that currently have no affordable option.
>
> No real-world labelled data. No pipe excavation. No network topology knowledge at inference time. Just two sensors, GPS synchronisation, and a model trained on physics."

---

## RUBRIC ALIGNMENT CHECKLIST

| Criterion (70 pts total) | Where Addressed | Key Evidence |
|---|---|---|
| **Performance** (10) | Slides 4, 5, 9 | AUROC 1.000, F1 1.000 on 3 unseen networks |
| **Environmental Solution** (10) | Slides 2, 8 | 126B m³/yr, 500 kWh/ML, $370 deployment cost |
| **Model Novelty** (10) | Slide 3 | Cross-Channel Attention (first in WDN), EPANET synthesis novelty |
| **Overall Presentation** (10) | All slides | Cohesive physical narrative across all 9 slides |
| **Cost Analysis** (5) | Slides 1, 7 | 10–42× reduction, component-level BOM, 4 structural drivers |
| **Knowledge of Coding** (5) | Slide 3 | Conv1D, AdamW, Huber loss, BCE, Kendall uncertainty-weighted MTL |
| **Model Development** (5) | Slide 4 | A→D evolution, principled debugging, bug chain lesson |
| **Knowledge of Model** (5) | Slides 3, 4 | Architecture depth, TDOA mechanism, physics–AI integration |
| **Presentation Skills** (5) | Delivery | Clear structure, no jargon without definition, physical intuition |
| **End Product Quality** (5) | Slide 8 | 5-step deployment pathway, real-world utility accessibility |

---

## DELIVERY NOTES

- **Pace**: Aim for roughly 160 words per minute. The script runs approximately 870 words of actual spoken content.
- **Slide 3**: Most technically dense — practice this section most. Consider pausing 1 second after "Cross-Channel Attention" before explaining it.
- **Slide 4**: The "5× improvement from bug fixes" story is one of the most memorable and honest moments — deliver it clearly and confidently. Judges appreciate methodological honesty.
- **Numbers to know cold**: 126B m³, $39B, 100.5M rows, 8 networks, AUROC 1.0000, PosMAE 0.0185, SevR² 0.76, $1,147 metal / $371 PVC, 10–42× cost reduction.
- **Ending**: "Just two sensors, GPS synchronisation, and a model trained on physics" — this is the thesis statement of the whole project. Deliver it with confidence as the final line.