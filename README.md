# AcousticLeakNet — Deep Learning for Water Pipe Leak Detection

A 1D CNN with Cross-Channel Attention for detecting, localising, and severity-scoring leaks in water distribution networks using dual-sensor acoustic data. Built for the SJWP (Stockholm Junior Water Prize) research project.

## Problem

Non-revenue water (NRW) losses from leaking pipes cost municipalities billions annually. Traditional acoustic sensors struggle in noisy urban environments. This project uses synthetic acoustic waveforms generated from EPANET hydraulic simulations to train a neural network that detects leaks from paired sensor signals, learning time-difference-of-arrival (TDOA) representations end-to-end.

## Architecture

**AcousticLeakNet** takes a 2-channel waveform (two sensors, 5kHz sampling) and pipe metadata as input, and outputs:

| Task | Output | Loss |
|------|--------|------|
| Detection | Binary logit | BCEWithLogitsLoss |
| Localisation | Normalised position [0, 1] | HuberLoss (leak-only) |
| Severity | Flow rate (L/s) | HuberLoss (leak-only) |

Key architectural novelty: **Cross-Channel Attention** — the model learns to compare features between the two sensor channels, computationally equivalent to cross-correlation TDOA but trained end-to-end.

Uncertainty-weighted multi-task loss (Kendall et al., 2018) automatically balances the three tasks during training.

## Project Structure

```
├── scripts/              # Main training pipeline (base variant)
│   ├── build_index.py    # Scan datasets, build CSV index files
│   ├── pregenerate.py    # Pre-generate waveforms to .npy cache
│   ├── dataset.py        # LeakDataset — physics-based waveform synthesis
│   ├── model.py          # AcousticLeakNet architecture + UncertaintyLoss
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Cross-network test evaluation
│   └── index_sampling.py # Stratified sampling for train/val splits
│
├── model_C/              # Model C variant (calibrated SNR, correlated channels)
│   ├── pregen_c.py
│   ├── train_c.py
│   ├── evaluate_c.py
│   └── dataset_c.py
│
├── Model_D/              # Model D variant
│   ├── pregen_D.py
│   ├── train_D.py
│   ├── eval_D.py
│   └── dataset_d.py
│
├── scriptnewbby/         # Mendeley accelerometer data experiments
├── inp/                  # EPANET network input files (LFS)
├── csv/                  # Index and split CSV files (LFS)
├── models/               # Saved model checkpoints (LFS)
├── plots/                # Generated evaluation plots
├── results/              # JSON evaluation metrics
├── cache/                # Pre-generated signal cache (gitignored)
├── datasets/             # Raw EPANET simulation CSVs (gitignored)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/AaravGupta17/SJWP_Research.git
cd SJWP_Research
pip install -r requirements.txt
```

Git LFS is used for large files (`.inp`, `.csv`, `.pt`, `.pth`). Install LFS if you haven't:

```bash
git lfs install
git lfs pull
```

## Usage

### 1. Build the index

Scans the raw dataset CSVs and produces index files mapping each sample to its source file and row.

```bash
cd scripts
python build_index.py
python index_sampling.py   # creates train/val splits
```

### 2. Pre-generate signals

Converts EPANET simulation data into acoustic waveforms and caches them as `.npy` files. This must be done before training.

```bash
python pregenerate.py --split all
# or individually:
python pregenerate.py --split train
python pregenerate.py --split val
python pregenerate.py --split test_network_3
```

### 3. Train

```bash
python train.py
```

Checkpoints are saved to `models/` when validation AUROC improves.

### 4. Evaluate

```bash
python evaluate.py
```

Evaluates on unseen test networks (L-TOWN, KY15, Richmond) and saves plots to `plots/` and metrics to `results/`.

### Model C / Model D variants

Same pipeline, different directories:

```bash
cd model_C
python pregen_c.py --split all
python train_c.py
python evaluate_c.py

cd ../Model_D
python pregen_D.py --split all
python train_D.py
python eval_D.py
```

## Data

The dataset consists of EPANET hydraulic simulation outputs for 8 municipal water networks, across 4 pipe materials (CI, DI, PVC, STEEL), with multiple leak scenarios and demand multipliers.

| Split | Networks | Purpose |
|-------|----------|---------|
| Train | 1, 2, 4, 5, 7 | Model training |
| Test | 3, 6, 8 | Cross-network generalisation |

Each simulation row contains pipe geometry, flow conditions, pressure readings, and leak metadata. The `LeakDataset` class synthesises 2-channel acoustic waveforms on-the-fly from this metadata using physics-based signal generation (Gaussian pulse source, material-specific acoustic properties, TDOA delays, attenuation, and coloured noise).

## Signal Generation

Waveforms are synthesised from simulation parameters, not recorded from real sensors:

- **Source**: Gaussian-envelope pulse at material-dependent centre frequency
- **Propagation**: Time-delayed to each sensor based on distance and wave speed
- **Attenuation**: Exponential decay with material-specific damping
- **Noise**: Colored (flow turbulence) + white (thermal) + ground vibration
- **Augmentation**: ±15% wave speed perturbation, amplitude scaling, optional Gaussian noise

