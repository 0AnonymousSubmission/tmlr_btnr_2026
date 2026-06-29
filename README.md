# BMPO - Bayesian Tensor Network Experiments

A framework for running Bayesian Tensor Network for Regression (BTNR) experiments and baselines using Hydra configuration management.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

```bash
# Run with default settings (MPO2 model, BTN method, concrete dataset)
python run.py
```
    
```bash
# Run a specific configuration
python run.py dataset=abalone model=btt method=btn seed=42
```

## Usage

The `run.py` script is the main entry point for all experiments. It uses [Hydra](https://hydra.cc/) for configuration management.

### Basic Command Structure

```bash
python run.py [OPTIONS]
```

### Configuration Options

#### Methods

| Method | Description |
|--------|-------------|
| `btn` | Bayesian Tensor Network (default) |
| `als` | Alternating Least Squares |
| `baseline` | Baseline methods (GP, BNN, etc.) |

```bash
python run.py method=btn      # Bayesian Tensor Network
python run.py method=als      # Alternating Least Squares
python run.py method=baseline # Baseline methods
```

#### Models

**Tensor Network Models** (for `btn` and `als` methods):

| Model | Description |
|-------|-------------|
| `mpo2` | Matrix Product Operator (default) |
| `lmpo2` | Local MPO |
| `btt` | Binary Tensor Train |
| `cpd` | CP Decomposition |

**Baseline Models** (for `baseline` method):

| Model | Description |
|-------|-------------|
| `exact_gp` | Exact Gaussian Process |
| `sparse_gp` | Sparse Gaussian Process |
| `horseshoe_bnn` | Horseshoe Bayesian Neural Network |
| `bde_mile` | BDE MILE |
| `mvbayes` | Multivariate Bayes |
| `bayesian_widedeep` | Bayesian Wide & Deep |
| `bayesian_tabnet` | Bayesian TabNet |
| `deep_ensemble` | Deep Ensemble |
| `mc_dropout` | MC Dropout |

```bash
python run.py model=mpo2              # MPO model
python run.py model=btt               # Binary Tensor Train
python run.py method=baseline model=sparse_gp  # Sparse GP baseline
```

#### Datasets

| Dataset | Description |
|---------|-------------|
| `concrete` | Concrete strength (default) |
| `abalone` | Abalone age prediction |
| `ai4i` | AI4I predictive maintenance |
| `appliances` | Appliances energy |
| `bike` | Bike sharing demand |
| `energy_efficiency` | Building energy efficiency |
| `obesity` | Obesity levels |
| `realstate` | Real estate valuation |
| `seoulBike` | Seoul bike sharing |
| `student_perf` | Student performance |

```bash
python run.py dataset=concrete
python run.py dataset=abalone
python run.py dataset=bike
```

### Model Parameters

```bash
# Tensor network parameters
python run.py model.L=3               # Number of layers (default: 3)
python run.py model.bond_dim=18       # Bond dimension (default: 18)
python run.py model.init_strength=0.1 # Initialization strength

# BTN-specific parameters
python run.py method.bond_prior_alpha=5.0  # Prior strength
python run.py method.trimming_threshold=0.1  # Trimming threshold
```

### Training Parameters

```bash
python run.py training.n_epochs=100   # Number of epochs
python run.py training.batch_size=512 # Batch size
python run.py training.patience=50    # Early stopping patience
```

### Other Options

```bash
python run.py seed=42                 # Random seed
python run.py skip_completed=true     # Skip already completed runs (default)
python run.py device=auto             # Device selection (auto/cpu/cuda)
```

## Hydra Multirun (Grid Search)

Run experiments across multiple configurations:

```bash
# Multiple seeds
python run.py --multirun seed=42,7,123,256,999

# Multiple models and seeds
python run.py --multirun model=mpo2,btt,cpd seed=42,123

# Multiple datasets
python run.py --multirun dataset=concrete,abalone,bike

# Full ablation study
python run.py --multirun \
    model=mpo2,btt,cpd,lmpo2 \
    dataset=concrete,abalone \
    model.L=3,4 \
    seed=42,7,123,256,999
```

## Pre-configured Training Profiles

Use pre-defined training configurations:

```bash
# BTN ablation with high prior (5.0) and init (0.1)
python run.py training=btn_ablation_high

# BTN ablation with low prior (1.0) and init (0.01)
python run.py training=btn_ablation_low

# ALS ablation
python run.py method=als training=als_ablation_high
```

## Output Structure

Results are saved to:
- **Hydra logs**: `runs/<method>/<dataset>/<model>/...`
- **Results JSON**: `outputs/<method>/<dataset>/<model>/...`

## Examples

### Run BTN with MPO2 on concrete dataset

```bash
python run.py method=btn model=mpo2 dataset=concrete seed=42
```

### Run baseline Sparse GP experiment

```bash
python run.py method=baseline model=sparse_gp dataset=abalone seed=42
```

### Full BTN ablation on all datasets

```bash
python run.py --multirun \
    method=btn \
    model=mpo2,btt,cpd,lmpo2 \
    dataset=concrete,abalone,ai4i,appliances,bike,energy_efficiency,obesity,realstate,seoulBike,student_perf \
    model.L=3,4 \
    seed=42,7,123,256,999 \
    training=btn_ablation_high
```

## Project Structure

```
BMPO/
├── run.py                 # Main entry point
├── conf/                  # Hydra configuration
│   ├── config.yaml        # Default configuration
│   ├── method/            # Method configs (btn, als, baseline)
│   ├── model/             # Model configs
│   ├── dataset/           # Dataset configs
│   ├── training/          # Training profiles
│   └── tracker/           # Tracking backends
├── experiments/           # Experiment runners
├── model/                 # Model implementations
├── baselines/             # Baseline implementations
└── utils/                 # Utilities
```
