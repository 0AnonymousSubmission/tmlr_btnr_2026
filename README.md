# BTNR - Bayesian Tensor Network for Regression

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

### Bond Trimming (BTN)

BTN prunes bond dimensions during training. The scoring strategy and what may be
trimmed are configurable:

```bash
# Trimming strategy (score used to decide which dimensions to keep)
python run.py method.trim_method=relevance   # 1/E[lambda]           (default)
python run.py method.trim_method=gamma       # effective parameters  (1 - alpha*Sigma)

# When/how aggressively to trim
python run.py method.trimming_threshold=0.1  # keep dims with score >= threshold
python run.py method.trim_every=6            # trim every N epochs after warmup
```

| Flag | Default | Description |
|------|---------|-------------|
| `method.trim_method` | `relevance` | Scoring strategy: `relevance` or `gamma`. |
| `method.trim_nt_nodes` | `false` | If `true`, bonds touching a non-trainable (`NT`) block may be trimmed (the NT node is co-sliced); otherwise such bonds are protected. |
| `method.trim_input` | `false` | If `true`, input (feature) legs may be trimmed; feature columns are dropped from all data streams. An input leg trimmed to 0 is fully detached (feature removed). |
| `method.allow_empty_input` | `false` | If `true`, even the last input leg may be removed (constant model); otherwise at least one input is always kept. |
| `method.remove_trivial_bonds` | `true` | If `true`, an internal bond trimmed to dimension 1 is squeezed out losslessly (predictions unchanged). |

#### Controlling input (feature) trimming

By default only internal bonds are trimmed; input legs are protected. To let BTN
prune features (Bayesian feature selection):

```bash
# Enable input-leg trimming (feature columns are dropped from all data streams)
python run.py method.trim_input=true

# Trim harder so more inputs are removed
python run.py method.trim_input=true method.trimming_threshold=1.0

# Allow even the LAST input to be removed (degenerate constant model)
python run.py method.trim_input=true method.allow_empty_input=true
```

- With `trim_input=false` (default): input legs are never touched.
- With `trim_input=true`: an input leg may shrink, and if its kept-set becomes
  empty it is **fully detached** (the feature is removed and its edge label is
  dropped from the node). At least one input is kept unless
  `allow_empty_input=true`.

#### Controlling NT-block and trivial-bond trimming

```bash
# Trim through a fixed NT block too (the NT node is co-sliced)
python run.py method.trim_nt_nodes=true

# Keep size-1 internal bonds instead of squeezing them out
python run.py method.remove_trivial_bonds=false
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

## Adding a Model

A tensor-network model is a plain Python class whose `__init__` builds a quimb
`TensorNetwork` and exposes the attributes consumed by the runner and BTN
(use `model/MPO2_models.py`'s `MPO2` as the canonical template):

| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.tn` | `quimb.tensor.TensorNetwork` | the trainable network with open input (+ optional output) legs |
| `self.input_labels` | `list` | how inputs are named/built for the data loader (e.g. `["x0", ..., "x{L-1}"]`) |
| `self.input_dims` | `list[str]` | the open input index names contracted with the data |
| `self.output_dims` | `list[str]` | output index name(s); `["out"]` if `output_dim > 1`, else `[]` (regression uses `output_dim=1` → `[]`) |
| `self.bond_prior_alpha` *(optional)* | `float` | default bond-prior strength for this model |

Conventions:
- `create_model` calls the class with `L`, `bond_dim`, `phys_dim`, `output_dim=1`,
  `init_strength`, `use_tn_normalization` (see `core/models.py`); read any extra
  hyperparameters from `cfg.model.*` there.
- Tag each tensor uniquely (e.g. `Node{i}`) so model tensors don't clash with the
  input tensors created by the builder.
- Mark any **non-trainable** tensors with the tag `"NT"` (see `MMPO2`'s mask) —
  BTN keeps them fixed and protects their bonds during trimming.

Steps:

1. **Implement the class** under `model/` following the contract above.
2. **Register it** in `model/__init__.py`: import it and add it to the `MODELS`
   dict.
3. **Add a Hydra config** `conf/model/mymodel.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - _base

   model:
     name: MyModel    # MUST match the registry key in model/__init__.py
   ```

4. **Verify it works** with the sanity-check suite before running experiments:

   ```bash
   python run.py --sanity-check model=mymodel
   ```

   This builds the model via the normal path on small synthetic data and checks
   it end-to-end with the BTN machinery — training, all trimming methods, NT
   blocks, input detachment, trivial-bond removal, ELBO monotonicity, and
   `gamma >= 0`. It prints a pass/fail table and exits non-zero if any check
   fails (checks live in `code_sanity_checks/`). A new model should pass every
   check before being used in experiments.

   ```bash
   # accepts the usual overrides
   python run.py --sanity-check model=btt model.L=4 model.bond_dim=6
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
src/
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
├── code_sanity_checks/    # --sanity-check suite for new models
└── utils/                 # Utilities
```
