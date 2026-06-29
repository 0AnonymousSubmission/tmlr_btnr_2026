#!/usr/bin/env python3
"""
Extract best model configurations from ablation study results.

Scans runs/{BTN,ALS} directories to find the best configuration for each
(dataset, model, L, prior+init) combination for BTN, and additionally bond_dim for ALS.

Selection criterion: Lowest val_loss or highest val_quality (configurable), averaged across seeds.

Outputs YAML configs to conf/test_config/{btn,als}/<model>.yaml

Usage:
    python extract_best_ablation_configs.py [--metric val_loss|val_quality]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any
import statistics
import yaml
import numpy as np


def load_run_file(filepath: Path) -> dict[str, Any] | None:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def get_best_metric(data: dict[str, Any], metric: str) -> float | None:
    metrics_log = data.get("metrics_log", [])
    if not metrics_log:
        return None
    
    values = [m.get(metric) for m in metrics_log if m.get(metric) is not None]
    if not values:
        return None
    
    return max(values) if metric == "val_quality" else min(values)


def extract_run_info(data: dict[str, Any], filepath: Path, metric: str) -> dict[str, Any] | None:
    try:
        best_metric_value = get_best_metric(data, metric)
        if best_metric_value is None:
            return None
        
        config = data.get("config", {})
        hparams = data.get("hparams", {})
        model_config = config.get("model", {})
        method_config = config.get("method", {})
        dataset_config = config.get("dataset", {})
        
        return {
            "method": method_config.get("name") or hparams.get("method", "UNKNOWN"),
            "dataset": dataset_config.get("name") or hparams.get("dataset", "UNKNOWN"),
            "model": model_config.get("name") or hparams.get("model", "UNKNOWN"),
            "L": model_config.get("L") or hparams.get("L"),
            "bond_dim": model_config.get("bond_dim") or hparams.get("bond_dim"),
            "init_strength": model_config.get("init_strength"),
            "bond_prior_alpha": method_config.get("bond_prior_alpha"),
            "added_bias": dataset_config.get("added_bias"),
            "seed": config.get("seed") or hparams.get("seed"),
            "best_metric": best_metric_value,
            "filepath": str(filepath),
        }
    except Exception as e:
        print(f"Warning: Error extracting info from {filepath}: {e}")
        return None


def find_all_runs(base_path: Path, method: str, metric: str) -> list[dict[str, Any]]:
    runs_dir = base_path / "runs" / method
    if not runs_dir.exists():
        print(f"Warning: {runs_dir} does not exist")
        return []
    
    all_runs = []
    for json_file in runs_dir.rglob("*.json"):
        data = load_run_file(json_file)
        if data:
            info = extract_run_info(data, json_file, metric)
            if info:
                all_runs.append(info)
    
    return all_runs


def make_btn_config_key(run: dict) -> tuple:
    return (
        run["dataset"],
        run["model"],
        run["L"],
        run["bond_prior_alpha"],
        run["init_strength"],
    )


def make_als_config_key(run: dict) -> tuple:
    return (
        run["dataset"],
        run["model"],
        run["L"],
        run["bond_dim"],
        run["bond_prior_alpha"],
        run["init_strength"],
    )


# Baseline model hyperparameter keys (which params to extract for config key)
BASELINE_HPARAM_KEYS = {
    "HorseshoeBNN": ["n_hidden_units", "n_samples"],
    "BayesianWideDeep": ["mlp_hidden_dims", "n_samples"],
    "BdeMile": ["hidden_layers", "n_members"],
    "MvBayes": ["maxInt", "maxBasis"],
    "SparseGP": ["n_inducing", "kernel_type"],
    "ExactGP": ["kernel_type", "use_ard"],
}

# Map model class names to hydra config names
MODEL_TO_HYDRA_CONFIG = {
    "HorseshoeBNN": "horseshoe_bnn",
    "BayesianWideDeep": "bayesian_widedeep",
    "BdeMile": "bde_mile",
    "MvBayes": "mvbayes",
    "SparseGP": "sparse_gp",
    "ExactGP": "exact_gp",
    # TN models (already lowercase)
    "MPO2": "mpo2",
    "LMPO2": "lmpo2",
    "BTT": "btt",
    "CPD": "cpd",
}


def extract_baseline_run_info(data: dict[str, Any], filepath: Path, metric: str) -> dict[str, Any] | None:
    """Extract run info from baseline experiment JSON files."""
    try:
        # For baseline, metric is directly in summary, not metrics_log
        summary = data.get("summary", {})
        config = data.get("config", {})
        hparams = data.get("hparams", {})
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        
        # Get metric value - baseline uses val_r2, test_r2, etc.
        if metric == "val_quality":
            metric_value = summary.get("test_r2")  # or use val_r2 if available
        else:
            metric_value = summary.get("test_mse")
        
        if metric_value is None:
            return None
        
        model_name = model_config.get("name") or hparams.get("model", "UNKNOWN")
        
        # Extract model-specific hyperparameters
        hparam_keys = BASELINE_HPARAM_KEYS.get(model_name, [])
        model_hparams = {k: model_config.get(k) for k in hparam_keys if model_config.get(k) is not None}
        
        return {
            "method": "baseline",
            "dataset": dataset_config.get("name") or hparams.get("dataset", "UNKNOWN"),
            "model": model_name,
            "model_hparams": model_hparams,
            "seed": config.get("seed") or hparams.get("seed"),
            "best_metric": metric_value,
            "filepath": str(filepath),
        }
    except Exception as e:
        print(f"Warning: Error extracting baseline info from {filepath}: {e}")
        return None


def find_all_baseline_runs(base_path: Path, metric: str) -> list[dict[str, Any]]:
    """Find all baseline runs from runs/baseline directory."""
    runs_dir = base_path / "runs" / "baseline"
    if not runs_dir.exists():
        print(f"Warning: {runs_dir} does not exist")
        return []
    
    all_runs = []
    for json_file in runs_dir.rglob("*.json"):
        data = load_run_file(json_file)
        if data:
            info = extract_baseline_run_info(data, json_file, metric)
            if info:
                all_runs.append(info)
    
    return all_runs


def make_baseline_config_key(run: dict) -> tuple:
    """Create a unique key for baseline config (dataset, model, hparams)."""
    # Convert lists to tuples for hashability
    def make_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        return v
    
    hparams_tuple = tuple(sorted((k, make_hashable(v)) for k, v in run["model_hparams"].items()))
    return (
        run["dataset"],
        run["model"],
        hparams_tuple,
    )


def find_best_baseline_configs(runs: list[dict], metric: str) -> dict[tuple, dict]:
    """Find best config per (dataset, model) for baselines."""
    higher_is_better = metric == "val_quality"
    
    config_groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        config_key = make_baseline_config_key(run)
        config_groups[config_key].append(run)
    
    config_stats = {}
    for config_key, group_runs in config_groups.items():
        metric_values = [r["best_metric"] for r in group_runs]
        mean_metric = statistics.mean(metric_values)
        std_metric = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
        
        template = group_runs[0]
        config_stats[config_key] = {
            "dataset": template["dataset"],
            "model": template["model"],
            "model_hparams": template["model_hparams"],
            "mean_metric": mean_metric,
            "std_metric": std_metric,
            "n_seeds": len(group_runs),
        }
    
    # Find best per (dataset, model)
    best_configs: dict[tuple[str, str], dict] = {}
    for config_key, stats in config_stats.items():
        dm_key = (stats["dataset"], stats["model"])
        if dm_key not in best_configs:
            best_configs[dm_key] = stats
        elif higher_is_better and stats["mean_metric"] > best_configs[dm_key]["mean_metric"]:
            best_configs[dm_key] = stats
        elif not higher_is_better and stats["mean_metric"] < best_configs[dm_key]["mean_metric"]:
            best_configs[dm_key] = stats
    
    return best_configs


def save_baseline_yaml_configs(best_configs: dict, base_path: Path, metric: str):
    """Save baseline best configs to YAML files."""
    output_dir = base_path / "conf" / "test_config" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_configs: dict[str, dict] = defaultdict(dict)
    for (dataset, model), config in best_configs.items():
        model_configs[model][dataset] = {
            **config["model_hparams"],
            "_meta": {
                f"mean_{metric}": round(config["mean_metric"], 6),
                f"std_{metric}": round(config["std_metric"], 6),
                "n_seeds": config["n_seeds"],
            }
        }
    
    for model, datasets in model_configs.items():
        yaml_path = output_dir / f"{model.lower()}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(datasets, f, default_flow_style=False, sort_keys=True)
        print(f"Saved: {yaml_path}")


def generate_baseline_hydra_test_configs(best_configs: dict, base_path: Path, test_seeds: list[int], all_jobs: list[str]):
    """Generate hydra test configs for baseline models."""
    test_dir = base_path / "conf" / "training" / "test"
    baseline_dir = test_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    for (dataset, model), config in best_configs.items():
        # Get hydra config name for model
        hydra_model = MODEL_TO_HYDRA_CONFIG.get(model, model.lower())
        config_name = f"{dataset}_{hydra_model}"
        yaml_path = baseline_dir / f"{config_name}.yaml"
        
        hparams = config["model_hparams"]
        hparams_comment = ", ".join(f"{k}={v}" for k, v in hparams.items())
        hparams_str = "_".join(f"{k}{v}" for k, v in sorted(hparams.items()))
        
        content = f"""# @package _global_
# Auto-generated test config for baseline {model} on {dataset}
# Best config from ablation: {hparams_comment}

defaults:
  - /training/test/_base_uncertainty

# Baseline-specific settings
training:
  n_epochs: 200
  batch_size: 256

method:
  name: baseline
  bond_prior_alpha: 0
  warmup_epochs: 0

dataset:
  added_bias: false

"""
        # Add model-specific hyperparameters
        content += "model:\n"
        for k, v in hparams.items():
            if isinstance(v, list):
                content += f"  {k}: {v}\n"
            elif isinstance(v, str):
                content += f"  {k}: \"{v}\"\n"
            else:
                content += f"  {k}: {v}\n"
        
        # Add output dir and hydra sweep config
        content += f"""
output:
  save_models: false
  results_dir: tests_uncertainty/baseline/${{dataset.name}}/${{model.name}}/{hparams_str}

hydra:
  sweep:
    dir: tests_uncertainty_runs/baseline/${{dataset.name}}/${{model.name}}
    subdir: {hparams_str}/seed${{seed}}
"""
        
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"Saved: {yaml_path}")
        
        all_jobs.append(f"baseline:{dataset}:{hydra_model}")


def print_baseline_summary(best_configs: dict, metric: str):
    """Print summary of best baseline configurations."""
    criterion = "highest" if metric == "val_quality" else "lowest"
    print(f"\n{'=' * 100}")
    print(f"BEST CONFIGURATIONS FOR BASELINE (by {criterion} mean {metric})")
    print(f"{'=' * 100}")
    
    datasets = sorted(set(k[0] for k in best_configs.keys()))
    models = sorted(set(k[1] for k in best_configs.keys()))
    
    header = f"{'Dataset':<20} | {'Model':<18} | {'Hyperparameters':<40} | {'Mean':<12} | {'Std':<10} | {'N':<3}"
    print(header)
    print("-" * len(header))
    
    for dataset in datasets:
        for model in models:
            config = best_configs.get((dataset, model))
            if config:
                hparams_str = ", ".join(f"{k}={v}" for k, v in config["model_hparams"].items())
                if len(hparams_str) > 38:
                    hparams_str = hparams_str[:35] + "..."
                print(
                    f"{dataset:<20} | {model:<18} | {hparams_str:<40} | "
                    f"{config['mean_metric']:<12.6f} | {config['std_metric']:<10.6f} | {config['n_seeds']:<3}"
                )


def find_best_configs(runs: list[dict], make_key_fn, metric: str) -> dict[tuple, dict]:
    higher_is_better = metric == "val_quality"
    
    config_groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        config_key = make_key_fn(run)
        config_groups[config_key].append(run)
    
    config_stats = {}
    for config_key, group_runs in config_groups.items():
        metric_values = [r["best_metric"] for r in group_runs]
        mean_metric = statistics.mean(metric_values)
        std_metric = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
        
        template = group_runs[0]
        config_stats[config_key] = {
            "dataset": template["dataset"],
            "model": template["model"],
            "L": template["L"],
            "bond_dim": template["bond_dim"],
            "bond_prior_alpha": template["bond_prior_alpha"],
            "init_strength": template["init_strength"],
            "added_bias": template["added_bias"],
            "mean_metric": mean_metric,
            "std_metric": std_metric,
            "n_seeds": len(group_runs),
        }
    
    best_configs: dict[tuple[str, str], dict] = {}
    for config_key, stats in config_stats.items():
        dm_key = (stats["dataset"], stats["model"])
        if dm_key not in best_configs:
            best_configs[dm_key] = stats
        elif higher_is_better and stats["mean_metric"] > best_configs[dm_key]["mean_metric"]:
            best_configs[dm_key] = stats
        elif not higher_is_better and stats["mean_metric"] < best_configs[dm_key]["mean_metric"]:
            best_configs[dm_key] = stats
    
    return best_configs


def save_yaml_configs(best_configs: dict, method: str, base_path: Path, metric: str):
    output_dir = base_path / "conf" / "test_config" / method.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_configs: dict[str, dict] = defaultdict(dict)
    for (dataset, model), config in best_configs.items():
        model_configs[model][dataset] = {
            "L": config["L"],
            "bond_prior_alpha": config["bond_prior_alpha"],
            "init_strength": config["init_strength"],
            **({"bond_dim": config["bond_dim"]} if method == "ALS" else {}),
            "_meta": {
                f"mean_{metric}": round(config["mean_metric"], 6),
                f"std_{metric}": round(config["std_metric"], 6),
                "n_seeds": config["n_seeds"],
            }
        }
    
    for model, datasets in model_configs.items():
        yaml_path = output_dir / f"{model.lower()}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(datasets, f, default_flow_style=False, sort_keys=True)
        print(f"Saved: {yaml_path}")


def generate_hydra_test_configs(best_configs: dict, method: str, base_path: Path, test_seeds: list[int], all_jobs: list[str]):
    test_dir = base_path / "conf" / "training" / "test"
    method_dir = test_dir / method.lower()
    method_dir.mkdir(parents=True, exist_ok=True)
    
    seeds_str = ",".join(map(str, test_seeds))
    
    # Create _base_uncertainty.yaml for BTN/ALS tests with uncertainty metrics
    base_unc_yaml_path = test_dir / "_base_uncertainty.yaml"
    if not base_unc_yaml_path.exists():
        base_unc_content = f"""# @package _global_
# Shared test configuration with uncertainty metrics enabled
# NOTE: We don't inherit from gridsearch to avoid its model.L sweep.
# Test configs set the best L value directly from ablation results.

training:
  mode: gridsearch
  n_epochs: 100
  batch_size: 512
  patience: 50
  min_delta: 0.0001
  val_split: 0.15
  test_split: 0.15

uncertainty:
  enabled: true
  confidence: 0.95
  keep_curves: false
  outlier:
    enabled: true
    fraction: 0.1
    scale: 5.0

output:
  results_dir: tests_uncertainty/${{method.name}}/${{dataset.name}}/${{model.name}}_L${{model.L}}_d${{model.bond_dim}}/prior${{method.bond_prior_alpha}}_init${{model.init_strength}}

hydra:
  sweeper:
    params:
      seed: {seeds_str}
  sweep:
    dir: tests_uncertainty_runs/${{method.name}}/${{dataset.name}}/${{model.name}}_L${{model.L}}_d${{model.bond_dim}}/prior${{method.bond_prior_alpha}}_init${{model.init_strength}}
    subdir: seed${{seed}}
"""
        with open(base_unc_yaml_path, "w") as f:
            f.write(base_unc_content)
        print(f"Saved: {base_unc_yaml_path}")
    
    for (dataset, model), config in best_configs.items():
        config_name = f"{dataset}_{model.lower()}"
        yaml_path = method_dir / f"{config_name}.yaml"
        
        L = config["L"]
        bond_prior_alpha = config["bond_prior_alpha"]
        init_strength = config["init_strength"]
        bond_dim = config.get("bond_dim")
        
        content = f"""# @package _global_
# Auto-generated test config for {method} {model} on {dataset}
# Best config from ablation: L={L}, prior={bond_prior_alpha}, init={init_strength}"""
        
        if bond_dim:
            content += f", bond_dim={bond_dim}"
        
        content += f"""

defaults:
  - /training/test/_base_uncertainty

method:
  bond_prior_alpha: {bond_prior_alpha}

model:
  L: {L}
  init_strength: {init_strength}"""
        
        if method.upper() == "ALS" and bond_dim:
            content += f"""
  bond_dim: {bond_dim}"""
        
        content += "\n"
        
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"Saved: {yaml_path}")
        
        all_jobs.append(f"{method.lower()}:{dataset}:{model.lower()}")


def generate_jobs_env(all_jobs: list[str], base_path: Path):
    submit_dir = base_path / "submit_test"
    submit_dir.mkdir(parents=True, exist_ok=True)
    
    jobs_env_path = submit_dir / "jobs.env"
    
    btn_jobs = sorted([j for j in all_jobs if j.startswith("btn:")])
    baseline_jobs = sorted([j for j in all_jobs if j.startswith("baseline:")])
    
    total_jobs = len(all_jobs)
    
    lines = [
        "# Auto-generated test jobs",
        "# Format: method:dataset:model",
        f"# TOTAL JOBS: {total_jobs}",
        "#",
        f"# For HPC submission use: #BSUB -J \"unc_test[1-{total_jobs}]%30\"",
        "",
        "declare -a JOBS=("
    ]
    
    if btn_jobs:
        lines.append(f"    # BTN tests ({len(btn_jobs)} jobs)")
        for job in btn_jobs:
            lines.append(f'    "{job}"')
    
    if baseline_jobs:
        lines.append(f"    # Baseline tests ({len(baseline_jobs)} jobs)")
        for job in baseline_jobs:
            lines.append(f'    "{job}"')
    
    lines.append(")")
    
    with open(jobs_env_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"Saved: {jobs_env_path} ({total_jobs} jobs)")


def print_summary(best_configs: dict, method: str, metric: str):
    criterion = "highest" if metric == "val_quality" else "lowest"
    print(f"\n{'=' * 100}")
    print(f"BEST CONFIGURATIONS FOR {method} (by {criterion} mean {metric})")
    print(f"{'=' * 100}")
    
    datasets = sorted(set(k[0] for k in best_configs.keys()))
    models = sorted(set(k[1] for k in best_configs.keys()))
    
    header = f"{'Dataset':<20} | {'Model':<8} | {'L':<3} | {'D':<4} | {'Prior':<6} | {'Init':<6} | {'Mean':<12} | {'Std':<10} | {'N':<3}"
    print(header)
    print("-" * len(header))
    
    for dataset in datasets:
        for model in models:
            config = best_configs.get((dataset, model))
            if config:
                print(
                    f"{dataset:<20} | {model:<8} | {config['L']:<3} | "
                    f"{config['bond_dim'] or 'N/A':<4} | {config['bond_prior_alpha']:<6} | "
                    f"{config['init_strength']:<6} | {config['mean_metric']:<12.6f} | "
                    f"{config['std_metric']:<10.6f} | {config['n_seeds']:<3}"
                )


def main():
    parser = argparse.ArgumentParser(description="Extract best ablation configs")
    parser.add_argument(
        "--metric",
        choices=["val_loss", "val_quality"],
        default="val_quality",
        help="Metric to optimize: val_loss (lower is better) or val_quality (higher is better)",
    )
    parser.add_argument(
        "--generate-test-configs",
        action="store_true",
        help="Generate hydra training configs for test runs",
    )
    parser.add_argument(
        "--test-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for test runs (default: 10 random seeds < 10000)",
    )
    parser.add_argument(
        "--n-test-seeds",
        type=int,
        default=10,
        help="Number of test seeds to generate if --test-seeds not provided (default: 10)",
    )
    parser.add_argument(
        "--seed-rng",
        type=int,
        default=2024,
        help="RNG seed for generating test seeds (default: 2024)",
    )
    parser.add_argument(
        "--skip-btn",
        action="store_true",
        help="Skip BTN processing",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline processing",
    )
    args = parser.parse_args()
    metric = args.metric
    
    if args.test_seeds:
        test_seeds = [int(s.strip()) for s in args.test_seeds.split(",")]
    else:
        rng = np.random.default_rng(args.seed_rng)
        test_seeds = sorted(rng.integers(0, 10000, size=args.n_test_seeds).tolist())
        print(f"Generated test seeds: {test_seeds}")
    
    base_path = Path(__file__).parent
    all_jobs = []
    
    # Process BTN (has predictive uncertainty)
    if not args.skip_btn:
        print("=" * 100)
        print(f"Processing BTN runs... (optimizing {metric})")
        print("=" * 100)
        btn_runs = find_all_runs(base_path, "BTN", metric)
        print(f"Found {len(btn_runs)} BTN runs")
        
        if btn_runs:
            btn_best = find_best_configs(btn_runs, make_btn_config_key, metric)
            print_summary(btn_best, "BTN", metric)
            save_yaml_configs(btn_best, "BTN", base_path, metric)
            if args.generate_test_configs:
                print("\nGenerating hydra test configs for BTN...")
                generate_hydra_test_configs(btn_best, "BTN", base_path, test_seeds, all_jobs)
    
    # NOTE: ALS is skipped for uncertainty tests - it has no predictive variance
    
    # Process Baseline (Bayesian baselines have predictive uncertainty)
    if not args.skip_baseline:
        print("\n" + "=" * 100)
        print(f"Processing Baseline runs... (optimizing {metric})")
        print("=" * 100)
        baseline_runs = find_all_baseline_runs(base_path, metric)
        print(f"Found {len(baseline_runs)} Baseline runs")
        
        if baseline_runs:
            baseline_best = find_best_baseline_configs(baseline_runs, metric)
            print_baseline_summary(baseline_best, metric)
            save_baseline_yaml_configs(baseline_best, base_path, metric)
            if args.generate_test_configs:
                print("\nGenerating hydra test configs for Baseline...")
                generate_baseline_hydra_test_configs(baseline_best, base_path, test_seeds, all_jobs)
    
    if args.generate_test_configs and all_jobs:
        generate_jobs_env(all_jobs, base_path)
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    if not args.skip_btn:
        print(f"BTN configs saved to: {base_path / 'conf' / 'test_config' / 'btn'}")
    if not args.skip_baseline:
        print(f"Baseline configs saved to: {base_path / 'conf' / 'test_config' / 'baseline'}")
    if args.generate_test_configs:
        print(f"\nHydra test configs saved to: {base_path / 'conf' / 'training' / 'test'}")
        print("\nTo run tests, use:")
        print("  python run.py method=btn model=mpo2 dataset=concrete training=test/btn/concrete_mpo2 -m")
        print("  python run.py method=baseline model=horseshoe_bnn dataset=concrete training=test/baseline/concrete_horseshoe_bnn -m")


if __name__ == "__main__":
    main()
