#!/usr/bin/env python3
"""
Find the best validation quality configuration for each (model, dataset, method) combination.
Scans directories for experiment JSON files.

Supports two modes:
1. runs* directories: Individual run JSON files
2. results* directories: summary.json files with results arrays

Usage:
    python find_best_configs.py [--dir-pattern PATTERN] [--results-pattern PATTERN]

Examples:
    python find_best_configs.py  # Default: scans runs* directories
    python find_best_configs.py --dir-pattern "runs*"
    python find_best_configs.py --results-pattern "missingbtt_results*"
    python find_best_configs.py --dir-pattern "runs*" --results-pattern "missingbtt_results*"

Outputs:
1. Best singular run for each (dataset, model, method) combination
2. Best configuration averaged over seeds for each (dataset, model, method) combination
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Any
import statistics


def load_run_file(filepath: Path) -> dict[str, Any] | None:
    """Load and parse a run JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def parse_dir_name(dir_name: str) -> dict[str, Any]:
    """Extract init_strength, bond_prior_alpha, and added_bias from directory name.

    Examples:
        runs_init0.1_prior5.0_nobias -> init=0.1, prior=5.0, bias=False
        runs_init0.01_prior1.0_bias -> init=0.01, prior=1.0, bias=True
        runs_init0.1_prior5.0_als_nobias -> init=0.1, prior=5.0, bias=False (ALS)
    """
    result = {}

    # Extract init_strength
    init_match = re.search(r"init([\d.]+)", dir_name)
    if init_match:
        result["init_strength"] = float(init_match.group(1))

    # Extract bond_prior_alpha
    prior_match = re.search(r"prior([\d.]+)", dir_name)
    if prior_match:
        result["bond_prior_alpha"] = float(prior_match.group(1))

    # Extract added_bias
    if "nobias" in dir_name.lower():
        result["added_bias"] = False
    elif "bias" in dir_name.lower():
        # Check if it's explicitly 'bias' (not 'nobias')
        result["added_bias"] = True

    return result


def extract_run_info(
    data: dict[str, Any], filepath: Path, dir_params: dict[str, Any]
) -> dict[str, Any] | None:
    """Extract relevant info from a run file."""
    try:
        # Extract method (BTN or ALS)
        method = data.get("config", {}).get("method", "UNKNOWN")

        # Extract dataset
        dataset = data.get("config", {}).get("dataset") or data.get("hparams", {}).get(
            "dataset", "UNKNOWN"
        )

        # Extract model type
        model = data.get("hparams", {}).get("model") or data.get("config", {}).get(
            "params", {}
        ).get("model", "UNKNOWN")

        # Get best validation quality from summary first
        best_val_quality = data.get("summary", {}).get("best_val_quality")

        # If not in summary, compute from metrics_log
        if best_val_quality is None:
            metrics_log = data.get("metrics_log", [])
            if metrics_log:
                val_qualities = [
                    m.get("val_quality")
                    for m in metrics_log
                    if m.get("val_quality") is not None
                ]
                if val_qualities:
                    best_val_quality = max(val_qualities)

        if best_val_quality is None:
            return None

        # Extract hyperparameters from hparams (most reliable) and config.params
        hparams = data.get("hparams", {})
        config_params = data.get("config", {}).get("params", {})

        # Merge params (hparams takes precedence, then config.params, then dir_params)
        params = {**dir_params, **config_params, **hparams}

        # Extract key parameters
        L = params.get("L")
        bond_dim = params.get("bond_dim")
        added_bias = params.get("added_bias")
        init_strength = params.get("init_strength")
        bond_prior_alpha = params.get("bond_prior_alpha")
        seed = params.get("seed") or data.get("config", {}).get("seed")

        return {
            "model": model,
            "dataset": dataset,
            "method": method,
            "best_val_quality": best_val_quality,
            "filepath": str(filepath),
            "L": L,
            "bond_dim": bond_dim,
            "added_bias": added_bias,
            "init_strength": init_strength,
            "bond_prior_alpha": bond_prior_alpha,
            "seed": seed,
            "params": params,
        }
    except Exception as e:
        print(f"Warning: Error extracting info from {filepath}: {e}")
        return None


def find_all_runs(
    base_path: Path, dir_prefixes: list[str] | None = None
) -> list[tuple[Path, dict[str, Any]]]:
    """Find all JSON files in matching directories with their directory params.

    Args:
        base_path: Base directory to search in
        dir_prefixes: List of directory prefixes to match (default: ['runs'])
    """
    if dir_prefixes is None:
        dir_prefixes = ["runs"]

    run_files = []
    for item in base_path.iterdir():
        if item.is_dir() and any(
            item.name.startswith(prefix) for prefix in dir_prefixes
        ):
            # Skip test directories
            if "test" in item.name.lower():
                continue

            # Parse directory name for parameters
            dir_params = parse_dir_name(item.name)

            for json_file in item.glob("*.json"):
                run_files.append((json_file, dir_params))
    return run_files


def make_config_key(run: dict) -> tuple:
    """Create a hashable key for grouping runs by configuration (excluding seed)."""
    return (
        run["model"],
        run["dataset"],
        run["method"],
        run["L"],
        run["bond_dim"],
        run["added_bias"],
        run["init_strength"],
        run["bond_prior_alpha"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find best configurations from experiment runs"
    )
    parser.add_argument(
        "--dir-prefixes",
        type=str,
        nargs="+",
        default=["runs"],
        help="Directory prefixes to scan (default: runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for ALS specs (saves to root if specified)",
    )
    args = parser.parse_args()

    base_path = Path(__file__).parent

    print(
        f"Scanning for experiment runs in directories starting with: {args.dir_prefixes}"
    )
    run_files = find_all_runs(base_path, args.dir_prefixes)
    print(f"Found {len(run_files)} run files\n")

    all_runs = []
    for filepath, dir_params in run_files:
        data = load_run_file(filepath)
        if data:
            info = extract_run_info(data, filepath, dir_params)
            if info:
                all_runs.append(info)

    print(f"Successfully parsed {len(all_runs)} runs\n")

    # Group by (model, dataset, method) for best singular run
    best_singular: dict[tuple[str, str, str], dict] = {}
    for run in all_runs:
        key = (run["model"], run["dataset"], run["method"])
        if (
            key not in best_singular
            or run["best_val_quality"] > best_singular[key]["best_val_quality"]
        ):
            best_singular[key] = run

    # Group by configuration (excluding seed) for averaging
    config_groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in all_runs:
        config_key = make_config_key(run)
        config_groups[config_key].append(run)

    # Compute stats for each configuration
    config_stats = {}
    for config_key, runs in config_groups.items():
        val_qualities = [r["best_val_quality"] for r in runs]
        mean_val = statistics.mean(val_qualities)
        std_val = statistics.stdev(val_qualities) if len(val_qualities) > 1 else 0.0

        # Use first run as template for params
        template = runs[0]
        config_stats[config_key] = {
            "model": template["model"],
            "dataset": template["dataset"],
            "method": template["method"],
            "L": template["L"],
            "bond_dim": template["bond_dim"],
            "added_bias": template["added_bias"],
            "init_strength": template["init_strength"],
            "bond_prior_alpha": template["bond_prior_alpha"],
            "mean": mean_val,
            "std": std_val,
            "n_runs": len(runs),
        }

    # Find best averaged configuration for each (model, dataset, method)
    best_averaged: dict[tuple[str, str, str], dict] = {}
    for config_key, stats in config_stats.items():
        key = (stats["model"], stats["dataset"], stats["method"])
        if key not in best_averaged or stats["mean"] > best_averaged[key]["mean"]:
            best_averaged[key] = stats

    # Organize results
    datasets = sorted(set(r["dataset"] for r in all_runs))
    models = sorted(set(r["model"] for r in all_runs))
    methods = ["BTN", "ALS"]

    # ===== OUTPUT: Best Singular Run =====
    print("=" * 120)
    print("BEST SINGULAR RUN FOR EACH (DATASET, MODEL, METHOD)")
    print("=" * 120)

    for dataset in datasets:
        print(f"\n{'=' * 120}")
        print(f"DATASET: {dataset}")
        print(f"{'=' * 120}")
        print(
            f"{'Model':<8} | {'Method':<6} | {'Val Quality':<14} | {'L':<3} | {'D':<4} | {'Bias':<5} | {'Init':<6} | {'Prior':<6} | File"
        )
        print("-" * 120)

        for model in models:
            for method in methods:
                key = (model, dataset, method)
                run = best_singular.get(key)
                if run:
                    print(
                        f"{model:<8} | {method:<6} | {run['best_val_quality']:<14.6f} | {run['L']:<3} | {run['bond_dim']:<4} | {str(run['added_bias']):<5} | {run['init_strength']:<6} | {run['bond_prior_alpha']:<6} | {Path(run['filepath']).name}"
                    )

    # ===== OUTPUT: Best Averaged Configuration =====
    print(f"\n\n{'=' * 120}")
    print("BEST AVERAGED CONFIGURATION FOR EACH (DATASET, MODEL, METHOD)")
    print("(Averaged over seeds)")
    print("=" * 120)

    for dataset in datasets:
        print(f"\n{'=' * 120}")
        print(f"DATASET: {dataset}")
        print(f"{'=' * 120}")
        print(
            f"{'Model':<8} | {'Method':<6} | {'Mean':<12} | {'Std':<10} | {'N':<4} | {'L':<3} | {'D':<4} | {'Bias':<5} | {'Init':<6} | {'Prior':<6}"
        )
        print("-" * 120)

        for model in models:
            for method in methods:
                key = (model, dataset, method)
                stats = best_averaged.get(key)
                if stats:
                    print(
                        f"{model:<8} | {method:<6} | {stats['mean']:<12.6f} | {stats['std']:<10.6f} | {stats['n_runs']:<4} | {stats['L']:<3} | {stats['bond_dim']:<4} | {str(stats['added_bias']):<5} | {stats['init_strength']:<6} | {stats['bond_prior_alpha']:<6}"
                    )

    # ===== OUTPUT: JSON specs files =====
    # Generate BTN specs
    btn_specs = {}
    for (model, dataset, method), stats in best_averaged.items():
        if method == "BTN":
            if dataset not in btn_specs:
                btn_specs[dataset] = {}
            btn_specs[dataset][model] = {
                "L": stats["L"],
                "bond_dim": stats["bond_dim"],
                "added_bias": stats["added_bias"],
                "init_strength": stats["init_strength"],
                "bond_prior_alpha": stats["bond_prior_alpha"],
                "mean": stats["mean"],
                "std": stats["std"],
                "n_runs": stats["n_runs"],
            }

    # Generate ALS specs (includes bond_dim as a key parameter)
    als_specs = {}
    for (model, dataset, method), stats in best_averaged.items():
        if method == "ALS":
            if dataset not in als_specs:
                als_specs[dataset] = {}
            als_specs[dataset][model] = {
                "L": stats["L"],
                "bond_dim": stats["bond_dim"],  # bond_dim is important for ALS
                "added_bias": stats["added_bias"],
                "init_strength": stats["init_strength"],
                "bond_prior_alpha": stats["bond_prior_alpha"],
                "mean": stats["mean"],
                "std": stats["std"],
                "n_runs": stats["n_runs"],
            }

    # Save specs files
    btn_specs_path = base_path / "experiments" / "configs" / "kfold_best_specs_btn.json"
    als_specs_path = base_path / "experiments" / "configs" / "kfold_best_specs_als.json"

    # Use custom output path if specified
    if args.output:
        custom_output_path = Path(args.output)
        if not custom_output_path.is_absolute():
            custom_output_path = base_path / custom_output_path

        with open(custom_output_path, "w") as f:
            json.dump(als_specs, f, indent=2)
        print(f"\n\nALS specs saved to custom path: {custom_output_path}")
    else:
        with open(btn_specs_path, "w") as f:
            json.dump(btn_specs, f, indent=2)
        print(f"\n\nBTN specs saved to: {btn_specs_path}")

        with open(als_specs_path, "w") as f:
            json.dump(als_specs, f, indent=2)
        print(f"ALS specs saved to: {als_specs_path}")

    # ===== SUMMARY =====
    print(f"\n\n{'=' * 120}")
    print("SUMMARY")
    print("=" * 120)
    print(f"Total runs analyzed: {len(all_runs)}")
    print(f"Unique configurations (excl. seed): {len(config_groups)}")
    print(f"Unique (model, dataset, method) combinations: {len(best_singular)}")

    # Count wins
    btn_wins = als_wins = ties = 0
    for dataset in datasets:
        for model in models:
            btn_stats = best_averaged.get((model, dataset, "BTN"))
            als_stats = best_averaged.get((model, dataset, "ALS"))
            if btn_stats and als_stats:
                if btn_stats["mean"] > als_stats["mean"]:
                    btn_wins += 1
                elif als_stats["mean"] > btn_stats["mean"]:
                    als_wins += 1
                else:
                    ties += 1

    print(f"\nHead-to-head (averaged, where both methods have results):")
    print(f"  BTN wins: {btn_wins}")
    print(f"  ALS wins: {als_wins}")
    print(f"  Ties: {ties}")


if __name__ == "__main__":
    main()
