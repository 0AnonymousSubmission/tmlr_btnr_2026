"""
Run tracking utilities for BTN, ALS, and baseline experiments.

Uses a unified tracking.csv with run_id format matching Hydra sweep directories:
  BTN:      {exp}_{dataset}_{model}_L{L}_d{D}_p{prior}_i{init}_t{trim}_s{seed}
  ALS:      {exp}_{dataset}_{model}_L{L}_d{D}_p{prior}_i{init}_s{seed}
  baseline: baseline_{dataset}_{model}_{config}_s{seed}
"""

from __future__ import annotations

import csv
import fcntl
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)

TRACKING_COLUMNS = [
    "run_id",
    "experiment_type",
    "dataset",
    "model",
    "L",
    "D",
    "prior",
    "init",
    "trim",
    "config",
    "seed",
    "success",
    "singular",
    "test_quality",
    "val_quality",
    "test_r2",
    "test_mse",
    "test_nll",
    "n_parameters",
    "elapsed_time",
    "output_path",
    "timestamp",
]

DEFAULT_TRACKING_FILE = "tracking.csv"

_tracking_df = None


def generate_tracking_id(cfg: DictConfig, seed: int) -> str:
    """Generate tracking ID matching the backfill format from Hydra sweep dirs."""
    exp_type = cfg.method.name
    model_str = f"{cfg.model.name}_L{cfg.model.L}_d{cfg.model.bond_dim}"
    
    if exp_type == "BTN":
        trim_part = f"_t{cfg.method.trimming_threshold}"
        return (
            f"{exp_type}_{cfg.dataset.name}_{model_str}_"
            f"p{cfg.method.bond_prior_alpha}_i{cfg.model.init_strength}{trim_part}_s{seed}"
        )
    else:
        return (
            f"{exp_type}_{cfg.dataset.name}_{model_str}_"
            f"p{cfg.method.bond_prior_alpha}_i{cfg.model.init_strength}_s{seed}"
        )


def generate_baseline_tracking_id(cfg: DictConfig, seed: int) -> str:
    """Generate tracking ID for baseline experiments matching backfill format."""
    results_dir = cfg.output.results_dir
    config_part = results_dir.split("/")[-1]
    return f"baseline_{cfg.dataset.name}_{cfg.model.name}_{config_part}_s{seed}"


def load_tracking_file(path: str | Path = DEFAULT_TRACKING_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        log.info(f"Tracking file not found at {path}, starting fresh")
        df = pd.DataFrame(columns=TRACKING_COLUMNS)
        df.set_index("run_id", inplace=True)
        return df

    try:
        df = pd.read_csv(path)
        df.set_index("run_id", inplace=True)
        log.info(f"Loaded {len(df)} tracked runs from {path}")
        return df
    except Exception as e:
        log.warning(f"Error loading tracking file: {e}, starting fresh")
        df = pd.DataFrame(columns=TRACKING_COLUMNS)
        df.set_index("run_id", inplace=True)
        return df


def get_tracking_df(path: str | Path = DEFAULT_TRACKING_FILE) -> pd.DataFrame:
    global _tracking_df
    if _tracking_df is None:
        _tracking_df = load_tracking_file(path)
    return _tracking_df


def should_skip_run(df: pd.DataFrame, run_id: str) -> tuple[bool, str, float | None]:
    """Check if run should be skipped. Returns (skip, reason, val_quality)."""
    if run_id not in df.index:
        return False, "not in tracking", None

    row = df.loc[run_id]
    val_quality = row.get("val_quality") or row.get("test_quality") or row.get("test_r2")
    if pd.isna(val_quality):
        val_quality = None

    success = row.get("success")
    if bool(success) and not pd.isna(success):
        return True, "completed", val_quality

    singular = row.get("singular")
    if bool(singular) and not pd.isna(singular):
        return True, "singular", val_quality

    return False, "failed (retrying)", val_quality


def append_run_result(
    result: dict,
    cfg: DictConfig,
    output_path: str | Path,
    path: str | Path = DEFAULT_TRACKING_FILE,
) -> None:
    """Append a run result to the unified tracking CSV."""
    path = Path(path)
    exp_type = cfg.method.name
    is_baseline = exp_type == "baseline"
    
    if is_baseline:
        run_id = generate_baseline_tracking_id(cfg, cfg.seed)
        results_dir = cfg.output.results_dir
        config_part = results_dir.split("/")[-1]
    else:
        run_id = generate_tracking_id(cfg, cfg.seed)

    row = {
        "run_id": run_id,
        "experiment_type": exp_type,
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "L": None if is_baseline else cfg.model.L,
        "D": None if is_baseline else cfg.model.bond_dim,
        "prior": None if is_baseline else cfg.method.bond_prior_alpha,
        "init": None if is_baseline else cfg.model.init_strength,
        "trim": cfg.method.trimming_threshold if exp_type == "BTN" else None,
        "config": config_part if is_baseline else None,
        "seed": cfg.seed,
        "success": result.get("success", False),
        "singular": result.get("singular", False),
        "test_quality": result.get("test_quality"),
        "val_quality": result.get("val_quality") or result.get("best_val_quality"),
        "test_r2": result.get("test_r2"),
        "test_mse": result.get("test_mse"),
        "test_nll": result.get("test_nll"),
        "n_parameters": result.get("n_parameters"),
        "elapsed_time": result.get("elapsed_time"),
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }

    file_exists = path.exists()

    try:
        with open(path, "a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        log.error(f"Failed to append to tracking file: {e}")
