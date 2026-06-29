# type: ignore
import time
import torch
import numpy as np
from typing import Dict
from omegaconf import DictConfig

from utils.dataset_loader import append_bias
from core.models import create_baseline_model
from core.metrics import safe_float
from core.uncertainty import compute_uncertainty_metrics


def generate_baseline_run_id(cfg: DictConfig, seed: int, fold: int = None) -> str:
    parts = [cfg.method.name, cfg.model.name, f"seed{seed}"]
    if fold is not None:
        parts.append(f"fold{fold}")
    return "-".join(parts)


def run_baseline_experiment(
    cfg: DictConfig,
    data: Dict,
    seed: int,
    verbose: bool = False,
    tracker=None,
    fold: int = None,
) -> dict:
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    working_data = {k: v.clone() for k, v in data.items()}

    if cfg.dataset.added_bias:
        working_data = append_bias(working_data)

    n_features = working_data["X_train"].shape[1]

    try:
        model = create_baseline_model(cfg, seed=seed)

        if tracker:
            hparams = {
                "seed": seed,
                "dataset": cfg.dataset.name,
                "n_features": n_features,
                "model": cfg.model.name,
                "method": cfg.method.name,
            }
            if fold is not None:
                hparams["fold"] = fold
            tracker.log_hparams(hparams)

        fit_result = model.fit(
            X_train=working_data["X_train"],
            y_train=working_data["y_train"],
            X_val=working_data["X_val"],
            y_val=working_data["y_val"],
            verbose=verbose,
        )

        train_metrics = model.evaluate(working_data["X_train"], working_data["y_train"])
        val_metrics = model.evaluate(working_data["X_val"], working_data["y_val"])
        test_metrics = model.evaluate(working_data["X_test"], working_data["y_test"])
        
        test_result = model.predict(working_data["X_test"])
        extra = test_result.extra

        # Uncertainty-quality suite (same metrics as BTN; gated by config).
        # Uses the model's predictive mean + std -> Gaussian predictive variance.
        unc_cfg = cfg.get("uncertainty")
        uncertainty_metrics = None
        if unc_cfg is not None and unc_cfg.get("enabled", False):
            try:
                y_true_unc = working_data["y_test"].reshape(-1).to(torch.float64)
                mean = test_result.mean.reshape(-1).to(torch.float64)
                std = test_result.std.reshape(-1).to(torch.float64)
                total_var = (std ** 2).clamp_min(1e-12)
                outlier_cfg = unc_cfg.get("outlier", {})
                uncertainty_metrics = compute_uncertainty_metrics(
                    y_true_unc,
                    mean,
                    total_var,
                    confidence=unc_cfg.get("confidence", 0.95),
                    run_outlier=outlier_cfg.get("enabled", True),
                    outlier_fraction=outlier_cfg.get("fraction", 0.1),
                    outlier_scale=outlier_cfg.get("scale", 5.0),
                    seed=seed,
                    keep_curves=unc_cfg.get("keep_curves", False),
                )
                if "epistemic_std" in extra:
                    uncertainty_metrics["unc_epistemic_std"] = safe_float(
                        extra["epistemic_std"].reshape(-1).to(torch.float64).mean()
                    )
                if "aleatoric_std" in extra:
                    uncertainty_metrics["unc_aleatoric_std"] = safe_float(
                        extra["aleatoric_std"].reshape(-1).to(torch.float64).mean()
                    )
            except Exception as unc_err:
                uncertainty_metrics = {"unc_error": str(unc_err)}
                if verbose:
                    print(f"  [uncertainty] skipped: {unc_err}")

        elapsed_time = time.time() - start_time

        result = {
            "run_id": generate_baseline_run_id(cfg, seed, fold),
            "seed": seed,
            "fold": fold,
            "model": cfg.model.name,
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "n_features": n_features,
            "elapsed_time": elapsed_time,
            "train_mse": safe_float(train_metrics["mse"]),
            "train_r2": safe_float(train_metrics["r2"]),
            "train_nll": safe_float(train_metrics["nll"]),
            "val_mse": safe_float(val_metrics["mse"]),
            "val_r2": safe_float(val_metrics["r2"]),
            "val_nll": safe_float(val_metrics["nll"]),
            "test_mse": safe_float(test_metrics["mse"]),
            "test_r2": safe_float(test_metrics["r2"]),
            "test_nll": safe_float(test_metrics["nll"]),
            "test_quality": safe_float(test_metrics["r2"]),
            "n_parameters": model.get_num_parameters(),
            "extra": extra,
            "success": True,
        }

        if uncertainty_metrics is not None:
            result["uncertainty"] = uncertainty_metrics

        if tracker:
            tracker.log_summary(
                {
                    "test_r2": result["test_r2"],
                    "test_mse": result["test_mse"],
                    "test_nll": result["test_nll"],
                    "n_parameters": result["n_parameters"],
                    "elapsed_time": elapsed_time,
                }
            )
            # Persist the uncertainty-quality metrics (scalars -> summary,
            # reliability/sparsification curves -> curves), mirroring BTN.
            if uncertainty_metrics is not None:
                tracker.log_summary(
                    {k: v for k, v in uncertainty_metrics.items()
                     if isinstance(v, (int, float))}
                )
                tracker.log_curves(
                    {k: v for k, v in uncertainty_metrics.items()
                     if isinstance(v, dict)}
                )

        if verbose:
            print(f"\nResults for {cfg.model.name}:")
            print(
                f"  Train R²: {result['train_r2']:.4f} | Val R²: {result['val_r2']:.4f} | Test R²: {result['test_r2']:.4f}"
            )
            print(
                f"  Test NLL: {result['test_nll']:.4f} | Parameters: {result['n_parameters']}"
            )

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "run_id": generate_baseline_run_id(cfg, seed, fold),
            "seed": seed,
            "fold": fold,
            "model": cfg.model.name,
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "elapsed_time": elapsed_time,
            "success": False,
            "error": str(e),
        }
