# type: ignore
import time
import torch
from typing import Dict
from omegaconf import DictConfig

from utils.dataset_loader import append_bias
from utils.device_utils import DEVICE, move_tn_to_device
from tensor.btn import BTN
from tensor.tn_als import TNALS
from model.utils import REGRESSION_METRICS
from core.data import create_data_loaders
from core.models import create_model, count_parameters
from core.metrics import (
    safe_float,
    extract_loss,
    extract_bond_dims,
    extract_btn_metrics,
    compute_quality,
)
from core.uncertainty import compute_uncertainty_metrics


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "auto":
        return DEVICE
    return torch.device(cfg.device)


def is_singular_error(e):
    if isinstance(e, torch.linalg.LinAlgError):
        return True
    if isinstance(e, ValueError):
        msg = str(e).lower()
        return "positivedefinite" in msg or "covariance" in msg
    return False


def generate_run_id(cfg: DictConfig, seed: int, fold: int = None) -> str:
    parts = [
        cfg.method.name,
        cfg.model.name,
        f"L{cfg.model.L}",
        f"d{cfg.model.bond_dim}",
    ]

    if cfg.method.name == "BTN":
        parts.append(f"trim{cfg.method.trimming_threshold:.2f}".rstrip("0").rstrip("."))
        parts.append(cfg.method.trim_method)

    parts.append(f"bias{1 if cfg.dataset.added_bias else 0}")
    parts.append(f"seed{seed}")

    if fold is not None:
        parts.append(f"fold{fold}")

    return "-".join(parts)


def build_base_result(cfg, seed, fold, n_features, elapsed_time):
    return {
        "run_id": generate_run_id(cfg, seed, fold),
        "seed": seed,
        "fold": fold,
        "model": cfg.model.name,
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "n_features": n_features,
        "elapsed_time": elapsed_time,
    }


def build_singular_result(cfg, seed, fold, error):
    return {
        "run_id": generate_run_id(cfg, seed, fold),
        "seed": seed,
        "fold": fold,
        "model": cfg.model.name,
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "success": True,
        "singular": True,
        "error": str(error),
    }


@torch.inference_mode()
def run_tn_experiment(
    cfg: DictConfig,
    data: Dict,
    seed: int,
    verbose: bool = False,
    tracker=None,
    fold: int = None,
) -> dict:
    start_time = time.time()
    device = get_device(cfg)
    method = cfg.method.name

    torch.manual_seed(seed)

    working_data = {k: v.clone() for k, v in data.items()}

    if cfg.dataset.added_bias:
        working_data = append_bias(working_data)

    n_features = working_data["X_train"].shape[1]

    try:
        model = create_model(cfg, n_features)
        move_tn_to_device(model.tn)

        train_loader, val_loader, test_loader = create_data_loaders(
            working_data, model.input_dims, model.output_dims, cfg.training.batch_size
        )

        if method == "BTN":
            tn = BTN(
                mu=model.tn,
                data_stream=train_loader,
                batch_dim="s",
                method="cholesky",
                device=device,
                bond_prior_alpha=cfg.method.bond_prior_alpha,
            )
            tn.threshold = cfg.method.trimming_threshold
        else:
            tn = TNALS(
                mu=model.tn,
                data_stream=train_loader,
                batch_dim="s",
                method="cholesky",
                device=device,
                bond_prior_alpha=cfg.method.bond_prior_alpha,
            )

        tn.register_data_streams(val_loader, test_loader)

        if tracker:
            hparams = {
                "seed": seed,
                "dataset": cfg.dataset.name,
                "n_features": n_features,
                "L": cfg.model.L,
                "bond_dim": cfg.model.bond_dim,
                "model": cfg.model.name,
                "method": method,
            }
            if method == "BTN":
                hparams["trimming_threshold"] = cfg.method.trimming_threshold

            if fold is not None:
                hparams["fold"] = fold
            tracker.log_hparams(hparams)

        excluded = (
            set(tn.output_dimensions) if hasattr(tn, "output_dimensions") else set()
        )
        bonds = [i for i in tn.mu.ind_map if i not in excluded]
        nodes = list(tn.mu.tag_map.keys())

        if method == "BTN":
            _ = tn.compute_elbo(verbose=False, relative=True)
            initial_metrics = extract_btn_metrics(tn)
        else:
            initial_metrics = {
                "bond_dims": extract_bond_dims(tn, excluded),
                "n_parameters": count_parameters(tn.mu),
            }

        best_val_quality = float("-inf")
        best_test_quality = None
        best_test_loss = None
        best_epoch = -1
        best_model_state = None
        stopped_early = False
        patience_counter = 0
        soft_trim_started_epoch = None

        n_epochs = cfg.training.n_epochs
        patience = cfg.training.patience
        min_delta = cfg.training.min_delta
        warmup_epochs = cfg.method.warmup_epochs

        for epoch in range(n_epochs):
            if method == "BTN":
                for node_tag in nodes:
                    tn.update_sigma_node(node_tag)
                    tn.update_mu_node(node_tag)

                if epoch >= warmup_epochs:
                    excluded_bonds = tn.get_soft_trim_excluded_bonds()
                    for bond_tag in bonds:
                        if bond_tag not in excluded_bonds:
                            tn.update_bond(bond_tag)

                tn.update_tau()

                if tn.has_pending_soft_trims():
                    if (
                        epoch - soft_trim_started_epoch
                        >= cfg.method.soft_trim_relaxation
                    ):
                        tn.finalize_soft_trim_bonds(verbose=False)
                        soft_trim_started_epoch = None

                trim_start = warmup_epochs + cfg.method.trim_every
                if cfg.method.trim_every and epoch >= trim_start:
                    trim_epoch = epoch - trim_start
                    if (
                        trim_epoch % cfg.method.trim_every == 0
                        and not tn.has_pending_soft_trims()
                    ):
                        if cfg.method.trim_method == "gamma":
                            tn.trim_bonds_by_gamma(
                                threshold=cfg.method.trimming_threshold, verbose=False
                            )
                    elif cfg.method.trim_method == "relevance":
                        tn.threshold = cfg.method.trimming_threshold
                        tn.trim_bonds(verbose=False)
            else:
                for node_tag in nodes:
                    tn.update_mu_node(node_tag)

                if epoch >= warmup_epochs and cfg.method.decay != 1.0:
                    for bond_tag in bonds:
                        c0 = tn.q_bonds[bond_tag].concentration
                        r0 = tn.q_bonds[bond_tag].rate
                        tn.update_bond(bond_tag, c0 * cfg.method.decay, r0)

            train_scores = tn.evaluate(REGRESSION_METRICS, data_stream=train_loader)
            val_scores = tn.evaluate(REGRESSION_METRICS, data_stream=val_loader)
            test_scores = tn.evaluate(REGRESSION_METRICS, data_stream=test_loader)

            train_quality = compute_quality(train_scores)
            val_quality = compute_quality(val_scores)
            test_quality = compute_quality(test_scores)
            train_loss = extract_loss(train_scores)
            val_loss = extract_loss(val_scores)
            test_loss = extract_loss(test_scores)

            if val_quality is not None and val_quality > best_val_quality + min_delta:
                best_val_quality = val_quality
                best_test_quality = test_quality
                best_test_loss = test_loss
                best_epoch = epoch
                patience_counter = 0
                if method == "ALS":
                    best_model_state = {
                        tag: tn.mu[tag].data.clone() for tag in tn.mu.tag_map.keys()
                    }
            else:
                patience_counter += 1

            if tracker:
                metrics = {
                    "train_loss": safe_float(train_loss),
                    "train_quality": safe_float(train_quality),
                    "val_loss": safe_float(val_loss),
                    "val_quality": safe_float(val_quality),
                    "test_loss": safe_float(test_loss),
                    "test_quality": safe_float(test_quality),
                    "patience_counter": patience_counter,
                }
                if method == "BTN":
                    btn_metrics = extract_btn_metrics(tn)
                    metrics.update(
                        {
                            k: v
                            for k, v in btn_metrics.items()
                            if k not in ("bond_expectations", "bond_dims")
                        }
                    )
                    for bond_tag, dim in btn_metrics.get("bond_dims", {}).items():
                        metrics[f"dim_{bond_tag}"] = dim
                tracker.log_metrics(metrics, step=epoch + 1)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                msg = f"  Epoch {epoch + 1:3d} | Train: {train_quality:.4f} | Val: {val_quality:.4f}"
                if method == "BTN":
                    btn_m = extract_btn_metrics(tn)
                    msg += f" | ELBO: {btn_m['elbo_relative']:.2f} | Tau: {btn_m['tau_mean']:.4f}"
                print(msg)

            if patience and patience_counter >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                stopped_early = True
                break

        if method == "ALS" and best_model_state:
            for tag, tensor_data in best_model_state.items():
                tn.mu[tag].modify(data=tensor_data)

        test_scores = tn.evaluate(REGRESSION_METRICS, data_stream=test_loader)
        test_quality = compute_quality(test_scores)
        test_loss = extract_loss(test_scores)

        # Uncertainty-quality suite (BTN only; ALS has no predictive variance).
        # Gated by cfg.uncertainty.enabled; never breaks a run on failure.
        unc_cfg = cfg.get("uncertainty")
        uncertainty_metrics = None
        if method == "BTN" and unc_cfg is not None and unc_cfg.get("enabled", False):
            try:
                y_true_unc = working_data["y_test"].reshape(-1).to(torch.float64)
                mean, epi_var, ale_var = tn.predict_mean_var(data_stream=test_loader)
                total_var = epi_var + ale_var
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
                # Average uncertainty decomposition magnitudes.
                uncertainty_metrics["unc_epistemic_std"] = safe_float(
                    epi_var.clamp_min(0).sqrt().mean()
                )
                uncertainty_metrics["unc_aleatoric_std"] = safe_float(
                    ale_var.clamp_min(0).sqrt().mean()
                )
                if tracker:
                    tracker.log_summary(
                        {k: v for k, v in uncertainty_metrics.items()
                         if isinstance(v, (int, float))}
                    )
                    tracker.log_curves(
                        {k: v for k, v in uncertainty_metrics.items()
                         if isinstance(v, dict)}
                    )
            except Exception as unc_err:
                import traceback
                traceback.print_exc()
                uncertainty_metrics = {"unc_error": str(unc_err)}
                if verbose:
                    print(f"  [uncertainty] skipped: {unc_err}")

        if method == "BTN":
            final_metrics = extract_btn_metrics(tn)
        else:
            final_metrics = {
                "bond_dims": extract_bond_dims(tn, excluded),
                "n_parameters": count_parameters(tn.mu),
            }

        elapsed_time = time.time() - start_time

        result = build_base_result(cfg, seed, fold, n_features, elapsed_time)
        result.update(
            {
                "train_loss": safe_float(train_loss),
                "train_quality": safe_float(train_quality),
                "val_loss": safe_float(val_loss),
                "val_quality": safe_float(best_val_quality)
                if best_val_quality != float("-inf")
                else 0.0,
                "test_loss": safe_float(best_test_loss),
                "test_quality": safe_float(best_test_quality),
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
                "patience_counter": patience_counter,
                "initial_bond_dims": initial_metrics.get("bond_dims", {}),
                "final_bond_dims": final_metrics.get("bond_dims", {}),
                "initial_n_parameters": initial_metrics.get("n_parameters", 0),
                "final_n_parameters": final_metrics.get("n_parameters", 0),
                "success": True,
                "singular": False,
            }
        )

        if method == "BTN":
            result.update(
                {
                    "elbo_raw": final_metrics["elbo_raw"],
                    "elbo_relative": final_metrics["elbo_relative"],
                    "tau_mean": final_metrics["tau_mean"],
                    "bond_mean_avg": final_metrics["bond_mean_avg"],
                }
            )

        if uncertainty_metrics is not None:
            result["uncertainty"] = uncertainty_metrics

        if tracker:
            tracker.log_summary(
                {
                    "test_quality": result["test_quality"],
                    "test_loss": result["test_loss"],
                    "best_val_quality": result["val_quality"],
                    "n_parameters": result["final_n_parameters"],
                    "elapsed_time": elapsed_time,
                }
            )

        return result

    except Exception as e:
        if is_singular_error(e):
            return build_singular_result(cfg, seed, fold, e)
        raise
