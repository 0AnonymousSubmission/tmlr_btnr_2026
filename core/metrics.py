# type: ignore
import math
import torch
import quimb.tensor as qt

from model.utils import compute_quality as _compute_quality


def safe_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            val = value.item() if value.numel() == 1 else float(value)
        else:
            val = float(value)
        return val if math.isfinite(val) else default
    except:
        return default


def extract_loss(scores):
    loss = scores.get("loss", (0, 1))
    if isinstance(loss, tuple):
        return loss[0] / loss[1] if loss[1] != 0 else 0
    return float(loss) if torch.is_tensor(loss) else loss


def extract_bond_dims(tn, excluded_dims):
    bond_dims = {}
    try:
        for bond_tag in tn.q_bonds:
            if bond_tag not in excluded_dims:
                bond_dims[bond_tag] = int(tn.mu.ind_size(bond_tag))
    except:
        pass
    return bond_dims


def extract_btn_metrics(btn) -> dict:
    metrics = {}

    extractors = [
        ("elbo_raw", lambda: btn._compute_raw_elbo()),
        ("elbo_relative", lambda: btn.compute_elbo(verbose=False, relative=True)),
        ("e_log_p_nodes", lambda: btn._compute_e_log_p_nodes()),
        ("e_log_p_bonds", lambda: btn._compute_e_log_p_bonds()),
        ("e_log_p_tau", lambda: btn._compute_e_log_p_tau()),
        ("h_nodes", lambda: btn._H_nodes()),
        ("h_bonds", lambda: btn._H_bonds()),
        ("h_tau", lambda: btn._H_tau()),
    ]

    for key, fn in extractors:
        try:
            metrics[key] = safe_float(fn())
        except:
            metrics[key] = 0.0

    try:
        tau_mean = btn.q_tau.mean()
        if isinstance(tau_mean, qt.Tensor):
            tau_mean = tau_mean.data
        metrics["tau_mean"] = float(
            tau_mean.item() if torch.is_tensor(tau_mean) else tau_mean
        )
    except:
        metrics["tau_mean"] = 0.0

    try:
        from core.models import count_parameters

        excluded = set(btn.output_dimensions)
        bond_expectations = {}
        for bond_tag in btn.q_bonds:
            if bond_tag not in excluded:
                mean = btn.q_bonds[bond_tag].mean()
                if isinstance(mean, qt.Tensor):
                    mean = mean.data
                bond_expectations[bond_tag] = float(mean.mean().item())

        metrics["bond_expectations"] = bond_expectations
        metrics["bond_dims"] = extract_bond_dims(btn, excluded)
        metrics["bond_mean_avg"] = (
            sum(bond_expectations.values()) / len(bond_expectations)
            if bond_expectations
            else 0.0
        )
        metrics["n_parameters"] = count_parameters(btn.mu)
    except:
        metrics["bond_expectations"] = {}
        metrics["bond_dims"] = {}
        metrics["bond_mean_avg"] = 0.0
        metrics["n_parameters"] = 0

    return metrics


def compute_quality(scores):
    return _compute_quality(scores)
