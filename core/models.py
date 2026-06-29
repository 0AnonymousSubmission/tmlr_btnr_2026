# type: ignore
import torch
from omegaconf import DictConfig, OmegaConf

from model import MODELS


def create_model(cfg: DictConfig, n_features: int):
    model_name = cfg.model.name
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODELS.keys())}"
        )

    params = {
        "L": cfg.model.L,
        "bond_dim": cfg.model.bond_dim,
        "phys_dim": n_features,
        "output_dim": 1,
        "init_strength": cfg.model.init_strength,
        "use_tn_normalization": True,
    }

    if cfg.model.output_site is not None:
        params["output_site"] = cfg.model.output_site

    if model_name == "LMPO2":
        params["reduction_factor"] = cfg.model.reduction_factor
        params["mpo_bond_dim"] = cfg.model.mpo_bond_dim

    return MODELS[model_name](**params)


TN_ONLY_KEYS = {"name", "patience", "L", "bond_dim", "init_strength", "output_site"}


def create_baseline_model(cfg: DictConfig, seed: int = 0):
    from baselines import BASELINE_MODELS

    model_name = cfg.model.name
    if model_name not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline: {model_name}. Available: {list(BASELINE_MODELS.keys())}"
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = {k: v for k, v in model_cfg.items() if k not in TN_ONLY_KEYS and v is not None}
    model_cfg["seed"] = seed

    return BASELINE_MODELS[model_name](**model_cfg)


def count_parameters(tn) -> int:
    return sum(t.data.numel() for t in tn.tensors)
