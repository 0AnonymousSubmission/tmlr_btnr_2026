# type: ignore
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from omegaconf import DictConfig

from utils.dataset_loader import load_dataset, append_bias
from utils.device_utils import move_data_to_device
from tensor.builder import Inputs
from model.load_ucirepo import (
    datasets as uci_datasets,
    one_hot_with_cap,
    DATASETS_WITH_TARGET_FIX,
)

_DATA_CACHE = {}


def load_full_dataset_for_kfold(
    dataset_name: str, cap: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    dataset_map = {name: (dataset_id, task) for name, dataset_id, task in uci_datasets}
    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    dataset_id, task = dataset_map[dataset_name]
    dataset = fetch_ucirepo(id=dataset_id)
    X, y = dataset.data.features, dataset.data.targets

    if dataset_id in DATASETS_WITH_TARGET_FIX:
        target_col = DATASETS_WITH_TARGET_FIX[dataset_id]
        y = X[[target_col]]
        X = X.drop(columns=[target_col])

    X = X.dropna(axis=1)
    X_all, orig_num_cols, _ = one_hot_with_cap(X, cap=cap)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if (
        pd.api.types.is_string_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or y.dtype == "object"
    ):
        y = y.astype("category").cat.codes.astype(float)

    X_tensor = torch.tensor(X_all.values, dtype=torch.float64)
    y_tensor = torch.tensor(y.values, dtype=torch.float64)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)

    return (
        X_tensor,
        y_tensor,
        {
            "name": dataset_name,
            "dataset_id": dataset_id,
            "n_samples": len(X_tensor),
            "n_features": X_tensor.shape[1],
            "task": task,
            "orig_num_cols": orig_num_cols,
        },
    )


def prepare_fold_data(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_ratio: float = 0.15,
) -> Dict:
    n_val = int(len(train_idx) * val_ratio)
    rng = np.random.default_rng(42)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)

    return {
        "X_train": X[shuffled[n_val:]],
        "y_train": y[shuffled[n_val:]],
        "X_val": X[shuffled[:n_val]],
        "y_val": y[shuffled[:n_val]],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }


def normalize_fold_data(data: Dict, orig_num_cols: List, n_features: int) -> Dict:
    n_numeric = len(orig_num_cols) if orig_num_cols else n_features
    if n_numeric <= 0:
        return data

    X_train_numeric = data["X_train"][:, :n_numeric]
    mean = X_train_numeric.mean(dim=0, keepdim=True)
    std = X_train_numeric.std(dim=0, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))

    for key in ["X_train", "X_val", "X_test"]:
        X = data[key]
        X_normalized = (X[:, :n_numeric] - mean) / std
        data[key] = (
            torch.cat([X_normalized, X[:, n_numeric:]], dim=1)
            if n_numeric < X.shape[1]
            else X_normalized
        )

    return data


def create_data_loaders(data: Dict, input_dims, output_dims, batch_size: int):
    def make_loader(X, y, bs):
        y_squeezed = y.squeeze(-1) if not output_dims else y
        return Inputs(
            inputs=[X],
            outputs=[y_squeezed],
            outputs_labels=output_dims,
            input_labels=input_dims,
            batch_dim="s",
            batch_size=bs,
        )

    return (
        make_loader(data["X_train"], data["y_train"], batch_size),
        make_loader(data["X_val"], data["y_val"], batch_size),
        make_loader(data["X_test"], data["y_test"], data["X_test"].shape[0]),
    )


def get_data_for_run(cfg: DictConfig):
    global _DATA_CACHE

    if cfg.training.mode == "kfold":
        cache_key = (cfg.dataset.name, cfg.dataset.cap, cfg.seed, cfg.fold_idx)
        
        if cache_key in _DATA_CACHE:
            print(f"Using cached data for {cfg.dataset.name} (fold {cfg.fold_idx})")
            return _DATA_CACHE[cache_key]

        X, y, info = load_full_dataset_for_kfold(cfg.dataset.name, cap=cfg.dataset.cap)
        kf = KFold(n_splits=cfg.training.n_folds, shuffle=True, random_state=cfg.seed)
        folds = list(kf.split(X))
        train_idx, test_idx = folds[cfg.fold_idx]

        data = prepare_fold_data(X, y, train_idx, test_idx, cfg.training.val_ratio)
        data = normalize_fold_data(
            data, info.get("orig_num_cols", []), info["n_features"]
        )
        _DATA_CACHE[cache_key] = (move_data_to_device(data), info["n_features"])
        return _DATA_CACHE[cache_key]

    csv_path = cfg.dataset.get("csv_path")
    task = cfg.dataset.get("task", "regression")
    cache_key = (cfg.dataset.name, cfg.dataset.cap, csv_path)
    
    if cache_key in _DATA_CACHE:
        print(f"Using cached data for {cfg.dataset.name}")
        return _DATA_CACHE[cache_key]

    data, info = load_dataset(
        cfg.dataset.name, csv_path=csv_path, task=task, cap=cfg.dataset.cap
    )
    _DATA_CACHE[cache_key] = (move_data_to_device(data), info["n_features"])
    return _DATA_CACHE[cache_key]



