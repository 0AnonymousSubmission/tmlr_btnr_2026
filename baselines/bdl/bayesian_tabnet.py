# type: ignore
from typing import Dict, Optional, Any, List
import torch
import numpy as np
import warnings

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    warnings.warn(
        "pytorch-tabnet not installed. Install: pip install pytorch-tabnet"
    )


class BayesianTabNet(BaselineModel):

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        momentum: float = 0.02,
        n_epochs: int = 200,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        lr: float = 0.02,
        patience: int = 15,
        n_ensemble: int = 5,
    ):
        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet not installed. Install: pip install pytorch-tabnet"
            )

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.lr = lr
        self.patience = patience
        self.n_ensemble = n_ensemble
        self.device = DEVICE

        self.models: List[TabNetRegressor] = []
        self._n_features: int = 0

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X = X_train.cpu().numpy().astype(np.float32)
        y = y_train.cpu().numpy().astype(np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self._n_features = X.shape[1]

        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.cpu().numpy().astype(np.float32)
            y_v = y_val.cpu().numpy().astype(np.float32)
            if y_v.ndim == 1:
                y_v = y_v.reshape(-1, 1)
            eval_set = [(X_v, y_v)]

        self.models = []
        device_name = "cuda" if self.device.type == "cuda" else "cpu"

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        
        try:
            for i in range(self.n_ensemble):
                if verbose:
                    print(f"Training TabNet ensemble member {i + 1}/{self.n_ensemble}")

                model = TabNetRegressor(
                    n_d=self.n_d,
                    n_a=self.n_a,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                    n_independent=self.n_independent,
                    n_shared=self.n_shared,
                    lambda_sparse=self.lambda_sparse,
                    momentum=self.momentum,
                    seed=i,
                    optimizer_params=dict(lr=self.lr),
                    device_name=device_name,
                    verbose=0,
                )

                model.fit(
                    X_train=X,
                    y_train=y,
                    eval_set=eval_set,
                    eval_metric=["rmse"],
                    max_epochs=self.n_epochs,
                    patience=self.patience,
                    batch_size=self.batch_size,
                    virtual_batch_size=self.virtual_batch_size,
                )

                self.models.append(model)
        finally:
            torch.set_default_dtype(prev_dtype)

        return {"n_ensemble": self.n_ensemble}

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_np = X.cpu().numpy().astype(np.float32)

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        
        try:
            all_preds = []
            for model in self.models:
                pred = model.predict(X_np)
                all_preds.append(pred)
        finally:
            torch.set_default_dtype(prev_dtype)

        all_preds = np.stack(all_preds, axis=0)

        mean = np.mean(all_preds, axis=0)
        std = np.std(all_preds, axis=0)

        mean_tensor = torch.from_numpy(mean.astype(np.float64))
        std_tensor = torch.from_numpy(std.astype(np.float64))

        if mean_tensor.ndim == 1:
            mean_tensor = mean_tensor.unsqueeze(-1)
        if std_tensor.ndim == 1:
            std_tensor = std_tensor.unsqueeze(-1)

        return BaselineResult(
            mean=mean_tensor,
            std=std_tensor,
            extra={"method": "BayesianTabNet", "n_ensemble": self.n_ensemble},
        )

    def get_num_parameters(self) -> int:
        if not self.models:
            return 0
        total = 0
        for model in self.models:
            if hasattr(model, "network") and model.network is not None:
                total += sum(p.numel() for p in model.network.parameters())
        return total

    def get_feature_importance(self) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit() first.")
        importances = [model.feature_importances_ for model in self.models]
        return np.mean(importances, axis=0)
