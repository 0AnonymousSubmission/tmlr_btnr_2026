# type: ignore
from typing import Dict, Optional, Any, List
import torch
import numpy as np
import os
import warnings

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE

try:
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    from bde import BdeRegressor
    from bde.loss import GaussianNLL

    BDE_AVAILABLE = True
except ImportError:
    BDE_AVAILABLE = False
    warnings.warn(
        "bde package not installed. Install with: pip install sklearn-contrib-bde"
    )


class BdeMile(BaselineModel):

    def __init__(
        self,
        hidden_layers: List[int] = None,
        n_members: int = 8,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 5000,
        n_samples: int = 2000,
        n_thinning: int = 2,
        patience: int = 10,
        validation_split: float = 0.15,
        seed: int = 0,
        activation: str = "relu",
    ):
        if not BDE_AVAILABLE:
            raise ImportError(
                "bde package not installed. Install with: pip install sklearn-contrib-bde"
            )

        self.hidden_layers = hidden_layers if hidden_layers is not None else [16, 16]
        self.n_members = n_members
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.n_samples = n_samples
        self.n_thinning = n_thinning
        self.patience = patience
        self.validation_split = validation_split
        self.seed = seed
        self.activation = activation
        self.device = DEVICE

        self.model: Optional[BdeRegressor] = None
        self._n_features: int = 0
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X = X_train.cpu().numpy().astype(np.float32)
        y = y_train.cpu().numpy().astype(np.float32)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self._n_features = X.shape[1]

        self._x_mean = np.mean(X, axis=0)
        self._x_std = np.std(X, axis=0) + 1e-8
        self._y_mean = np.mean(y, axis=0)
        self._y_std = np.std(y, axis=0) + 1e-8

        X_scaled = (X - self._x_mean) / self._x_std
        y_scaled = (y - self._y_mean) / self._y_std

        self.model = BdeRegressor(
            hidden_layers=self.hidden_layers,
            n_members=self.n_members,
            seed=self.seed,
            loss=GaussianNLL(),
            epochs=self.n_epochs,
            validation_split=self.validation_split,
            lr=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            n_samples=self.n_samples,
            n_thinning=self.n_thinning,
            patience=self.patience,
            activation=self.activation,
        )

        if verbose:
            print(f"Training BDE-MILE with {self.n_members} members...")
            print(f"  Hidden layers: {self.hidden_layers}")
            print(f"  Warmup steps: {self.warmup_steps}")
            print(f"  MCMC samples: {self.n_samples}")

        self.model.fit(x=X_scaled, y=y_scaled.ravel())

        return {
            "n_members": self.n_members,
            "n_samples": self.n_samples,
            "warmup_steps": self.warmup_steps,
        }

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_np = X.cpu().numpy().astype(np.float32)
        X_scaled = (X_np - self._x_mean) / self._x_std

        means_scaled, sigmas_scaled = self.model.predict(X_scaled, mean_and_std=True)

        means = np.asarray(means_scaled) * self._y_std + self._y_mean
        sigmas = np.asarray(sigmas_scaled) * self._y_std

        mean_tensor = torch.from_numpy(means.astype(np.float64))
        std_tensor = torch.from_numpy(sigmas.astype(np.float64))

        if mean_tensor.ndim == 1:
            mean_tensor = mean_tensor.unsqueeze(-1)
        if std_tensor.ndim == 1:
            std_tensor = std_tensor.unsqueeze(-1)

        return BaselineResult(
            mean=mean_tensor,
            std=std_tensor,
            extra={
                "n_members": self.n_members,
                "n_samples": self.n_samples,
                "method": "MILE",
            },
        )

    def get_num_parameters(self) -> int:
        if self.model is None or self._n_features == 0:
            return 0

        total_params = 0
        prev_dim = self._n_features

        for hidden_dim in self.hidden_layers:
            total_params += prev_dim * hidden_dim + hidden_dim
            prev_dim = hidden_dim

        total_params += prev_dim * 2 + 2

        return total_params * self.n_members
