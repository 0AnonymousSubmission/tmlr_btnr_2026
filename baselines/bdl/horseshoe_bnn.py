# type: ignore
from typing import Dict, Optional, Any
from datetime import datetime
import torch
import torch.optim as optim
import warnings

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE

try:
    from horseshoe_bnn.models import HorseshoeBNN as _HorseshoeBNNModel
    from horseshoe_bnn.data_handling.dataset import Dataset as HorseshoeDataset
    from horseshoe_bnn.parameters import HorseshoeHyperparameters
    HORSESHOE_AVAILABLE = True
except ImportError:
    HORSESHOE_AVAILABLE = False
    warnings.warn(
        "horseshoe-bnn not installed. Install: pip install git+https://github.com/microsoft/horseshoe-bnn.git"
    )

class HorseshoeBNN(BaselineModel):

    def __init__(
        self,
        n_hidden_units: int = 64,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.001,
        n_samples: int = 10,
        n_samples_testing: int = 100,
        var_noise: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        if not HORSESHOE_AVAILABLE:
            raise ImportError(
                "horseshoe-bnn not installed. Install: pip install git+https://github.com/microsoft/horseshoe-bnn.git"
            )

        self.n_hidden_units = n_hidden_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_samples = n_samples
        self.n_samples_testing = n_samples_testing
        self.var_noise = var_noise
        self.seed = seed
        self.device = DEVICE

        self.model = None
        self._n_features: int = 0
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X = X_train.cpu().numpy().astype('float32')
        y = y_train.cpu().numpy().astype('float32').ravel()
        self._n_features = X.shape[1]
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8
        y_scaled = (y - self._y_mean)/self._y_std

        hyperparams = HorseshoeHyperparameters(
            n_features=X.shape[1],
            n_hidden_units=self.n_hidden_units,
            n_samples=self.n_samples,
            n_samples_testing=self.n_samples_testing,
            batch_size=self.batch_size,
            var_noise=self.var_noise,
            learning_rate=self.lr,
            classification=False,
            dataset_name="tabular",
            timestamp=datetime.now().timestamp(),
            mixing_coefficient=0.25,
            sigma1=1.0,
            sigma2=0.002,
            bayesian_weight_rho_scale=-3.0,
            bayesian_bias_rho_scale=-3.0,
            bayesian_scale=0.1,
            horseshoe_scale=0.1,
            weight_cauchy_scale=1.0,
            global_cauchy_scale=1.0,
            beta_rho_scale=-3.0,
            log_tau_mean=None,
            log_tau_rho_scale=-3.0,
            bias_rho_scale=-3.0,
            log_v_mean=None,
            log_v_rho_scale=-3.0,
        )

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        
        try:
            self.model = _HorseshoeBNNModel(torch.device('cpu'), hyperparams)
            self.model = self.model.float()

            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            train_dataset = HorseshoeDataset(X, y_scaled)

            for epoch in range(self.n_epochs):
                loss, rmse, mae = self.model.train_model(train_dataset, epoch, optimizer, visualize_errors=False)
                if verbose and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                    print(f"Epoch {epoch + 1:3d}/{self.n_epochs} | Loss: {float(loss):.4f} | RMSE: {float(rmse):.4f}")
        finally:
            torch.set_default_dtype(prev_dtype)

        return {"n_epochs": self.n_epochs}

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_np = X.cpu().numpy().astype('float32')
        y_dummy = X_np[:, 0] * 0

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        
        try:
            test_dataset = HorseshoeDataset(X_np, y_dummy)
            pred_dist, _, _, _ = self.model.predict(
                test_dataset, mean_y_train=self._y_mean, std_y_train=self._y_std
            )
        finally:
            torch.set_default_dtype(prev_dtype)

        import numpy as np
        means = torch.tensor([float(d.mean) for d in pred_dist.distributions], dtype=torch.float64)
        stds = torch.tensor([np.sqrt(float(d.variance) + float(d.var_noise)) for d in pred_dist.distributions], dtype=torch.float64)

        return BaselineResult(
            mean=means.unsqueeze(-1),
            std=stds.unsqueeze(-1),
            extra={"method": "HorseshoeBNN", "n_samples": self.n_samples_testing},
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
