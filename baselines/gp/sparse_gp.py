# type: ignore
from typing import Dict, Optional, Any
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE


class _SparseGPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: str = "rbf",
        ard_num_dims: Optional[int] = None,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()

        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGP(BaselineModel):
    def __init__(
        self,
        n_inducing: int = 100,
        kernel_type: str = "rbf",
        use_ard: bool = True,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.01,
        seed: int = 0,
        **kwargs,
    ):
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.use_ard = use_ard
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = DEVICE

        self.model: Optional[_SparseGPModel] = None
        self.likelihood: Optional[GaussianLikelihood] = None
        self._n_features: int = 0

    def _select_inducing_points(
        self, X_train: torch.Tensor, n_inducing: int
    ) -> torch.Tensor:
        n_samples = X_train.shape[0]

        if n_samples <= n_inducing:
            return X_train.clone()

        indices = torch.randperm(n_samples)[:n_inducing]
        return X_train[indices].clone()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X_train = X_train.to(dtype=torch.float64, device=self.device)
        y_train = y_train.squeeze().to(dtype=torch.float64, device=self.device)

        self._n_features = X_train.shape[1]
        ard_dims = self._n_features if self.use_ard else None

        inducing_points = self._select_inducing_points(X_train, self.n_inducing)

        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = _SparseGPModel(
            inducing_points,
            kernel_type=self.kernel_type,
            ard_num_dims=ard_dims,
        ).to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.lr,
        )

        mll = VariationalELBO(self.likelihood, self.model, num_data=X_train.size(0))

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val)
                history["val_loss"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        "model": {
                            k: v.clone() for k, v in self.model.state_dict().items()
                        },
                        "likelihood": {
                            k: v.clone()
                            for k, v in self.likelihood.state_dict().items()
                        },
                    }

            if verbose and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                msg = f"Epoch {epoch + 1:3d}/{self.n_epochs} | Loss: {avg_loss:.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {history['val_loss'][-1]:.4f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.likelihood.load_state_dict(best_state["likelihood"])

        return {"history": history, "best_val_loss": best_val_loss}

    def _compute_val_loss(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        self.model.eval()
        self.likelihood.eval()

        X_val = X_val.to(dtype=torch.float64, device=self.device)
        y_val = y_val.squeeze().to(dtype=torch.float64, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_val))
            mse = torch.mean((pred.mean - y_val) ** 2).item()

        self.model.train()
        self.likelihood.train()
        return mse

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        self.likelihood.eval()

        X = X.to(dtype=torch.float64, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X))
            mean = pred.mean.unsqueeze(-1)
            std = pred.stddev.unsqueeze(-1)

        return BaselineResult(
            mean=mean.cpu(),
            std=std.cpu(),
            extra={
                "lengthscales": self.model.covar_module.base_kernel.lengthscale.detach().cpu().tolist(),
                "noise": self.likelihood.noise.detach().cpu().item(),
                "n_inducing": self.n_inducing,
            },
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
