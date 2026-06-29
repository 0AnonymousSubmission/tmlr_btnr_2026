# type: ignore
from typing import Dict, Optional, Any
import torch
import gpytorch
from gpytorch.models import ExactGP as GPyTorchExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE


class _ExactGPModel(GPyTorchExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_type: str = "rbf",
        ard_num_dims: Optional[int] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
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


class ExactGP(BaselineModel):
    def __init__(
        self,
        kernel_type: str = "rbf",
        use_ard: bool = True,
        n_epochs: int = 100,
        lr: float = 0.1,
        seed: int = 0,
        **kwargs,
    ):
        self.kernel_type = kernel_type
        self.use_ard = use_ard
        self.n_epochs = n_epochs
        self.lr = lr
        # Accepted for interface compatibility; ExactGP training is deterministic
        # and does not depend on the seed.
        self.seed = seed
        self.device = DEVICE

        self.model: Optional[_ExactGPModel] = None
        self.likelihood: Optional[GaussianLikelihood] = None
        self._n_features: int = 0

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

        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = _ExactGPModel(
            X_train,
            y_train,
            self.likelihood,
            kernel_type=self.kernel_type,
            ard_num_dims=ard_dims,
        ).to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            history["train_loss"].append(loss.item())

            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val)
                history["val_loss"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }

            if verbose and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                msg = f"Epoch {epoch + 1:3d}/{self.n_epochs} | Loss: {loss.item():.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {history['val_loss'][-1]:.4f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)

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
            },
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
