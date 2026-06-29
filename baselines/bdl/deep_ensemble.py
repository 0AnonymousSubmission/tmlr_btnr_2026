# type: ignore
from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE


class _EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim * 2))

        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        out = self.network(x)
        mean = out[..., : self.output_dim]
        log_var = out[..., self.output_dim :]
        return mean, log_var


class DeepEnsemble(BaselineModel):
    def __init__(
        self,
        n_ensemble: int = 5,
        hidden_dims: List[int] = None,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 20,
        activation: str = "relu",
    ):
        self.n_ensemble = n_ensemble
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.activation = activation
        self.device = DEVICE

        self.models: List[_EnsembleMLP] = []
        self._n_features: int = 0

    def _gaussian_nll_loss(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        var = torch.exp(log_var) + 1e-6
        return 0.5 * torch.mean(log_var + (target - mean) ** 2 / var)

    def _train_single_model(
        self,
        model: _EnsembleMLP,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        verbose: bool,
        model_idx: int,
    ) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                mean, log_var = model(x_batch)
                loss = self._gaussian_nll_loss(mean, log_var, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            val_loss = avg_loss
            if X_val is not None and y_val is not None:
                val_loss = self._compute_single_val_loss(model, X_val, y_val)
                history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(
                        f"  Model {model_idx + 1}: Early stopping at epoch {epoch + 1}"
                    )
                break

            if verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                msg = f"  Model {model_idx + 1} Epoch {epoch + 1:3d}/{self.n_epochs} | Loss: {avg_loss:.4f}"
                if X_val is not None:
                    msg += f" | Val: {history['val_loss'][-1]:.4f}"
                print(msg)

        if best_state is not None:
            model.load_state_dict(best_state)

        return {"history": history, "best_val_loss": best_val_loss}

    def _compute_single_val_loss(
        self,
        model: _EnsembleMLP,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        model.eval()
        with torch.no_grad():
            mean, log_var = model(X_val)
            loss = self._gaussian_nll_loss(mean, log_var, y_val)
        return loss.item()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X_train = X_train.to(dtype=torch.float64, device=self.device)
        y_train = y_train.to(dtype=torch.float64, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)

        if X_val is not None:
            X_val = X_val.to(dtype=torch.float64, device=self.device)
        if y_val is not None:
            y_val = y_val.to(dtype=torch.float64, device=self.device)
            if y_val.ndim == 1:
                y_val = y_val.unsqueeze(-1)

        self._n_features = X_train.shape[1]
        self.models = []

        all_histories = []

        for i in range(self.n_ensemble):
            if verbose:
                print(f"Training ensemble member {i + 1}/{self.n_ensemble}")

            torch.manual_seed(torch.initial_seed() + i)

            model = _EnsembleMLP(
                input_dim=self._n_features,
                hidden_dims=self.hidden_dims,
                output_dim=1,
                activation=self.activation,
            ).to(dtype=torch.float64, device=self.device)

            result = self._train_single_model(
                model, X_train, y_train, X_val, y_val, verbose, i
            )

            self.models.append(model)
            all_histories.append(result)

        return {"ensemble_histories": all_histories}

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = X.to(dtype=torch.float64, device=self.device)

        all_means = []
        all_vars = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                mean, log_var = model(X)
                var = torch.exp(log_var)
                all_means.append(mean)
                all_vars.append(var)

        all_means = torch.stack(all_means, dim=0)
        all_vars = torch.stack(all_vars, dim=0)

        ensemble_mean = all_means.mean(dim=0)

        aleatoric_var = all_vars.mean(dim=0)
        epistemic_var = all_means.var(dim=0)
        total_var = aleatoric_var + epistemic_var
        ensemble_std = torch.sqrt(total_var)

        return BaselineResult(
            mean=ensemble_mean.cpu(),
            std=ensemble_std.cpu(),
            extra={
                "n_ensemble": self.n_ensemble,
                "aleatoric_std": torch.sqrt(aleatoric_var).cpu(),
                "epistemic_std": torch.sqrt(epistemic_var).cpu(),
            },
        )

    def get_num_parameters(self) -> int:
        if not self.models:
            return 0
        return sum(p.numel() for m in self.models for p in m.parameters())
