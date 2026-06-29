# type: ignore
from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from baselines.base import BaselineModel, BaselineResult
from utils.device_utils import DEVICE


class _DropoutMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
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
                    nn.Dropout(p=dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MCDropoutNet(BaselineModel):
    def __init__(
        self,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.1,
        n_mc_samples: int = 50,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 20,
        activation: str = "relu",
    ):
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.activation = activation
        self.device = DEVICE

        self.model: Optional[_DropoutMLP] = None
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
        y_train = y_train.to(dtype=torch.float64, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)

        self._n_features = X_train.shape[1]

        self.model = _DropoutMLP(
            input_dim=self._n_features,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            val_loss = avg_loss
            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val)
                history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                msg = f"Epoch {epoch + 1:3d}/{self.n_epochs} | Loss: {avg_loss:.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {history['val_loss'][-1]:.4f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"history": history, "best_val_loss": best_val_loss}

    def _compute_val_loss(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        self.model.eval()

        X_val = X_val.to(dtype=torch.float64, device=self.device)
        y_val = y_val.to(dtype=torch.float64, device=self.device)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(-1)

        with torch.no_grad():
            y_pred = self.model(X_val)
            mse = torch.mean((y_pred - y_val) ** 2).item()

        return mse

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = X.to(dtype=torch.float64, device=self.device)

        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = self.model(X)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return BaselineResult(
            mean=mean.cpu(),
            std=std.cpu(),
            extra={
                "n_mc_samples": self.n_mc_samples,
                "dropout_rate": self.dropout_rate,
            },
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
