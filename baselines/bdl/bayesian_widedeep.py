# type: ignore
from typing import Dict, Optional, Any, List, Callable
import torch
import numpy as np
import warnings

from baselines.base import BaselineModel, BaselineResult

try:
    from pytorch_widedeep.bayesian_models import BayesianTabMlp
    from pytorch_widedeep.training import BayesianTrainer
    from pytorch_widedeep.preprocessing import TabPreprocessor
    from pytorch_widedeep.metrics import R2Score
    from pytorch_widedeep.callbacks import EarlyStopping, Callback
    WIDEDEEP_AVAILABLE = True
except ImportError:
    WIDEDEEP_AVAILABLE = False
    Callback = object
    warnings.warn(
        "pytorch-widedeep not installed. Install: pip install pytorch-widedeep"
    )


class _HistorySaver(Callback):
    """Callback that saves history to external dict after each epoch."""
    
    def __init__(self, history_dict: Dict[str, List[float]], on_epoch_callback: Callable = None):
        super().__init__()
        self.history_dict = history_dict
        self.on_epoch_callback = on_epoch_callback
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None):
        logs = logs or {}
        for key in ["train_loss", "train_r2", "val_loss", "val_r2"]:
            if key in logs:
                self.history_dict.setdefault(key, []).append(logs[key])
        if self.on_epoch_callback:
            self.on_epoch_callback(epoch, self.history_dict)


class BayesianWideDeep(BaselineModel):

    def __init__(
        self,
        mlp_hidden_dims: list = None,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
        n_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.005,
        n_samples: int = 20,
        patience: int = 20,
        track_metrics: bool = True,
        seed: int = 0,
    ):
        if not WIDEDEEP_AVAILABLE:
            raise ImportError(
                "pytorch-widedeep not installed. Install: pip install pytorch-widedeep"
            )

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.mlp_hidden_dims = mlp_hidden_dims or [128, 64]
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_samples = n_samples
        self.patience = patience
        self.track_metrics = track_metrics

        self.model = None
        self.trainer = None
        self.tab_preprocessor = None
        self._n_features: int = 0
        self._singular: bool = False
        self._history: Dict[str, List[float]] = {}

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
        continuous_cols = [f"col_{i}" for i in range(self._n_features)]

        import pandas as pd
        df_train = pd.DataFrame(X, columns=continuous_cols)
        df_train["target"] = y

        self.tab_preprocessor = TabPreprocessor(
            continuous_cols=continuous_cols,
            scale=True,
        )
        X_tab = self.tab_preprocessor.fit_transform(df_train)

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        try:
            self.model = BayesianTabMlp(
                column_idx=self.tab_preprocessor.column_idx,
                continuous_cols=continuous_cols,
                mlp_hidden_dims=self.mlp_hidden_dims,
                prior_sigma_1=self.prior_sigma_1,
                prior_sigma_2=self.prior_sigma_2,
                prior_pi=self.prior_pi,
                posterior_mu_init=self.posterior_mu_init,
                posterior_rho_init=self.posterior_rho_init,
                pred_dim=1,
            )

            # Set up callbacks for tracking and early stopping
            callbacks = [_HistorySaver(self._history)]
            if self.track_metrics and X_val is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_r2",
                        mode="max",
                        patience=self.patience,
                        restore_best_weights=True,
                    )
                )

            # Set up metrics (R2 for regression)
            metrics = [R2Score()] if self.track_metrics else None

            self.trainer = BayesianTrainer(
                model=self.model,
                objective="regression",
                optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
                callbacks=callbacks if callbacks else None,
                metrics=metrics,
                verbose=1 if verbose else 0,
            )

            eval_kwargs = {}
            if X_val is not None and y_val is not None:
                X_v = X_val.cpu().numpy().astype(np.float32)
                y_v = y_val.cpu().numpy().astype(np.float32)
                if y_v.ndim == 1:
                    y_v = y_v.reshape(-1, 1)
                df_val = pd.DataFrame(X_v, columns=continuous_cols)
                df_val["target"] = y_v
                X_tab_val = self.tab_preprocessor.transform(df_val)
                eval_kwargs["X_tab_val"] = X_tab_val
                eval_kwargs["target_val"] = y_v

            self.trainer.fit(
                X_tab=X_tab,
                target=y,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                **eval_kwargs,
            )
        except ValueError as e:
            if "within the support" in str(e):
                self._singular = True
                raise RuntimeError(f"NaN in Bayesian weights (gradient explosion): {e}")
            raise
        finally:
            torch.set_default_dtype(prev_dtype)

        return {
            "n_epochs": self.n_epochs,
            "history": self._history,
            "best_val_r2": self._get_best_val_r2(),
        }

    def _get_best_val_r2(self) -> Optional[float]:
        """Get best validation R2 from history."""
        if "val_r2" in self._history and self._history["val_r2"]:
            return max(self._history["val_r2"])
        return None

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self._singular:
            raise RuntimeError("Model is singular (NaN weights)")
        if self.model is None or self.trainer is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_np = X.cpu().numpy().astype(np.float32)
        continuous_cols = [f"col_{i}" for i in range(self._n_features)]

        import pandas as pd
        df_test = pd.DataFrame(X_np, columns=continuous_cols)
        X_tab = self.tab_preprocessor.transform(df_test)

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        try:
            preds = self.trainer.predict(
                X_tab=X_tab,
                return_samples=True,
                n_samples=self.n_samples,
            )
        finally:
            torch.set_default_dtype(prev_dtype)

        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)

        mean_tensor = torch.from_numpy(mean.astype(np.float64))
        std_tensor = torch.from_numpy(std.astype(np.float64))

        if mean_tensor.ndim == 1:
            mean_tensor = mean_tensor.unsqueeze(-1)
        if std_tensor.ndim == 1:
            std_tensor = std_tensor.unsqueeze(-1)

        return BaselineResult(
            mean=mean_tensor,
            std=std_tensor,
            extra={"method": "BayesianWideDeep", "n_samples": self.n_samples},
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
