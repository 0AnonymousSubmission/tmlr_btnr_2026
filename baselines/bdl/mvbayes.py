# type: ignore
from typing import Dict, Any
import torch
import numpy as np
import warnings

from baselines.base import BaselineModel, BaselineResult

try:
    import pyBASS as pb
    PYBASS_AVAILABLE = True
except ImportError:
    PYBASS_AVAILABLE = False
    warnings.warn(
        "pyBASS not installed. Install: pip install pybass-emu"
    )


class MvBayes(BaselineModel):

    def __init__(
        self,
        nmcmc: int = 10000,
        nburn: int = 9000,
        thin: int = 1,
        maxInt: int = 3,
        maxBasis: int = 50,
        seed: int = 0,
        **kwargs,
    ):
        if not PYBASS_AVAILABLE:
            raise ImportError(
                "pyBASS not installed. Install: pip install pybass-emu"
            )

        self.nmcmc = nmcmc
        self.nburn = nburn
        self.thin = thin
        self.maxInt = maxInt
        self.maxBasis = maxBasis
        self.seed = seed
        self.model = None

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X = X_train.cpu().numpy().astype('float64')
        y = y_train.cpu().numpy().astype('float64').ravel()

        self.model = pb.bass(
            X, y,
            nmcmc=self.nmcmc,
            nburn=self.nburn,
            thin=self.thin,
            maxInt=self.maxInt,
            maxBasis=self.maxBasis,
            verbose=verbose,
        )

        return {"nmcmc": self.nmcmc, "nburn": self.nburn}

    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_np = X.cpu().numpy().astype('float64')
        y_post = self.model.predict(X_np)

        y_mean = np.mean(y_post, axis=0)
        y_std = np.std(y_post, axis=0)

        return BaselineResult(
            mean=torch.tensor(y_mean, dtype=torch.float64).unsqueeze(-1),
            std=torch.tensor(y_std, dtype=torch.float64).unsqueeze(-1),
            extra={},
        )

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return int(np.mean(self.model.samples.nbasis)) * (self.model.data.p + 1)
