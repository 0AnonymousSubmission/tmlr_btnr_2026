from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

from utils.device_utils import DEVICE


@dataclass
class BaselineResult:
    mean: torch.Tensor
    std: torch.Tensor
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def lower(self) -> torch.Tensor:
        return self.mean - 2 * self.std

    @property
    def upper(self) -> torch.Tensor:
        return self.mean + 2 * self.std


class BaselineModel(ABC):
    dtype = torch.float64

    @abstractmethod
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor, return_std: bool = True) -> BaselineResult:
        pass

    @abstractmethod
    def get_num_parameters(self) -> int:
        pass

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        result = self.predict(X, return_std=True)
        y_pred = result.mean
        y_std = result.std

        if y_pred.shape != y.shape:
            y = y.view_as(y_pred)

        mse = torch.mean((y - y_pred) ** 2).item()

        ss_res = torch.sum((y - y_pred) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

        nll = self._compute_nll(y, y_pred, y_std)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2,
            "nll": nll,
        }

    def _compute_nll(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        y_std: torch.Tensor,
    ) -> float:
        var = y_std**2 + 1e-6
        nll = 0.5 * torch.mean(torch.log(2 * np.pi * var) + (y - y_pred) ** 2 / var)
        return nll.item()
