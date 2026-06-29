from baselines.base import BaselineModel, BaselineResult
from baselines.gp import ExactGP, SparseGP
from baselines.bdl import MCDropoutNet, DeepEnsemble, BdeMile, HorseshoeBNN, BayesianTabNet, BayesianWideDeep, MvBayes

BASELINE_MODELS = {
    "ExactGP": ExactGP,
    "SparseGP": SparseGP,
    "MCDropout": MCDropoutNet,
    "DeepEnsemble": DeepEnsemble,
    "BdeMile": BdeMile,
    "HorseshoeBNN": HorseshoeBNN,
    "BayesianTabNet": BayesianTabNet,
    "BayesianWideDeep": BayesianWideDeep,
    "MvBayes": MvBayes,
}

__all__ = [
    "BayesianWideDeep",
    "BaselineModel",
    "BaselineResult",
    "ExactGP",
    "SparseGP",
    "MCDropoutNet",
    "DeepEnsemble",
    "BdeMile",
    "HorseshoeBNN",
    "BayesianTabNet",
    "MvBayes",
    "BASELINE_MODELS",
]
