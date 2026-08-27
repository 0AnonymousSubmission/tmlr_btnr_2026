"""Collective registry of all tensor-network models."""

from model.models.mpo2 import MPO2
from model.models.tr import TR
from model.models.cmpo2 import CMPO2
from model.models.lmpo2 import LMPO2
from model.models.mmpo2 import MMPO2
from model.models.btt import BTT
from model.models.cpd import CPD

MODELS = {
    "MPO2": MPO2,
    "LMPO2": LMPO2,
    "CMPO2": CMPO2,
    "MMPO2": MMPO2,
    "TR": TR,
    "BTT": BTT,
    "CPD": CPD,
}

__all__ = ["MPO2", "LMPO2", "CMPO2", "MMPO2", "TR", "BTT", "CPD", "MODELS"]
