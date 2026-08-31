"""
Sanity checks for BTNR tensor-network models.

Run via:  python run.py --sanity-check model=mpo2 [model.L=3 model.bond_dim=6 ...]

These checks build a model through the normal path (core.models.create_model)
and exercise the full BTN machinery on small synthetic data, verifying that a
(possibly newly-added) model behaves correctly with training, trimming, NT
blocks, input detachment and bond removal.
"""

from code_sanity_checks.checks import run_sanity_checks

__all__ = ["run_sanity_checks"]
