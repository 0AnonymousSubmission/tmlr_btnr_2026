# type: ignore
import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class TR:
    """
    Tensor Ring (TR) model.

    Same formalism as MPO2 (open MPS / tensor train), but with a *periodic*
    boundary condition: an extra bond `b{L-1}` connects the last site back to
    site 0. Every site is therefore a rank-3 core `(b{i-1}, x{i}, b{i})` with
    bond indices taken modulo L, plus an optional output leg on `output_site`.

        MPS (open):   x0-[]-b0-[]-b1-...-[]-x{L-1}
        TR  (ring):   ...-b{L-1}-[x0]-b0-[x1]-b1-...-[x{L-1}]-b{L-1}-...  (closed loop)

    The Bayesian machinery (BTNBuilder / BTN / TNALS) is topology-agnostic:
    it iterates over `mu.ind_map` and `mu` tensors, so the periodic bond is
    handled automatically without any changes to the inference code.
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        """
        Args:
            L: Number of sites (ring cores)
            bond_dim: Bond dimension for every bond (including the periodic one)
            phys_dim: Physical (input) dimension for each site
            output_dim: Output dimension
            output_site: Which site carries the output leg (default: last site)
            init_strength: Base init scale (used only if use_tn_normalization=False)
            use_tn_normalization: Apply TN normalization after init
            tn_target_std: Target output std for normalization
            sample_inputs: Sample inputs for output-based normalization
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []
        for i in range(L):
            # Periodic boundary: bond entering site i is b{i-1 mod L},
            # bond leaving site i is b{i mod L}. With L sites there are L bonds
            # b0..b{L-1}; b{L-1} closes the ring (site L-1 -> site 0).
            left_bond = f"b{(i - 1) % L}"
            right_bond = f"b{i % L}"
            shape = (bond_dim, phys_dim, bond_dim)
            inds = (left_bond, f"x{i}", right_bond)

            if i == self.output_site and output_dim > 1:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
            tensors.append(tensor)

        self.tn = qt.TensorNetwork(tensors)

        if use_tn_normalization:
            from model.initialization import normalize_tn_output, normalize_tn_frobenius

            if sample_inputs is not None:
                normalize_tn_output(
                    self.tn,
                    sample_inputs,
                    output_dims=["out"],
                    batch_dim="s",
                    target_std=tn_target_std,
                )
            else:
                # L bonds now (ring has one extra bond vs open MPS)
                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]

        self.input_dims = [f"x{i}" for i in range(L)]

        self.output_dims = ["out"] if output_dim > 1 else []
