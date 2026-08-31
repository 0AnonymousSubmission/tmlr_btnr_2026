# type: ignore
import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class MPO2:
    """
    Simple MPS with output dimension.

    Structure:
    - Standard MPS chain with one site containing the output dimension
    - Tags: Node{i} for each site
    - Physical indices: x{i} for each site
    """

    # Prior strength for bond precision initialization
    bond_prior_alpha = 5.0

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
            L: Number of sites
            bond_dim: Bond dimension for MPS
            phys_dim: Physical dimension for each site
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which site gets the output dimension (default: middle site)
            init_strength: Base initialization strength before normalization (default: 0.001)
                          Only used if use_tn_normalization=False
            use_tn_normalization: Apply TN normalization after initialization (default: True)
                                 This eliminates seed-dependent collapses by normalizing outputs
            tn_target_std: Target standard deviation for TN normalization (default: 0.1)
            sample_inputs: Sample TN inputs for normalization. If None and use_tn_normalization=True,
                          will use Frobenius norm normalization instead
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []
        for i in range(L):
            if i == 0:
                shape = (phys_dim, bond_dim)
                inds = (f"x{i}", f"b{i}")
            elif i == L - 1:
                shape = (bond_dim, phys_dim)
                inds = (f"b{i - 1}", f"x{i}")
            else:
                shape = (bond_dim, phys_dim, bond_dim)
                inds = (f"b{i - 1}", f"x{i}", f"b{i}")

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
                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]

        self.input_dims = [f"x{i}" for i in range(L)]

        self.output_dims = ["out"] if output_dim > 1 else []
