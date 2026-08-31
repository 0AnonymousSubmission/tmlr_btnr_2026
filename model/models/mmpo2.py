# type: ignore
import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class MMPO2:
    """
    Masking MPO2: Non-trainable MPO mask, then trainable MPS.

    Structure:
    - MPO layer: input_dims × input_dims (NOT trainable, cumulative sum mask)
    - MPS layer: input_dims → output_dims (trainable)
    - Tags: {i}_Mask for MPO nodes (with NT tag), {i}_MPS for MPS nodes

    Mask is defined as: C^{i_in, i_out}_{b_left, b_right} = sum_k H_{b_left, k} * D_{k, i_in, i_out, b_right}
    where H_{ij} = theta(j-i) (Heaviside) and D is Kronecker delta
    """

    bond_prior_alpha = 5.0

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        rank: Optional[int] = None,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        """
        Args:
            L: Number of sites
            bond_dim: Bond dimension for MPS (NOT for mask MPO!)
            phys_dim: Input physical dimension (also mask MPO bond dimension)
            output_dim: Output dimension
            output_site: Which MPS site gets the output dimension (default: middle)
            init_strength: Initialization strength
            rank: Unused (for API compatibility with other models)
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.input_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        H = torch.zeros(phys_dim, phys_dim)
        for i in range(phys_dim):
            for j in range(phys_dim):
                H[i, j] = 1.0 if j >= i else 0.0

        mask_bond_dim = phys_dim
        self.mask_tensors = []

        for site_idx in range(L):
            if site_idx == 0:
                Delta = torch.zeros(phys_dim, phys_dim, mask_bond_dim)
                for k in range(phys_dim):
                    Delta[k, k, k] = 1.0
                data = Delta

                inds = (f"x{site_idx}", f"{site_idx}_masked", f"b_mask_{site_idx}")
                tags = {f"{site_idx}_Mask", "NT"}

            elif site_idx == L - 1:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k] = 1.0

                data = torch.einsum("bk,kio->bio", H, Delta)

                inds = (f"b_mask_{site_idx - 1}", f"x{site_idx}", f"{site_idx}_masked")
                tags = {f"{site_idx}_Mask", "NT"}

            else:
                Delta = torch.zeros(mask_bond_dim, phys_dim, phys_dim, mask_bond_dim)
                for k in range(mask_bond_dim):
                    Delta[k, k, k, k] = 1.0

                data = torch.einsum("bk,kior->bior", H, Delta)

                inds = (
                    f"b_mask_{site_idx - 1}",
                    f"x{site_idx}",
                    f"{site_idx}_masked",
                    f"b_mask_{site_idx}",
                )
                tags = {f"{site_idx}_Mask", "NT"}

            self.mask_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        self.mps_tensors = []
        for i in range(L):
            if i == 0:
                data = torch.randn(phys_dim, bond_dim) * base_init
                inds = (f"{i}_masked", f"b_mps_{i}")
                tags = {f"{i}_MPS"}
            elif i == L - 1:
                data = torch.randn(bond_dim, phys_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_masked")
                tags = {f"{i}_MPS"}
            else:
                data = torch.randn(bond_dim, phys_dim, bond_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_masked", f"b_mps_{i}")
                tags = {f"{i}_MPS"}

            self.mps_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        # Add output dimension to MPS output site
        output_tensor = self.mps_tensors[self.output_site]
        new_inds = list(output_tensor.inds) + ["out"]
        new_shape = output_tensor.shape + (output_dim,)
        output_tensor.modify(data=torch.randn(*new_shape) * base_init, inds=new_inds)

        self.tn = qt.TensorNetwork(self.mask_tensors + self.mps_tensors)

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
                import numpy as np

                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]

        self.input_dims = [f"x{i}" for i in range(L)]

        self.output_dims = ["out"]
