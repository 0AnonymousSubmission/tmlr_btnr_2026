# type: ignore
import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class LMPO2:
    """
    Linear MPO2: MPO for dimensionality reduction, then MPS for output.

    Structure:
    - MPO layer: input_dims → reduced_dims (trainable)
    - MPS layer: reduced_dims → output_dims (trainable)
    - Tags: {i}_MPO for MPO nodes, {i}_MPS for MPS nodes

    The reduction factor can be specified as reduced_dim/input_dim.
    For example, input_dim=10, reduced_dim=5 gives 50% reduction.
    """

    bond_prior_alpha = 1.0

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        reduced_dim: Optional[int] = None,
        reduction_factor: Optional[float] = None,
        output_dim: int = 1,
        mpo_bond_dim: int = 1,
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
            bond_dim: Bond dimension
            phys_dim: Input physical dimension
            reduced_dim: Reduced dimension after MPO (explicit value)
            reduction_factor: Alternative to reduced_dim - fraction of phys_dim to keep (e.g., 0.5 for 50%)
            output_dim: Output dimension
            output_site: Which MPS site gets the output dimension (default: middle)
            init_strength: Initialization strength
            rank: Alias for reduced_dim (for consistency with grid search)

        Note: Specify either reduced_dim OR reduction_factor OR rank, not multiple.
        """
        if rank is not None:
            reduced_dim = rank

        if reduced_dim is not None:
            if reduction_factor is not None:
                raise ValueError("Specify either reduced_dim/rank OR reduction_factor, not both")
        elif reduction_factor is not None:
            reduced_dim = max(2, int(phys_dim * reduction_factor))
        else:
            raise ValueError("Must specify either reduced_dim, rank, or reduction_factor")

        self.L = L
        self.bond_dim = bond_dim
        self.mpo_bond_dim = mpo_bond_dim
        self.phys_dim = phys_dim
        self.input_dim = phys_dim
        self.reduced_dim = reduced_dim
        self.reduction_factor = reduced_dim / phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        self.mpo_tensors = []
        for i in range(L):
            if mpo_bond_dim == 1:
                data = torch.randn(phys_dim, reduced_dim) * base_init
                inds = (f"x{i}", f"{i}_reduced")
                tags = {f"{i}_MPO"}
            elif i == 0:
                data = torch.randn(phys_dim, reduced_dim, mpo_bond_dim) * base_init
                inds = (f"x{i}", f"{i}_reduced", f"b_mpo_{i}")
                tags = {f"{i}_MPO"}
            elif i == L - 1:
                data = torch.randn(mpo_bond_dim, phys_dim, reduced_dim) * base_init
                inds = (f"b_mpo_{i - 1}", f"x{i}", f"{i}_reduced")
                tags = {f"{i}_MPO"}
            else:
                data = torch.randn(mpo_bond_dim, phys_dim, reduced_dim, mpo_bond_dim) * base_init
                inds = (f"b_mpo_{i - 1}", f"x{i}", f"{i}_reduced", f"b_mpo_{i}")
                tags = {f"{i}_MPO"}

            self.mpo_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        # Create MPS that takes reduced dimensions as input
        self.mps_tensors = []
        for i in range(L):
            if i == 0:
                data = torch.randn(reduced_dim, bond_dim) * base_init
                inds = (f"{i}_reduced", f"b_mps_{i}")
                tags = {f"{i}_MPS"}
            elif i == L - 1:
                data = torch.randn(bond_dim, reduced_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_reduced")
                tags = {f"{i}_MPS"}
            else:
                data = torch.randn(bond_dim, reduced_dim, bond_dim) * base_init
                inds = (f"b_mps_{i - 1}", f"{i}_reduced", f"b_mps_{i}")
                tags = {f"{i}_MPS"}

            self.mps_tensors.append(qt.Tensor(data=data, inds=inds, tags=tags))

        if output_dim > 1:
            output_tensor = self.mps_tensors[self.output_site]
            new_inds = list(output_tensor.inds) + ["out"]
            new_shape = output_tensor.shape + (output_dim,)
            output_tensor.modify(data=torch.randn(*new_shape) * base_init, inds=new_inds)

        self.tn = qt.TensorNetwork(self.mpo_tensors + self.mps_tensors)

        if use_tn_normalization:
            from model.initialization import normalize_tn_output, normalize_tn_frobenius

            if sample_inputs is not None:
                normalize_tn_output(
                    self.tn,
                    sample_inputs,
                    output_dims=["out"] if output_dim > 1 else [],
                    batch_dim="s",
                    target_std=tn_target_std,
                )
            else:
                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]

        self.input_dims = [f"x{i}" for i in range(L)]

        self.output_dims = ["out"] if output_dim > 1 else []
