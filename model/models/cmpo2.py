# type: ignore
import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class CMPO2:
    """
    Cross MPO2: Two MPS layers (pixels and patches) that cross-connect.

    Structure:
    - Pixel MPS (psi): processes pixel dimensions, contains output
    - Patch MPS (phi): processes patch dimensions
    - Tags: {i}_Pi for pixel MPS nodes, {i}_Pa for patch MPS nodes
    """

    bond_prior_alpha = 5.0

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim_pixels: int,
        phys_dim_patches: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.01,
    ):
        """
        Args:
            L: Number of sites
            bond_dim: Bond dimension for MPS
            phys_dim_pixels: Physical dimension for pixel MPS
            phys_dim_patches: Physical dimension for patch MPS
            output_dim: Output dimension (e.g., number of classes)
            output_site: Which site gets the output dimension (default: middle site)
            init_strength: Initialization strength for output dimension
        """
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim_pixels = phys_dim_pixels
        self.phys_dim_patches = phys_dim_patches
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        # Create pixel MPS
        self.psi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim_pixels)

        # Create patch MPS
        self.phi = qt.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim_patches)

        # Add output dimension to pixel MPS at output_site
        output_node = self.psi[f"I{self.output_site}"]
        output_node.new_ind(
            "out", size=output_dim, axis=-1, mode="random", rand_strength=init_strength
        )

        # Convert to torch tensors
        self.psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
        self.phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

        # Reindex physical dimensions
        self.psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
        self.phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

        # Replace default I{i} tags with unique tags to avoid conflict with input tensors
        for i in range(L):
            # Remove default I{i} tag and add custom tag
            psi_tensor = self.psi[f"I{i}"]
            psi_tensor.drop_tags(f"I{i}")
            psi_tensor.add_tag(f"{i}_Pi")

            phi_tensor = self.phi[f"I{i}"]
            phi_tensor.drop_tags(f"I{i}")
            phi_tensor.add_tag(f"{i}_Pa")

        # Combine into tensor network
        self.tn = self.psi & self.phi

        # Define input labels for builder
        self.input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]

        # Define input_dims for NTN (simple site labels)
        self.input_dims = [str(i) for i in range(L)]

        # Define output dimensions
        self.output_dims = ["out"]
