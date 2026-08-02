from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GaussianRadialBasis(nn.Module):
    """Gaussian radial basis expansion for pairwise distances.

    Shapes:
        x: [N, D]
        edge_index: [2, E]
        output: [E, K]

    Operation:
        r_ij = ||x_i - x_j||
        phi_k(r_ij) = exp(-gamma * (r_ij - center_k)^2)

    Complexity:
        O(ED + EK), for E edges, D coordinate dimensions, and K basis functions.
    """

    def __init__(self,
                 num_basis: int = 32,
                 cutoff: float = 10.0,
                 gamma: Optional[float] = None,
                 eps: float = 1e-8,
                 ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.cutoff = cutoff
        self.eps = eps

        if num_basis <= 0:
            raise ValueError(f"Expected positive num_basis, got {num_basis}")
        if cutoff <= 0.: 
            raise ValueError(f"Expected a positive cutoff, instead got {cutoff}")
        if gamma is not None and gamma <= 0.:
            raise ValueError(f"Expected a positive gamma, instead got {gamma}")
        
        centers = torch.linspace(0, cutoff, steps = self.num_basis)
        self.register_buffer("centers", centers)

        if gamma is None:
            spacing = cutoff if num_basis == 1 else cutoff / (num_basis - 1)
            self.gamma = 1.0 / (spacing * spacing)
        else:
            self.gamma = gamma

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return radial edge features with shape [E, K]."""

        src, dst = edge_index
        relative = x[src] - x[dst] # [E, D]
        distance = torch.sqrt(relative.pow(2).sum(dim = -1, keepdim = True) + self.eps) # [E, 1]

        centers = self.centers.to(dtype = x.dtype)
        rbf = torch.exp(-self.gamma * (distance - centers).pow(2)) # [E, K]

        mask = distance > self.cutoff
        return rbf.masked_fill(mask, 0.0)
