from __future__ import annotations

import torch
from torch import nn


class PairwiseDistance(nn.Module):
    """Compute invariant pairwise distances for directed graph edges.

    Shapes:
        x: [N, D]
        edge_index: [2, E]
        output: [E, 1]

    Operation:
        relative_ij = x_i - x_j
        squared_distance_ij = sum_d relative_ij[d]^2

        if squared=True:
            return squared_distance_ij
        else:
            return sqrt(squared_distance_ij + eps)

    Complexity:
        O(ED), for E edges and D coordinate dimensions.
    """

    def __init__(self, squared: bool = False, eps: float = 1e-8) -> None:
        super().__init__()
        self.squared = squared
        self.eps = eps

        if self.eps < 0:
            raise ValueError(f"eps must be non-negative, got {self.eps}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return pairwise distances with shape [E, 1]."""

        src, dst = edge_index
        relative = x[src] - x[dst] # [E, D]
        squared_distance = relative.pow(2).sum(dim = -1, keepdim = True) # [E, 1]

        if self.squared:
            return squared_distance

        return torch.sqrt(squared_distance + self.eps) # [E, 1]
