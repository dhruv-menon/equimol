from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from equimol.layers.distance import PairwiseDistance


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
        self.distance = PairwiseDistance(squared = False, eps = eps)

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

        distance = self.distance(x, edge_index) # [E, 1]
        centers = self.centers.to(dtype = x.dtype)
        rbf = torch.exp(-self.gamma * (distance - centers).pow(2)) # [E, K]

        mask = distance > self.cutoff
        return rbf.masked_fill(mask, 0.0)


class PolynomialCutoff(nn.Module):
    """Smooth envelope that takes pairwise distances to zero at a cutoff.

    Multiplying each message by u(r_ij) makes it fade out as atom j leaves the
    cutoff sphere of atom i, so the energy and forces stay continuous when an
    edge of a radius graph appears or disappears.

    Shapes:
        distance: any shape, e.g. [E] or [E, 1]
        output: same shape and dtype as distance

    Operation (DimeNet, Gasteiger et al. 2020), with d = r / r_cut:
        u(d) = 1 - (p+1)(p+2)/2 * d^p + p(p+2) * d^(p+1) - p(p+1)/2 * d^(p+2)   for d < 1
        u(d) = 0                                                                for d >= 1

    Properties:
        u(0) = 1, and u, du/dr and d2u/dr2 are all zero at r_cut.
        u is monotonically non-increasing on [0, r_cut], so 0 <= u <= 1.
    """

    def __init__(self, r_cut: float, p: int = 6) -> None:
        """
        Args:
            r_cut: Cutoff radius, in the same units as the distances.
            p: Polynomial order; larger p keeps u close to 1 for longer.

        Raises:
            ValueError: If r_cut <= 0 or p < 1.
        """
        super().__init__()
        # TODO 1: validate r_cut and p (see Raises), then store both.
        raise NotImplementedError

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Return the envelope value for each distance."""
        # TODO 2: scale the distances to d = r / r_cut.
        # TODO 3: evaluate the polynomial in d.
        # TODO 4: set the envelope to zero for d >= 1. The polynomial itself grows
        #   again past d = 1, so this step is not optional. The value and the
        #   gradient must both be exactly zero there, and finite even when the
        #   polynomial overflows for very large d: multiplying by a 0/1 mask
        #   gives 0 * inf = NaN. torch.where and masked_fill have the same trap
        #   in the backward pass, because the gradient also flows through the
        #   branch that was not selected. Make sure the polynomial never sees
        #   a large d in the first place.
        raise NotImplementedError
