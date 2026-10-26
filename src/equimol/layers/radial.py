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
    """Smooth envelope that takes pairwise distances to zero at r_cut (DimeNet).

    With d = r / r_cut:
        u(d) = 1 - (p+1)(p+2)/2 * d^p + p(p+2) * d^(p+1) - p(p+1)/2 * d^(p+2)   for d < 1
        u(d) = 0                                                                for d >= 1

    u(0) = 1, and u and its first two derivatives vanish at r_cut.
    """

    def __init__(self, r_cut: float, p: int = 6) -> None:
        super().__init__()
        if r_cut <= 0:
            raise ValueError(f"r_cut must be > 0, got {r_cut}")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        self.r_cut = r_cut
        self.p = p

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d = distance / self.r_cut
        outside = d >= 1
        # Keep the polynomial finite where it is masked out; otherwise its
        # overflow turns the gradient of the final where into NaN.
        d = torch.where(outside, 0.0, d)
        p = self.p
        u = (
            1
            - (p + 1) * (p + 2) / 2 * d**p
            + p * (p + 2) * d ** (p + 1)
            - p * (p + 1) / 2 * d ** (p + 2)
        )
        return torch.where(outside, 0.0, u)


class BesselBasis(nn.Module):
    """Bessel radial basis (DimeNet), the radial input of NequIP and MACE.

    Shapes:
        distance: [E] or [E, 1]
        output: [E, K]

    Operation, for n = 1..K:
        e_n(r) = sqrt(2 / r_cut) * sin(n pi r / r_cut) / r

    Properties:
        Every e_n is zero at r = r_cut, and e_n has n - 1 roots inside (0, r_cut).
        The e_n are orthonormal on [0, r_cut] with weight r^2.
        As r -> 0, e_n(r) -> sqrt(2 / r_cut) * n pi / r_cut (finite).
    Multiply by PolynomialCutoff to make the basis vanish smoothly beyond r_cut.
    """

    def __init__(self, r_cut: float, num_basis: int = 8) -> None:
        super().__init__()
        if r_cut <= 0:
            raise ValueError(f"r_cut must be positive, got {r_cut}")
        if num_basis < 1:
            raise ValueError(f"num_basis must be >= 1, got {num_basis}")
        self.r_cut = r_cut
        self.num_basis = num_basis
        # n, not n * pi / r_cut: integers are exact in float32, so the frequencies
        # can be formed in the input's dtype without inheriting float32 rounding.
        self.register_buffer("n", torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()))

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        r = distance.reshape(-1, 1)
        n = self.n.to(r.dtype)
        a = torch.pi * n / self.r_cut
        # sin(a r) / r = a * sinc(n r / r_cut), with torch's sinc(x) = sin(pi x) / (pi x):
        # finite at r = 0, with finite gradients.
        return (2 / self.r_cut) ** 0.5 * a * torch.sinc(n * r / self.r_cut)
