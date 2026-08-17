from __future__ import annotations

import torch


def bond_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the angle ABC in radians
    Shapes:
        - a: [..., 3]
        - b: [..., 3]
        - c: [..., 3]
        - output: [...]

    Operation:
        - Computes the angle between vectors a - b and c - b.

    Symmetry:
        - The output is invariant to global translation and rotation.

    Complexity:
        - O(M), where M is the number of angle triplets"""

    if not (a.shape[-1] == b.shape[-1] == c.shape[-1] == 3):
        raise ValueError("The final coordinate dimension must be 3")

    u = a - b
    v = c - b
    cross = torch.linalg.cross(u, v, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1).clamp_min(eps)
    dot = (u * v).sum(dim=-1)

    return torch.atan2(cross_norm, dot)
