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
    u_norm = torch.linalg.norm(u, dim = -1)
    v_norm = torch.linalg.norm(v, dim = -1)
    dotproduct = (u * v).sum(dim = -1)
    cos = dotproduct / (u_norm * v_norm + eps)
    cos = cos.clamp(-1., 1.)
    return torch.acos(cos)
