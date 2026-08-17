from __future__ import annotations

import torch


def dihedral_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the signed dihedral angle A-B-C-D in radians

    Shapes:
        - a: [..., 3]
        - b: [..., 3]
        - c: [..., 3]
        - d: [..., 3]
        - output: [...]

    Operation:
        - Computes the signed torsion around the central bond b-c

    Symmetry:
        - The output is invariant to global translation and rotation

    Complexity:
        - O(M), where M is the number of coordinate quartets
    """
    if not (a.shape[-1] == b.shape[-1] == c.shape[-1] == d.shape[-1] == 3):
        raise ValueError(
            "The final coordinate dimension must be 3; "
            f"got a={a.shape[-1]}, b={b.shape[-1]}, "
            f"c={c.shape[-1]}, d={d.shape[-1]}."
        )

    b0 = b - a
    b1 = c - b
    b2 = d - c

    n1 = torch.linalg.cross(b0, b1, dim = -1)
    n2 = torch.linalg.cross(b1, b2, dim = -1)

    n1 = n1 / torch.linalg.norm(n1, dim = -1, keepdim = True).clamp_min(eps)
    n2 = n2 / torch.linalg.norm(n2, dim = -1, keepdim = True).clamp_min(eps)
    b1 = b1 / torch.linalg.norm(b1, dim = -1, keepdim = True).clamp_min(eps)

    x = (n1 * n2).sum(dim = -1)
    y = (torch.linalg.cross(n1, n2, dim = -1) * b1).sum(dim = -1)

    return torch.atan2(y, x)
