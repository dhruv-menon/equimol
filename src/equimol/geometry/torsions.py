from __future__ import annotations

import torch


def dihedral_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the signed dihedral angle A-B-C-D in radians."""
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


def _validate_indexed_torsions(
    coordinates: torch.Tensor,
    torsion_index: torch.Tensor,
) -> None:
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates with shape [N, 3], got {tuple(coordinates.shape)}."
        )
    if torsion_index.ndim != 2 or torsion_index.shape[0] != 4:
        raise ValueError(
            f"Expected torsion_index with shape [4, T], got {tuple(torsion_index.shape)}."
        )
    if torsion_index.dtype != torch.long:
        raise TypeError(f"torsion_index must have dtype torch.long, got {torsion_index.dtype}.")
    if torsion_index.numel() == 0:
        return
    if torsion_index.min() < 0:
        raise ValueError("torsion_index cannot contain negative node indices.")
    if torsion_index.max() >= coordinates.shape[0]:
        raise ValueError(
            f"torsion_index contains node index {int(torsion_index.max())}, "
            f"but coordinates has {coordinates.shape[0]} nodes."
        )


def dihedral_angles_from_index(
    coordinates: torch.Tensor,
    torsion_index: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return signed dihedral angles for torsion_index columns [i, j, k, l]."""
    _validate_indexed_torsions(coordinates, torsion_index)

    i, j, k, l = torsion_index
    return dihedral_angle(
        coordinates[i],
        coordinates[j],
        coordinates[k],
        coordinates[l],
        eps=eps,
    )


def dihedral_features_from_index(
    coordinates: torch.Tensor,
    torsion_index: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return [sin(phi), cos(phi)] torsion features with shape [T, 2]."""
    phi = dihedral_angles_from_index(coordinates, torsion_index, eps=eps)
    return torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
