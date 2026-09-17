from __future__ import annotations

import torch


def bond_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the angle ABC in radians."""
    if not (a.shape[-1] == b.shape[-1] == c.shape[-1] == 3):
        raise ValueError("The final coordinate dimension must be 3")

    u = a - b
    v = c - b
    cross = torch.linalg.cross(u, v, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1).clamp_min(eps)
    dot = (u * v).sum(dim=-1)

    return torch.atan2(cross_norm, dot)


def _validate_indexed_angles(
    coordinates: torch.Tensor,
    angle_index: torch.Tensor,
) -> None:
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates with shape [N, 3], got {tuple(coordinates.shape)}."
        )
    if angle_index.ndim != 2 or angle_index.shape[0] != 3:
        raise ValueError(
            f"Expected angle_index with shape [3, A], got {tuple(angle_index.shape)}."
        )
    if angle_index.dtype != torch.long:
        raise TypeError(f"angle_index must have dtype torch.long, got {angle_index.dtype}.")
    if angle_index.numel() == 0:
        return
    if angle_index.min() < 0:
        raise ValueError("angle_index cannot contain negative node indices.")
    if angle_index.max() >= coordinates.shape[0]:
        raise ValueError(
            f"angle_index contains node index {int(angle_index.max())}, "
            f"but coordinates has {coordinates.shape[0]} nodes."
        )


def bond_angles_from_index(
    coordinates: torch.Tensor,
    angle_index: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return angles for angle_index columns [i, j, k], centered at j."""
    _validate_indexed_angles(coordinates, angle_index)

    i, j, k = angle_index
    return bond_angle(coordinates[i], coordinates[j], coordinates[k], eps=eps)


def bond_angle_features_from_index(
    coordinates: torch.Tensor,
    angle_index: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return cos(theta) angle features with shape [A, 1]."""
    theta = bond_angles_from_index(coordinates, angle_index, eps=eps)
    return torch.cos(theta).unsqueeze(-1)
