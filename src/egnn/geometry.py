"""Geometry utilities for equivariance tests and tutorials.

Problem solved:
    Apply rigid transformations to coordinates while keeping scalar node
    features untouched.

Shapes:
    x: ``[N, D]`` coordinates.
    translation: ``[D]`` vector.
    rotation: ``[D, D]`` orthogonal matrix.

Equivariance principle:
    EGNN coordinate updates depend on relative vectors ``x_i - x_j`` and
    invariant squared distances ``||x_i - x_j||^2``. Translations cancel in
    relative vectors, and rotations commute with weighted sums of relative
    vectors.

Complexity:
    Each transform is ``O(ND)``.

Common failure modes:
    Feeding absolute coordinates into node or edge MLPs breaks translation
    equivariance. Using non-scalar coordinate-dependent features can also break
    rotation equivariance unless they transform correctly.
"""

from __future__ import annotations

import torch


def translate(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Translate coordinates.

    Args:
        x: Coordinate tensor with shape ``[N, D]``.
        shift: Translation vector with shape ``[D]``.

    Returns:
        Tensor with shape ``[N, D]``.
    """

    return x + shift.to(device=x.device, dtype=x.dtype)


def rotate(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate row-vector coordinates by an orthogonal matrix.

    Args:
        x: Coordinate tensor with shape ``[N, D]``.
        rotation: Orthogonal matrix with shape ``[D, D]``.

    Returns:
        Tensor with shape ``[N, D]``.
    """

    return x @ rotation.to(device=x.device, dtype=x.dtype).T


def random_rotation(dim: int = 3, *, device=None, dtype=None) -> torch.Tensor:
    """Sample a random rotation/reflection matrix with shape ``[D, D]``.

    QR gives an orthogonal matrix. For equivariance tests, allowing determinant
    ``-1`` is fine because EGNNs are E(n)-equivariant, not only SO(n)-equivariant.
    """

    q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, dtype=dtype))
    return q


def squared_distances(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute invariant pairwise squared distances for directed edges.

    Args:
        x: Coordinate tensor with shape ``[N, D]``.
        edge_index: Long tensor with shape ``[2, E]``. The first row is source
            node indices and the second row is target node indices.

    Returns:
        Tensor with shape ``[E, 1]``.
    """

    row, col = edge_index
    rel = x[row] - x[col]
    return (rel * rel).sum(dim=-1, keepdim=True)
