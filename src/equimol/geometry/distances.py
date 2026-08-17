from __future__ import annotations

import torch


def distance(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Euclidean distances between coordinate tensors.

    Shapes:
        - a: [..., D]
        - b: [..., D]
        - output: [...]

    Operation:
        - Computes ||a - b||

    Symmetry:
        - The output is invariant to global translation and rotation.

    Complexity:
        - O(MD), where M is the number of coordinate pairs"""

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Dimensional mismatch between tensors a: {a.shape[-1]} and b: {b.shape[-1]}")
    rel = a - b
    sq_dist = (rel * rel).sum(dim = -1)
    return torch.sqrt(sq_dist + eps)

def squared_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return squared Euclidean distances between coordinate tensors.

    Shapes:
        - a: [..., D]
        - b: [..., D]
        - output: [...]

    Operation:
        - Computes ||a - b||^2.

    Symmetry:
        - The output is invariant to global translation and rotation.

    Complexity:
        - O(MD), where M is the number of coordinate pairs"""

    if a.shape[-1] != b.shape[-1]:
            raise ValueError(f"Dimensional mismatch between tensors a: {a.shape[-1]} and b: {b.shape[-1]}")
    rel = a - b
    squared_dist = (rel * rel).sum(dim = -1)
    return squared_dist
