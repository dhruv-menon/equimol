"""Radius graph construction."""

from __future__ import annotations

from typing import Optional

import torch

from .fully_connected import fully_connected_edges


def radius_graph(
    x: torch.Tensor,
    radius: float,
    *,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
) -> torch.Tensor:
    """Build directed edges for node pairs within a radius.

    Args:
        x: Coordinates with shape ``[N, D]``.
        radius: Distance cutoff.
        batch: Optional graph id per node with shape ``[N]``.
        loop: Whether to include self-edges.

    Returns:
        ``edge_index`` with shape ``[2, E]``.

    Complexity:
        This readable implementation computes dense distances, so it is
        ``O(N^2D)``. Production code should use spatial indexing or PyG kernels.
    """

    edge_index = fully_connected_edges(x.size(0), batch=batch, loop=loop, device=x.device)
    if edge_index.numel() == 0:
        return edge_index

    row, col = edge_index
    dist2 = ((x[row] - x[col]) ** 2).sum(dim=-1)
    return edge_index[:, dist2 <= radius * radius]
