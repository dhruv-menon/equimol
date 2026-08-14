"""Fully connected graph construction."""

from __future__ import annotations
from typing import Optional

import torch

from .utils import empty_edges


def fully_connected_edges(
    num_nodes: int,
    *,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Build directed dense edges within each graph.

    Args:
        num_nodes: Total number of nodes N.
        batch: Optional graph id per node with shape [N]
        loop: Whether to include i -> i self-edges.
        device: Device for the returned tensor.

    Returns:
        edge_index with shape [2, E]
    """

    if batch is None:
        device = device or torch.device("cpu")
        nodes = torch.arange(num_nodes, device = device)

        row = nodes.repeat_interleave(num_nodes)
        col = nodes.repeat(num_nodes)
        if not loop:
            keep = row != col
            row, col = row[keep], col[keep]
        return torch.stack([row, col], dim = 0)

    device = batch.device
    edges = []
    for graph_id in torch.unique(batch, sorted = True):
        nodes = torch.nonzero(batch == graph_id, as_tuple = False).flatten()
        n = nodes.numel()
        row = nodes.repeat_interleave(n) # O(n)
        col = nodes.repeat(n) # O(n)
        if not loop:
            keep = row != col
            row, col = row[keep], col[keep]
        edges.append(torch.stack([row, col], dim = 0))
    return torch.cat(edges, dim = 1) if edges else empty_edges(device)
