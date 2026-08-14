"""graph construction between sequential edges within a specified window"""
from __future__ import annotations
from typing import Optional

import torch

from .utils import empty_edges

def sequential_edges(
        num_nodes: int, 
        *,
        batch: Optional[torch.Tensor] = None, 
        window: int = 1,
        directed: bool = False,
        loop: bool = False,
        device: Optional[torch.device] = None,
        ) -> torch.Tensor:
    """Build a graph with sequential edge connectivity within a specified window.

    Args:
        num_nodes: Total number of nodes ``N``.
        batch: Optional graph id per node with shape ``N``.
        window: Number of neighboring nodes to be included.
        directed: Whether to consider edges as directed.
        loop: Whether to include ``i -> i`` self-edges.
        device: Device for the returned tensor.
    
    Returns:
        ``edge_index`` with shape ``[2, E]``.
    """
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")
    if num_nodes == 0:
        return empty_edges(device or batch.device if batch is not None else torch.device("cpu"))
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    if batch is not None:
        if batch.numel() != num_nodes:
            raise ValueError(f"number of elements in batch must be equal to num_nodes")

    if batch is None:
        device = device or torch.device("cpu")
        nodes = torch.arange(num_nodes, device = device)
        rows, cols = [], []

        if loop:
            rows.append(nodes)
            cols.append(nodes)
            
        for offset in range(1, window + 1):
            if offset >= nodes.numel(): 
                break

            src = nodes[: -offset]
            dst = nodes[offset: ]

            rows.append(src)
            cols.append(dst)

            if not directed:
                rows.append(dst)
                cols.append(src)

        if not rows:
            return empty_edges(device)

        row = torch.cat(rows)
        col = torch.cat(cols)
        edges = torch.stack([row, col], dim = 0)
        return edges

    device = batch.device
    batches = []
    for graph_id in torch.unique(batch, sorted = True):
        nodes = torch.nonzero(batch == graph_id, as_tuple = False).flatten()
        rows, cols = [], []

        if loop:
            rows.append(nodes)
            cols.append(nodes)

        for offset in range(1, window + 1):
            if offset >= nodes.numel():
                break

            src = nodes[: -offset]
            dst = nodes[offset : ]

            rows.append(src)
            cols.append(dst)

            if not directed:
                rows.append(dst)
                cols.append(src)

        if not rows: 
            continue 

        row = torch.cat(rows)
        col = torch.cat(cols)
        edge = torch.stack([row, col], dim = 0)
        batches.append(edge)
    return torch.cat(batches, dim = 1) if batches else empty_edges(device)