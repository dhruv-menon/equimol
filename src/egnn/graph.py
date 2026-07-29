"""Graph construction helpers for molecular and protein EGNN examples.

Problem solved:
    Build directed edge lists ``edge_index`` with shape ``[2, E]`` from
    coordinates or node counts.

Why graph choice matters:
    Dense molecular graphs expose every pairwise interaction but cost ``O(N^2)``.
    Radius and kNN graphs reduce edges to local neighborhoods, which is usually
    necessary for proteins and larger molecular systems.

Common failure modes:
    Mixing nodes from different graphs in a batch, accidentally including
    self-edges, or changing the edge set under a transformation that should not
    affect distances.
"""

from __future__ import annotations

from typing import Optional

import torch


def _empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty(2, 0, dtype=torch.long, device=device)


def fully_connected_edges(
    num_nodes: int,
    *,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build directed dense edges within each graph.

    Args:
        num_nodes: Total number of nodes ``N``.
        batch: Optional graph id per node with shape ``[N]``.
        loop: Whether to include ``i -> i`` self-edges.
        device: Device for the returned tensor.

    Returns:
        ``edge_index`` with shape ``[2, E]``.
    """

    if batch is None:
        device = device or torch.device("cpu")
        nodes = torch.arange(num_nodes, device=device)
        row = nodes.repeat_interleave(num_nodes)
        col = nodes.repeat(num_nodes)
        if not loop:
            keep = row != col
            row, col = row[keep], col[keep]
        return torch.stack([row, col], dim=0)

    device = batch.device
    edges = []
    for graph_id in torch.unique(batch, sorted=True):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        n = nodes.numel()
        row = nodes.repeat_interleave(n)
        col = nodes.repeat(n)
        if not loop:
            keep = row != col
            row, col = row[keep], col[keep]
        edges.append(torch.stack([row, col], dim=0))
    return torch.cat(edges, dim=1) if edges else _empty_edges(device)


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


def knn_graph(
    x: torch.Tensor,
    k: int,
    *,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
) -> torch.Tensor:
    """Build directed k-nearest-neighbor edges within each graph.

    Args:
        x: Coordinates with shape ``[N, D]``.
        k: Number of outgoing neighbors per node before optional self-edge
            removal.
        batch: Optional graph id per node with shape ``[N]``.
        loop: Whether a node can choose itself as a neighbor.

    Returns:
        ``edge_index`` with shape ``[2, E]`` where edges point source ``i`` to
        neighbor/target ``j``.

    Complexity:
        ``O(N^2D)`` for this readable implementation.
    """

    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    edges = []
    for graph_id in torch.unique(batch, sorted=True):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        n = nodes.numel()
        if n == 0:
            continue
        local_k = min(k + (0 if loop else 1), n)
        dist = torch.cdist(x[nodes], x[nodes])
        nbr = dist.topk(local_k, largest=False).indices
        if not loop:
            nbr = nbr[nbr != torch.arange(n, device=x.device).unsqueeze(1)].view(n, -1)
            nbr = nbr[:, : min(k, max(n - 1, 0))]
        row = nodes.repeat_interleave(nbr.size(1))
        col = nodes[nbr.reshape(-1)]
        edges.append(torch.stack([row, col], dim=0))
    return torch.cat(edges, dim=1) if edges else _empty_edges(x.device)
