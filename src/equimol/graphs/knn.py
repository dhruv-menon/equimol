"""k-nearest-neighbor graph construction."""

from __future__ import annotations

from typing import Optional

import torch

from .utils import empty_edges


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
        ``edge_index`` with shape ``[2, E]``.

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
    return torch.cat(edges, dim=1) if edges else empty_edges(x.device)
