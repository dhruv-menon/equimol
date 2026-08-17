from __future__ import annotations

from typing import Optional

import torch


def _validate_edge_index(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.ndim != 2:
        raise ValueError(
            f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}."
        )
    if edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index first dimension to be 2, got {edge_index.shape[0]}."
        )
    if edge_index.dtype != torch.long:
        raise TypeError(f"Expected edge_index dtype torch.long, got {edge_index.dtype}.")

    return edge_index[0], edge_index[1]


def _validate_num_nodes(edge_index: torch.Tensor, num_nodes: Optional[int]) -> None:
    if num_nodes is None or edge_index.numel() == 0:
        return
    if num_nodes < 0:
        raise ValueError(f"Expected num_nodes >= 0, got {num_nodes}.")
    if edge_index.min() < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if edge_index.max() >= num_nodes:
        raise ValueError(
            f"edge_index contains node index {int(edge_index.max())}, "
            f"but num_nodes={num_nodes}."
        )


def _empty_index(width: int, device: torch.device) -> torch.Tensor:
    return torch.empty((0, width), dtype=torch.long, device=device)


def _neighbors_by_center(src: torch.Tensor, dst: torch.Tensor) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {}

    for source, center in zip(src.tolist(), dst.tolist()):
        center_neighbors = neighbors.setdefault(center, [])
        if source not in center_neighbors:
            center_neighbors.append(source)

    return neighbors


def edge_pairs(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source and destination indices from a sparse edge index.

    Shapes:
        - edge_index: [2, E]
        - output: two tensors with shape [E]

    Complexity:
        - O(1)
    """
    return _validate_edge_index(edge_index)


def angle_triplets_from_edges(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    ordered: bool = False,
) -> torch.Tensor:
    """Build angle triplets centered at graph nodes.

    Shapes:
        - edge_index: [2, E]
        - output: [T, 3]

    Convention:
        - Each row [i, j, k] represents angle i-j-k centered at j.
        - Edges follow src -> dst as neighbor -> center.

    Complexity:
        - O(sum_j degree(j)^2)
    """
    src, dst = _validate_edge_index(edge_index)
    _validate_num_nodes(edge_index, num_nodes)

    neighbors_by_center = _neighbors_by_center(src, dst)
    triplets: list[list[int]] = []

    for center, neighbors in neighbors_by_center.items():
        for left in range(len(neighbors)):
            for right in range(left + 1, len(neighbors)):
                i = neighbors[left]
                k = neighbors[right]

                triplets.append([i, center, k])

                if ordered:
                    triplets.append([k, center, i])

    if not triplets:
        return _empty_index(3, edge_index.device)

    return torch.tensor(triplets, dtype=torch.long, device=edge_index.device)


def torsion_quartets_from_edges(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    ordered: bool = False,
) -> torch.Tensor:
    """Build torsion quartets from bonded graph paths.

    Shapes:
        - edge_index: [2, E]
        - output: [Q, 4]

    Convention:
        - Each row [i, j, k, l] represents torsion i-j-k-l around bond j-k.
        - Edges follow src -> dst as neighbor -> center.

    Complexity:
        - O(sum_(j,k) degree(j) degree(k)); O(E) for bounded-degree graphs.
    """
    src, dst = _validate_edge_index(edge_index)
    _validate_num_nodes(edge_index, num_nodes)

    neighbors_by_center = _neighbors_by_center(src, dst)
    quartets: list[list[int]] = []

    for j, k in zip(src.tolist(), dst.tolist()):
        if not ordered and j > k:
            continue

        left_neighbors = neighbors_by_center.get(j, [])
        right_neighbors = neighbors_by_center.get(k, [])

        for i in left_neighbors:
            if i == k:
                continue

            for l in right_neighbors:
                if l == j or len({i, j, k, l}) != 4:
                    continue

                quartets.append([i, j, k, l])

    if not quartets:
        return _empty_index(4, edge_index.device)

    return torch.tensor(quartets, dtype=torch.long, device=edge_index.device)
