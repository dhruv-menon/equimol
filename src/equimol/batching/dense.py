from __future__ import annotations

import torch

def dense_to_sparse_nodes(
    h: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert padded dense node tensors to sparse node tensors

    Shapes:
        h: [B, N_max, F]
        x: [B, N_max, D]
        mask: [B, N_max]

    Returns:
        h_sparse: [N_total, F]
        x_sparse: [N_total, D]
        batch: [N_total]
        local_index: [N_total]

    Complexity:
        O(B * N_max), plus the cost of copying valid node features.
    """

    if h.ndim != 3:
        raise ValueError(f"Expected h with shape [B, N_max, F], got {tuple(h.shape)}")
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, N_max, D], got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape [B, N_max], got {tuple(mask.shape)}")
    if not (h.shape[:2] == x.shape[:2] == mask.shape):
        raise ValueError(f"Alignment mismatch between h, x and mask dimensions")

    valid = mask.bool() # [B, N_max]

    batch, local_index = valid.nonzero(as_tuple = True)

    h_sparse = h[valid] # [N_total, F] 
    x_sparse = x[valid] # [N_total, D]

    return h_sparse, x_sparse, batch, local_index


def sparse_to_dense_nodes(
    values: torch.Tensor,
    batch: torch.Tensor,
    local_index: torch.Tensor,
    *,
    num_graphs: int | None = None,
    max_nodes: int | None = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Convert sparse node values back to padded dense node tensors.

    Shapes:
        values: [N_total, F]
        batch: [N_total]
        local_index: [N_total]

    Returns:
        dense: [B, N_max, F]

    Complexity:
        O(B * N_max * F), dominated by dense output allocation.
    """

    if values.ndim != 2:
        raise ValueError(f"Expected values with shape [N_total, F], got {tuple(values.shape)}")
    if batch.ndim != 1:
        raise ValueError(f"Expected batch with shape [N_total], got {tuple(batch.shape)}")
    if local_index.ndim != 1:
        raise ValueError(f"Expected local_index with shape [N_total], got {tuple(local_index.shape)}")
    if not (values.shape[0] == batch.shape[0] == local_index.shape[0]):
        raise ValueError(f"Dimensional mismatch between values, batch & local_index")

    if values.shape[0] == 0:
        if num_graphs is None or max_nodes is None:
            raise ValueError("num_graphs and max_nodes are required for empty sparse inputs")
        return torch.full(
            (num_graphs, max_nodes, values.shape[-1]),
            fill_value = fill_value,
            dtype = values.dtype,
            device = values.device,
        )

    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1
    if max_nodes is None:
        max_nodes = int(local_index.max().item()) + 1

    dense = torch.full((num_graphs, max_nodes, values.shape[-1]), 
                       fill_value = fill_value, 
                       dtype = values.dtype,
                       device = values.device)
    dense[batch, local_index] = values
    return dense
