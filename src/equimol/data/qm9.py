from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

from equimol.data.types import GeometricBatch
from equimol.graphs import fully_connected_edges, knn_graph, radius_graph
from equimol.layers import GaussianRadialBasis

GraphType = Literal["fully_connected", "knn", "radius"]

QM9_TARGETS = {
    "mu":       0,
    "alpha":    1,
    "homo":     2,
    "lumo":     3,
    "gap":      4,
    "r2":       5,
    "zpve":     6,
    "u0":       7,
    "u":        8,
    "h":        9,
    "g":        10,
    "cv":       11,
    "u0_atom":  12,
    "u_atom":   13,
    "h_atom":   14,
    "g_atom":   15,
    "a":        16,
    "b":        17,
    "c":        18,
}


def load_qm9(root: str) -> QM9:
    return QM9(root)


def get_qm9_target_index(target: str | int = "gap") -> int:
    if isinstance(target, int):
        if 0 <= target < len(QM9_TARGETS):
            return target
        raise ValueError(f"QM9 target index must be in [0, {len(QM9_TARGETS) - 1}], got {target}")

    name = target.lower()
    if not name:
        raise ValueError("A valid target must be passed")
    if name not in QM9_TARGETS:
        raise ValueError(f"Passed target {target} is not supported")
    return QM9_TARGETS[name]


def split_qm9(
        num_items: int, 
        train_frac: float = 0.9, 
        val_frac: float = 0.05, 
        seed: int = 42
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if train_frac <= 0 or train_frac >= 1.:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if val_frac < 0 or val_frac >= 1.:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.:
        raise ValueError("train and val fractions must leave a non-empty test split")

    train = int(num_items * train_frac)
    val = int(num_items * val_frac)
    if train == 0:
        raise ValueError("train split is empty")
    if num_items - train - val == 0:
        raise ValueError("test split is empty")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator = g)

    train_idx = perm[: train]
    val_idx = perm[train: train + val]
    test_idx = perm[train + val: ]
    return train_idx, val_idx, test_idx


def compute_target_stats(dataset, indices: torch.Tensor, target: str | int = "gap") -> tuple[torch.Tensor, torch.Tensor]:
    target_idx = get_qm9_target_index(target)
    values = torch.stack([
        dataset[int(i)].y.reshape(-1)[target_idx].float()
        for i in indices
    ])
    return values.mean(), values.std(unbiased=False).clamp_min(1e-12)


def build_radial_basis(
        num_basis: int = 32, 
        cutoff: float = 10.0, 
        gamma: float | None = None, 
        eps: float = 1e-8
        ) -> GaussianRadialBasis:

    return GaussianRadialBasis(
        num_basis=num_basis,
        cutoff=cutoff,
        gamma=gamma,
        eps=eps,
    )


def prepare_qm9_batch(
        data: Batch,
        target: str | int = "gap",
        radial_basis: GaussianRadialBasis | None = None, 
        graph: GraphType = "fully_connected",
        num_atom_types: int = 100,
        radius: float | None = None,
        k: int | None = None,
        ) -> GeometricBatch:

    if data is None:
        raise ValueError("data is None")

    target_idx = get_qm9_target_index(target = target)
    z = data.z.long()
    x = data.pos.float()
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(z.numel(), dtype=torch.long, device=z.device)
    else:
        batch = batch.long()

    # ---- One hot encode z --> h ----
    if z.numel() > 0 and int(z.max()) >= num_atom_types:
        raise ValueError(f"num_atom_types={num_atom_types} is too small for max z={int(z.max())}")
    h = F.one_hot(z, num_classes = num_atom_types).float() # [N, num_atom_types]

    # ---- build edge_index ----
    if graph == "fully_connected":
        edge_index = fully_connected_edges(num_nodes = z.numel(), batch = batch)
    elif graph == "radius":
        if radius is None:
            raise ValueError("radius must be passed when graph='radius'")
        edge_index = radius_graph(x, radius=radius, batch=batch)
    elif graph == "knn":
        if k is None:
            raise ValueError("k must be passed when graph='knn'")
        edge_index = knn_graph(x, k=k, batch=batch)
    else:
        raise ValueError(f"Unsupported graph type: {graph}")

    # ----- build edge attributes -----
    edge_attr = None
    if radial_basis is not None:
        edge_attr = radial_basis(x = x, edge_index = edge_index)

    y = data.y
    if y.ndim == 1:
        y = y.reshape(1, -1)
    y = y[:, target_idx].float()

    return GeometricBatch(h = h,
                          x = x,
                          edge_index = edge_index,
                          batch = batch,
                          edge_attr = edge_attr,
                          y = y,
                          mask = None)    
