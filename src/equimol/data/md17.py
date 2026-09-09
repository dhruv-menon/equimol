from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.datasets import MD17

from equimol.data.types import GeometricBatch
from equimol.graphs import fully_connected_edges, knn_graph, radius_graph
from equimol.layers import GaussianRadialBasis


GraphType = Literal["fully_connected", "knn", "radius"]

MD17_MOLECULES = {
    "aspirin",
    "aspirin CCSD",
    "azobenzene",
    "benzene",
    "benzene CCSD(T)",
    "benzene FHI-aims",
    "revised aspirin",
    "revised azobenzene",
    "revised benzene",
    "revised ethanol",
    "revised malonaldehyde",
    "revised naphthalene",
    "revised paracetamol",
    "revised salicylic acid",
    "revised toluene",
    "revised uracil",
}


def load_md17(root: str, name: str, train: bool | None = None) -> MD17:
    return MD17(root=root, name=name, train=train)


def split_md17(
    num_items: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
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

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator)
    return perm[:train], perm[train: train + val], perm[train + val:]


def compute_energy_stats(dataset, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([dataset[int(i)].energy.reshape(-1)[0].float() for i in indices])
    return values.mean(), values.std(unbiased=False).clamp_min(1e-12)


def prepare_md17_batch(
    data: Batch,
    radial_basis: GaussianRadialBasis | None = None,
    graph: GraphType = "radius",
    num_atom_types: int = 100,
    radius: float | None = 5.0,
    k: int | None = None,
) -> GeometricBatch:
    if data is None:
        raise ValueError("data is None")

    z = data.z.long()
    x = data.pos.float()
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(z.numel(), dtype=torch.long, device=z.device)
    else:
        batch = batch.long()

    if z.numel() > 0 and int(z.max()) >= num_atom_types:
        raise ValueError(f"num_atom_types={num_atom_types} is too small for max z={int(z.max())}")
    h = F.one_hot(z, num_classes=num_atom_types).float()

    if graph == "fully_connected":
        edge_index = fully_connected_edges(num_nodes=z.numel(), batch=batch)
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

    edge_attr = radial_basis(x=x, edge_index=edge_index) if radial_basis is not None else None
    energy = data.energy.float().reshape(-1)
    force = data.force.float()

    return GeometricBatch(
        h=h,
        x=x,
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        y=energy,
        force=force,
        mask=None,
    )
