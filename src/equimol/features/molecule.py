from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from equimol.adapters import MolecularGraphTensors
from equimol.geometry import distance

@dataclass(frozen=True)
class MolecularNodeFeatures:
    """Molecular node feature contract.

    Shapes:
        - node_ids: [N]
        - batch: [N] or None

    Notes:
        - node_ids are atomic numbers z.
        - This stores categorical ids, not neural embeddings.
    """

    node_ids: torch.Tensor
    batch: torch.Tensor | None = None


@dataclass(frozen=True)
class MolecularEdgeFeatures:
    """Molecular edge feature contract.

    Shapes:
        - edge_attr: [E, A] or None
        - edge_distance: [E, 1] or None
    """

    edge_attr: torch.Tensor | None = None
    edge_distance: torch.Tensor | None = None


def molecule_atom_features(molecule: MolecularGraphTensors) -> MolecularNodeFeatures:
    """Build molecule atom-level node feature ids.

    Shapes:
        - molecule.z: [N]
        - molecule.coordinates: [N, 3]
        - molecule.batch: [N] or None
        - output.node_ids: [N]
        - output.batch: [N] or None
    """
    coordinates = molecule.coordinates
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected molecule coordinates with shape [N, 3], "
            f"got {tuple(coordinates.shape)}."
        )

    num_nodes = coordinates.shape[0]
    device = coordinates.device

    node_ids = torch.as_tensor(molecule.z, dtype=torch.long, device=device)
    if node_ids.shape != (num_nodes,):
        raise ValueError(
            f"Expected z with shape [{num_nodes}], got {tuple(node_ids.shape)}."
        )

    batch = None
    if molecule.batch is not None:
        batch = torch.as_tensor(molecule.batch, dtype=torch.long, device=device)
        if batch.shape != (num_nodes,):
            raise ValueError(
                f"Expected batch with shape [{num_nodes}], got {tuple(batch.shape)}."
            )

    return MolecularNodeFeatures(node_ids=node_ids, batch=batch)


def molecule_edge_features(
    coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_attr: Optional[torch.Tensor] = None,
) -> MolecularEdgeFeatures:
    """Build invariant molecule edge features from coordinates and edges.

    Shapes:
        - coordinates: [N, 3]
        - edge_index: [2, E]
        - edge_attr: [E, A] or None
        - output.edge_distance: [E, 1]
    """
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates with shape [N, 3], instead got {tuple(coordinates.shape)}"
        )
    num_nodes = coordinates.shape[0]
    device = coordinates.device

    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}."
        )

    if edge_index.numel() > 0:
        if edge_index.min() < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if edge_index.max() >= num_nodes:
            raise ValueError(
                f"edge_index contains node index {int(edge_index.max())}, "
                f"but num_nodes={num_nodes}."
            )

    src, dst = edge_index
    edge_distance = distance(coordinates[src], coordinates[dst]).unsqueeze(-1)

    if edge_attr is not None:
        edge_attr = torch.as_tensor(edge_attr, dtype=coordinates.dtype, device=device)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.shape[0] != edge_index.shape[-1]:
            raise ValueError(
                f"Expected edge_attr first dimension to match E={edge_index.shape[1]}, "
                f"got {edge_attr.shape[0]}."
            )
        edge_attr = torch.cat([edge_attr, edge_distance], dim=-1)
    else:
        edge_attr = edge_distance

    return MolecularEdgeFeatures(
        edge_attr=edge_attr,
        edge_distance=edge_distance,
    )
