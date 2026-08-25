from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from equimol.adapters import ProteinAtomTensors, ProteinBackboneTensors
from equimol.geometry import distance


@dataclass(frozen=True)
class ProteinNodeFeatures:
    """Protein node feature contract.

    Shapes:
        - node_ids: [N]
        - residue_ids: [N] or None
        - node_mask: [N] or None
        - batch: [N] or None

    Notes:
        - This stores categorical ids and masks, not neural embeddings.
        - Models should own embedding layers.
    """

    node_ids: torch.Tensor
    residue_ids: torch.Tensor | None = None
    node_mask: torch.Tensor | None = None
    batch: torch.Tensor | None = None


@dataclass(frozen=True)
class ProteinEdgeFeatures:
    """Protein edge feature contract.

    Shapes:
        - edge_attr: [E, A] or None
        - edge_distance: [E, 1] or None
        - sequence_offset: [E, 1] or None
        - edge_type: [E] or None

    Notes:
        - This stores tensor features, not message-passing logic.
    """

    edge_attr: torch.Tensor | None = None
    edge_distance: torch.Tensor | None = None
    sequence_offset: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None


def protein_atom_features(atom_tensors: ProteinAtomTensors) -> ProteinNodeFeatures:
    """Build atom-level protein node feature ids.

    Shapes:
        - atom_tensors.coordinates: [N, 3]
        - atom_tensors.atom_types: [N]
        - atom_tensors.atom_to_residue: [N]
        - output.node_ids: [N]
        - output.residue_ids: [N] or None
        - output.node_mask: [N] or None
        - output.batch: [N] or None

    Expected behavior:
        - use atom_types as node ids
        - map residue_types to atoms when available
        - preserve atom_mask and batch
    """
    coordinates = atom_tensors.coordinates
    atom_types = atom_tensors.atom_types
    atom_to_residue = atom_tensors.atom_to_residue

    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected atom coordinates with shape [N, 3], got {tuple(coordinates.shape)}."
        )

    num_atoms = coordinates.shape[0]
    device = coordinates.device

    atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=device)
    if atom_types.shape != (num_atoms,):
        raise ValueError(
            f"Expected atom_types with shape [{num_atoms}], got {tuple(atom_types.shape)}."
        )

    atom_to_residue = torch.as_tensor(atom_to_residue, dtype=torch.long, device=device)
    if atom_to_residue.shape != (num_atoms,):
        raise ValueError(
            f"Expected atom_to_residue with shape [{num_atoms}], "
            f"got {tuple(atom_to_residue.shape)}."
        )

    residue_ids = None
    if atom_tensors.residue_types is not None:
        residue_types = torch.as_tensor(
            atom_tensors.residue_types,
            dtype=torch.long,
            device=device,
        )
        if atom_to_residue.numel() > 0:
            if atom_to_residue.min() < 0:
                raise ValueError("atom_to_residue cannot contain negative indices.")
            if atom_to_residue.max() >= residue_types.shape[0]:
                raise ValueError(
                    "atom_to_residue contains an index outside residue_types."
                )
        residue_ids = residue_types[atom_to_residue]

    node_mask = None
    if atom_tensors.atom_mask is not None:
        node_mask = torch.as_tensor(atom_tensors.atom_mask, dtype=torch.bool, device=device)
        if node_mask.shape != (num_atoms,):
            raise ValueError(
                f"Expected atom_mask with shape [{num_atoms}], got {tuple(node_mask.shape)}."
            )

    batch = None
    if atom_tensors.batch is not None:
        batch = torch.as_tensor(atom_tensors.batch, dtype=torch.long, device=device)
        if batch.shape != (num_atoms,):
            raise ValueError(
                f"Expected batch with shape [{num_atoms}], got {tuple(batch.shape)}."
            )

    return ProteinNodeFeatures(
        node_ids=atom_types,
        residue_ids=residue_ids,
        node_mask=node_mask,
        batch=batch,
    )

def protein_residue_features(backbone_tensors: ProteinBackboneTensors) -> ProteinNodeFeatures:
    """Build residue-level protein node feature ids.

    Shapes:
        - backbone_tensors.coordinates: [R, 4, 3]
        - backbone_tensors.residue_types: [R] or None
        - backbone_tensors.atom_mask: [R, 4] or None
        - output.node_ids: [R]
        - output.node_mask: [R] or None
        - output.batch: [R] or None

    Expected behavior:
        - use residue_types when available
        - create fallback residue ids when needed
        - reduce atom_mask to a residue-level mask when available
    """
    coordinates = backbone_tensors.coordinates
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 3):
        raise ValueError(
            f"Expected backbone coordinates with shape [R, 4, 3], "
            f"got {tuple(coordinates.shape)}."
        )

    num_residues = coordinates.shape[0]
    device = coordinates.device

    if backbone_tensors.residue_types is None:
        node_ids = torch.zeros(num_residues, dtype=torch.long, device=device)
    else:
        node_ids = torch.as_tensor(
            backbone_tensors.residue_types,
            dtype=torch.long,
            device=device,
        )
        if node_ids.shape != (num_residues,):
            raise ValueError(
                f"Expected residue_types with shape [{num_residues}], "
                f"got {tuple(node_ids.shape)}."
            )

    node_mask = None
    if backbone_tensors.atom_mask is not None:
        atom_mask = torch.as_tensor(
            backbone_tensors.atom_mask,
            dtype=torch.bool,
            device=device,
        )
        if atom_mask.shape != (num_residues, 4):
            raise ValueError(
                f"Expected atom_mask with shape [{num_residues}, 4], "
                f"got {tuple(atom_mask.shape)}."
            )
        node_mask = atom_mask.any(dim=1)

    batch = None
    if backbone_tensors.batch is not None:
        batch = torch.as_tensor(backbone_tensors.batch, dtype=torch.long, device=device)
        if batch.shape != (num_residues,):
            raise ValueError(
                f"Expected batch with shape [{num_residues}], got {tuple(batch.shape)}."
            )

    return ProteinNodeFeatures(
        node_ids=node_ids,
        residue_ids=None,
        node_mask=node_mask,
        batch=batch,
    )


def protein_edge_features(
    coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    residue_index: Optional[torch.Tensor] = None,
    edge_type: Optional[torch.Tensor] = None,
) -> ProteinEdgeFeatures:
    """Build invariant protein edge features from coordinates and edges.

    Shapes:
        - coordinates: [N, 3]
        - edge_index: [2, E]
        - residue_index: [N] or None
        - edge_type: [E] or None
        - output.edge_distance: [E, 1]
        - output.sequence_offset: [E, 1] or None
        - output.edge_type: [E] or None

    Expected behavior:
        - compute pairwise edge distances
        - compute absolute residue-index offsets when residue_index is provided
        - preserve optional edge_type ids
    """
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates with shape [N, 3], got {tuple(coordinates.shape)}."
        )

    num_nodes = coordinates.shape[0]
    device = coordinates.device

    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}.")

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

    sequence_offset = None
    if residue_index is not None:
        residue_index = torch.as_tensor(residue_index, dtype=torch.long, device=device)
        if residue_index.shape != (num_nodes,):
            raise ValueError(
                f"Expected residue_index with shape [{num_nodes}], "
                f"got {tuple(residue_index.shape)}."
            )
        sequence_offset = (residue_index[src] - residue_index[dst]).abs().unsqueeze(-1)
        sequence_offset = sequence_offset.to(dtype=coordinates.dtype)

    edge_type_ids = None
    if edge_type is not None:
        edge_type_ids = torch.as_tensor(edge_type, dtype=torch.long, device=device)
        if edge_type_ids.shape != (edge_index.shape[1],):
            raise ValueError(
                f"Expected edge_type with shape [{edge_index.shape[1]}], "
                f"got {tuple(edge_type_ids.shape)}."
            )

    edge_attr_parts = [edge_distance]
    if sequence_offset is not None:
        edge_attr_parts.append(sequence_offset)

    return ProteinEdgeFeatures(
        edge_attr=torch.cat(edge_attr_parts, dim=-1),
        edge_distance=edge_distance,
        sequence_offset=sequence_offset,
        edge_type=edge_type_ids,
    )
