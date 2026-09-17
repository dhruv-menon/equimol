from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from equimol.adapters import MolecularGraphTensors
from equimol.geometry import (
    bond_angle_features_from_index,
    dihedral_features_from_index,
    distance,
)

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


@dataclass(frozen=True)
class MolecularGeometryFeatures:
    """Molecular bonded-geometry feature contract.

    Shapes:
        - bond_index: [2, B] or None
        - angle_index: [3, A] or None
        - torsion_index: [4, T] or None
        - bond_lengths: [B, 1] or None
        - angle_features: [A, 1] or None
        - torsion_features: [T, 2] or None

    Feature convention:
        - bond length: ||x_i - x_j||
        - angle: cos(theta_ijk)
        - torsion: [sin(phi_ijkl), cos(phi_ijkl)]

    Notes:
        - Index construction belongs outside the calculators.
        - For molecules, build indices from covalent topology.
        - For proteins, reuse the same geometry logic with backbone topology.
    """

    bond_index: torch.Tensor | None = None
    angle_index: torch.Tensor | None = None
    torsion_index: torch.Tensor | None = None
    bond_lengths: torch.Tensor | None = None
    angle_features: torch.Tensor | None = None
    torsion_features: torch.Tensor | None = None


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


def molecule_geometry_features(
    coordinates: torch.Tensor,
    *,
    bond_index: torch.Tensor | None = None,
    angle_index: torch.Tensor | None = None,
    torsion_index: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> MolecularGeometryFeatures:
    """Build bond length, angle, and torsion features from molecular topology."""
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates with shape [N, 3], got {tuple(coordinates.shape)}."
        )

    device = coordinates.device
    num_nodes = coordinates.shape[0]

    def as_index(index: torch.Tensor | None, width: int, name: str) -> torch.Tensor | None:
        if index is None:
            return None
        index = torch.as_tensor(index, dtype=torch.long, device=device)
        if index.ndim != 2 or index.shape[0] != width:
            raise ValueError(f"Expected {name} with shape [{width}, T], got {tuple(index.shape)}.")
        if index.numel() == 0:
            return index
        if index.min() < 0:
            raise ValueError(f"{name} cannot contain negative node indices.")
        if index.max() >= num_nodes:
            raise ValueError(
                f"{name} contains node index {int(index.max())}, "
                f"but coordinates has {num_nodes} nodes."
            )
        return index

    bond_index = as_index(bond_index, 2, "bond_index")
    angle_index = as_index(angle_index, 3, "angle_index")
    torsion_index = as_index(torsion_index, 4, "torsion_index")

    bond_lengths = None
    if bond_index is not None:
        src, dst = bond_index
        bond_lengths = distance(coordinates[src], coordinates[dst], eps=eps).unsqueeze(-1)

    angle_features = None
    if angle_index is not None:
        angle_features = bond_angle_features_from_index(coordinates, angle_index, eps=eps)

    torsion_features = None
    if torsion_index is not None:
        torsion_features = dihedral_features_from_index(coordinates, torsion_index, eps=eps)

    return MolecularGeometryFeatures(
        bond_index=bond_index,
        angle_index=angle_index,
        torsion_index=torsion_index,
        bond_lengths=bond_lengths,
        angle_features=angle_features,
        torsion_features=torsion_features,
    )
