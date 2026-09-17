from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MolecularEdgeGeometryConfig:
    use_bond_features: bool = False
    use_angle_features: bool = False
    use_torsion_features: bool = False


def _validate_edge_index(edge_index: torch.Tensor) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.dtype != torch.long:
        raise TypeError(f"edge_index must have dtype torch.long, got {edge_index.dtype}.")


def _validate_index(name: str, index: torch.Tensor, width: int, edge_index: torch.Tensor) -> None:
    if index.ndim != 2 or index.shape[0] != width:
        raise ValueError(f"Expected {name} with shape [{width}, N], got {tuple(index.shape)}.")
    if index.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {index.dtype}.")
    if index.device != edge_index.device:
        raise ValueError(f"{name} must be on the same device as edge_index.")


def _edge_positions(edge_index: torch.Tensor) -> dict[tuple[int, int], list[int]]:
    positions: dict[tuple[int, int], list[int]] = {}
    for edge_id, (src, dst) in enumerate(edge_index.T.tolist()):
        positions.setdefault((src, dst), []).append(edge_id)
    return positions


def _accumulate_on_edges(
    edge_index: torch.Tensor,
    mapped_edges: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    _validate_edge_index(edge_index)
    if mapped_edges.ndim != 2 or mapped_edges.shape[0] != 2:
        raise ValueError(f"Expected mapped_edges with shape [2, M], got {tuple(mapped_edges.shape)}.")
    if values.ndim != 2 or values.shape[0] != mapped_edges.shape[1]:
        raise ValueError("values must have shape [M, D] matching mapped_edges.")
    if mapped_edges.device != edge_index.device or values.device != edge_index.device:
        raise ValueError("mapped_edges and values must be on the same device as edge_index.")

    out = torch.zeros(edge_index.shape[1], values.shape[1], dtype=values.dtype, device=edge_index.device)
    counts = torch.zeros(edge_index.shape[1], 1, dtype=values.dtype, device=edge_index.device)
    positions = _edge_positions(edge_index)

    for value_id, (src, dst) in enumerate(mapped_edges.T.tolist()):
        for key in ((src, dst), (dst, src)):
            for edge_id in positions.get(key, []):
                out[edge_id] += values[value_id]
                counts[edge_id] += 1

    return out / counts.clamp_min(1)


def edge_bond_features(
    edge_index: torch.Tensor,
    bond_index: torch.Tensor,
    bond_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return edge-aligned [is_bonded, bond_length] features."""
    _validate_edge_index(edge_index)
    _validate_index("bond_index", bond_index, 2, edge_index)
    if bond_lengths.ndim != 2 or bond_lengths.shape != (bond_index.shape[1], 1):
        raise ValueError(
            f"Expected bond_lengths with shape [{bond_index.shape[1]}, 1], "
            f"got {tuple(bond_lengths.shape)}."
        )
    if bond_lengths.device != edge_index.device:
        raise ValueError("bond_lengths must be on the same device as edge_index.")

    length_features = _accumulate_on_edges(edge_index, bond_index, bond_lengths)
    is_bonded = (length_features != 0).to(dtype=bond_lengths.dtype)
    return torch.cat([is_bonded, length_features], dim=-1)


def edge_angle_summary_features(
    edge_index: torch.Tensor,
    angle_index: torch.Tensor,
    angle_features: torch.Tensor,
) -> torch.Tensor:
    """Map angle i-j-k to edge i-k and average collisions."""
    _validate_edge_index(edge_index)
    _validate_index("angle_index", angle_index, 3, edge_index)
    if angle_features.ndim != 2 or angle_features.shape != (angle_index.shape[1], 1):
        raise ValueError(
            f"Expected angle_features with shape [{angle_index.shape[1]}, 1], "
            f"got {tuple(angle_features.shape)}."
        )
    if angle_features.device != edge_index.device:
        raise ValueError("angle_features must be on the same device as edge_index.")

    mapped_edges = torch.stack([angle_index[0], angle_index[2]], dim=0)
    return _accumulate_on_edges(edge_index, mapped_edges, angle_features)


def edge_torsion_summary_features(
    edge_index: torch.Tensor,
    torsion_index: torch.Tensor,
    torsion_features: torch.Tensor,
) -> torch.Tensor:
    """Map torsion i-j-k-l to central edge j-k and average collisions."""
    _validate_edge_index(edge_index)
    _validate_index("torsion_index", torsion_index, 4, edge_index)
    if torsion_features.ndim != 2 or torsion_features.shape != (torsion_index.shape[1], 2):
        raise ValueError(
            f"Expected torsion_features with shape [{torsion_index.shape[1]}, 2], "
            f"got {tuple(torsion_features.shape)}."
        )
    if torsion_features.device != edge_index.device:
        raise ValueError("torsion_features must be on the same device as edge_index.")

    mapped_edges = torch.stack([torsion_index[1], torsion_index[2]], dim=0)
    return _accumulate_on_edges(edge_index, mapped_edges, torsion_features)


def molecular_edge_geometry_features(
    edge_index: torch.Tensor,
    *,
    bond_index: torch.Tensor | None = None,
    bond_lengths: torch.Tensor | None = None,
    angle_index: torch.Tensor | None = None,
    angle_features: torch.Tensor | None = None,
    torsion_index: torch.Tensor | None = None,
    torsion_features: torch.Tensor | None = None,
    config: MolecularEdgeGeometryConfig = MolecularEdgeGeometryConfig(),
) -> torch.Tensor:
    """Build optional edge-aligned molecular geometry features."""
    _validate_edge_index(edge_index)
    parts: list[torch.Tensor] = []

    if config.use_bond_features:
        if bond_index is None or bond_lengths is None:
            raise ValueError("bond_index and bond_lengths are required for bond features.")
        parts.append(edge_bond_features(edge_index, bond_index, bond_lengths))

    if config.use_angle_features:
        if angle_index is None or angle_features is None:
            raise ValueError("angle_index and angle_features are required for angle features.")
        parts.append(edge_angle_summary_features(edge_index, angle_index, angle_features))

    if config.use_torsion_features:
        if torsion_index is None or torsion_features is None:
            raise ValueError("torsion_index and torsion_features are required for torsion features.")
        parts.append(edge_torsion_summary_features(edge_index, torsion_index, torsion_features))

    if not parts:
        return torch.empty(edge_index.shape[1], 0, device=edge_index.device)

    return torch.cat(parts, dim=-1)
