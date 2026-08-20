from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MolecularGraphTensors:
    """Tensor contract for molecule-level EGNN inputs.

    Shapes:
        - z: [N]
        - coordinates: [N, 3]
        - edge_index: [2, E]
        - edge_attr: [E, A] or None
        - batch: [N]
    """

    z: torch.Tensor
    coordinates: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None = None
    batch: torch.Tensor | None = None


class MoleculeAdapter:
    """Convert molecule objects into sparse graph tensors.

    Expected role:
        - read atom identifiers
        - read 3D coordinates
        - optionally read bond edges and bond features
        - return tensors compatible with equimol layers/models
    """

    def __call__(self, molecule: Any) -> MolecularGraphTensors:
        if not hasattr(molecule, "get"):
            raise TypeError("MoleculeGraphTensors currently expects a dict-like object.")

        coords = molecule.get("coordinates")
        if coords is None:
            raise ValueError("The molecule object must contain 'coordinates'.")
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords)

        if coords.ndim != 2:
            raise ValueError(
                f"Expected molecule coordinates with shape [N, 3], "
                f"got {tuple(coords.shape)}."
            )
        if coords.shape[-1] != 3:
            raise ValueError(
                f"Expected molecule coordinates with shape [N, 3], "
                f"got {tuple(coords.shape)}."
            )

        coords = coords.float()
        num_nodes = int(coords.shape[0])
        device = coords.device

        z = molecule.get("z")
        if z is None:
            raise ValueError("The molecule object must contain 'z'.")
        z = torch.as_tensor(z, dtype=torch.long, device=device)

        if z.ndim != 1:
            raise ValueError(f"Expected z with shape [N], got {tuple(z.shape)}.")
        if z.shape[0] != num_nodes:
            raise ValueError(
                f"Expected z with shape [{num_nodes}], got {tuple(z.shape)}."
            )

        edge_index = molecule.get("edge_index")
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
            if edge_index.ndim != 2:
                raise ValueError(
                    f"Expected edge_index with shape [2, E], "
                    f"got {tuple(edge_index.shape)}."
                )
            if edge_index.shape[0] != 2:
                raise ValueError(
                    f"Expected edge_index with shape [2, E], "
                    f"got {tuple(edge_index.shape)}."
                )
            if edge_index.numel() > 0:
                if edge_index.min() < 0:
                    raise ValueError("edge_index cannot contain negative node indices.")
                if edge_index.max() >= num_nodes:
                    raise ValueError(
                        f"edge_index contains node index {int(edge_index.max())}, "
                        f"but num_nodes={num_nodes}."
                    )

        edge_attr = molecule.get("edge_attr")
        if edge_attr is not None:
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.float, device=device)
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"Expected edge_attr first dimension to match E={edge_index.shape[1]}, "
                    f"got {edge_attr.shape[0]}."
                )

        batch = molecule.get("batch")
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            if batch.ndim != 1:
                raise ValueError(f"Expected batch with shape [N], got {tuple(batch.shape)}.")
            if batch.shape[0] != num_nodes:
                raise ValueError(
                    f"Expected batch with shape [{num_nodes}], got {tuple(batch.shape)}."
                )

        return MolecularGraphTensors(
            z=z,
            coordinates=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
