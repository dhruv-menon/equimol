from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ProteinBackboneTensors:
    """Tensor contract for protein backbone inputs

    Shapes:
        - coordinates: [R, 4, 3]
        - residue_types: [R] or None
        - residue_index: [R] or None
        - atom_mask: [R, 4] or None
        - batch: [R] or None

    Atom order:
        - 0: N
        - 1: CA
        - 2: C
        - 3: O
    """

    coordinates: torch.Tensor
    residue_types: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    atom_mask: torch.Tensor | None = None
    batch: torch.Tensor | None = None


@dataclass(frozen=True)
class ProteinAtomTensors:
    """Tensor contract for atom-major protein graph inputs

    Shapes:
        - coordinates: [N, 3]
        - atom_types: [N]
        - atom_to_residue: [N]
        - residue_types: [R] or None
        - residue_index: [R] or None
        - atom_mask: [N] or None
        - batch: [N] or None
    """

    coordinates: torch.Tensor
    atom_types: torch.Tensor
    atom_to_residue: torch.Tensor
    residue_types: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    atom_mask: torch.Tensor | None = None
    batch: torch.Tensor | None = None


class ProteinBackboneAdapter:
    """Normalize protein-like tensor data into equimol protein contracts

    Expected source fields:
        - coordinates: [R, 4, 3]
        - residue_types: [R] or None
        - residue_index: [R] or None
        - atom_mask: [R, 4] or None
        - batch: [R] or None

    Responsibilities:
        - validate residue-major backbone tensor shapes
        - normalize dtypes/devices where appropriate
        - create simple defaults where appropriate
        - wrap tensors in ProteinBackboneTensors
        - optionally convert residue-major tensors to atom-major tensors"""

    def to_backbone_tensors(self, protein: Any) -> ProteinBackboneTensors:
        """Return canonical residue-major backbone tensors.

        Shapes:
            - source coordinates: [R, 4, 3]
            - output.coordinates: [R, 4, 3]
            - output.atom_mask: [R, 4] or None
            - output.residue_types: [R] or None
            - output.residue_index: [R] or None
            - output.batch: [R] or None"""

        if not hasattr(protein, "get"):
            raise TypeError("ProteinBackboneAdapter currently expects a dict-like object.")

        coords = protein.get("coordinates")
        if coords is None:
            raise ValueError("The protein object must contain 'coordinates'.")
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords)

        if coords.ndim != 3:
            raise ValueError(
                f"Expected protein coordinates with shape [R, 4, 3], "
                f"got {tuple(coords.shape)}."
            )
        if coords.shape[-2] != 4 or coords.shape[-1] != 3:
            raise ValueError(
                f"Expected protein coordinates with shape [R, 4, 3], "
                f"got {tuple(coords.shape)}."
            )

        coords = coords.float()
        num_residues = coords.shape[0]
        device = coords.device

        mask = protein.get("atom_mask")
        if mask is None:
            mask = torch.ones((num_residues, 4), dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask.shape != (num_residues, 4):
                raise ValueError(
                    f"Expected atom_mask with shape [{num_residues}, 4], "
                    f"got {tuple(mask.shape)}."
                )

        residue_ids = protein.get("residue_types")
        if residue_ids is not None:
            residue_ids = torch.as_tensor(residue_ids, dtype=torch.long, device=device)
            if residue_ids.shape != (num_residues,):
                raise ValueError(
                    f"Expected residue_types with shape [{num_residues}], "
                    f"got {tuple(residue_ids.shape)}."
                )

        residue_idx = protein.get("residue_index")
        if residue_idx is None:
            residue_idx = torch.arange(num_residues, dtype=torch.long, device=device)
        else:
            residue_idx = torch.as_tensor(residue_idx, dtype=torch.long, device=device)
            if residue_idx.shape != (num_residues,):
                raise ValueError(
                    f"Expected residue_index with shape [{num_residues}], "
                    f"got {tuple(residue_idx.shape)}."
                )

        batch = protein.get("batch")
        if batch is None:
            batch = torch.zeros(num_residues, dtype=torch.long, device=device)
        else:
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            if batch.shape != (num_residues,):
                raise ValueError(
                    f"Expected batch with shape [{num_residues}], "
                    f"got {tuple(batch.shape)}."
                )

        return ProteinBackboneTensors(
            coordinates=coords,
            residue_types=residue_ids,
            residue_index=residue_idx,
            atom_mask=mask,
            batch=batch,
        )

    def to_atom_tensors(self, protein: Any) -> ProteinAtomTensors:
        """Return canonical atom-major tensors derived from backbone tensors.

        Shapes:
            - source coordinates: [R, 4, 3]
            - output.coordinates: [N, 3]
            - output.atom_types: [N]
            - output.atom_to_residue: [N]

        This is a tensor view conversion step. It should not parse external
        structure files.
        """
        backbone = self.to_backbone_tensors(protein)
        coords = backbone.coordinates
        num_residues = coords.shape[0]
        device = coords.device

        coords = coords.reshape(-1, 3)

        atom_types = torch.as_tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        atom_types = atom_types.repeat(num_residues)
        atom_to_residue = torch.arange(
            num_residues, dtype=torch.long, device=device
        ).repeat_interleave(4)

        return ProteinAtomTensors(
            coordinates=coords,
            atom_types=atom_types,
            atom_to_residue=atom_to_residue,
            residue_types=backbone.residue_types,
            residue_index=backbone.residue_index,
            atom_mask=backbone.atom_mask.reshape(-1),
            batch=backbone.batch.repeat_interleave(4),
        )
        
