from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProteinBackboneTopology:
    bond_index: torch.Tensor
    angle_index: torch.Tensor
    torsion_index: torch.Tensor


def backbone_atom_index(
    residue_index: torch.Tensor,
    atom_offset: int,
) -> torch.Tensor:
    """Map residue ids and backbone atom offsets to flattened atom ids.

    Convention:
        - per residue atom order is [N, CA, C, O]
        - flattened index = 4 * residue_index + atom_offset

    TODO:
        - validate atom_offset is one of 0, 1, 2, 3
        - validate residue_index is long/integer
        - return 4 * residue_index + atom_offset
    """
    raise NotImplementedError


def protein_backbone_bond_index(
    num_residues: int,
    *,
    directed: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build covalent backbone bonds for flattened [R, 4, 3] coordinates.

    Bonds:
        - intra-residue: N_i - CA_i, CA_i - C_i, C_i - O_i
        - peptide: C_i - N_{i+1}

    Shapes:
        - output: [2, B]

    TODO:
        - build residue ids 0..R-1
        - build intra-residue bonds
        - build peptide bonds for residues 0..R-2
        - duplicate reversed edges when directed=True
    """
    raise NotImplementedError


def protein_backbone_angle_index(
    num_residues: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build standard backbone angle triplets for flattened backbone atoms.

    Angles:
        - N_i - CA_i - C_i
        - CA_i - C_i - O_i
        - CA_i - C_i - N_{i+1}
        - C_i - N_{i+1} - CA_{i+1}

    Shape:
        - output: [3, A]

    TODO:
        - use backbone_atom_index for every atom id
        - skip cross-residue angles when num_residues < 2
        - return torch.long tensor on device
    """
    raise NotImplementedError


def protein_backbone_torsion_index(
    num_residues: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build phi/psi/omega torsion quartets for flattened backbone atoms.

    Torsions:
        - phi_i:   C_{i-1} - N_i - CA_i - C_i
        - psi_i:   N_i - CA_i - C_i - N_{i+1}
        - omega_i: CA_i - C_i - N_{i+1} - CA_{i+1}

    Shape:
        - output: [4, T]

    TODO:
        - phi exists for residues 1..R-1
        - psi and omega exist for residues 0..R-2
        - return torch.long tensor on device
    """
    raise NotImplementedError


def protein_backbone_topology(
    num_residues: int,
    *,
    device: torch.device | None = None,
) -> ProteinBackboneTopology:
    """Build bond, angle, and torsion topology for backbone coordinates."""
    # TODO:
    # - call protein_backbone_bond_index
    # - call protein_backbone_angle_index
    # - call protein_backbone_torsion_index
    # - return ProteinBackboneTopology
    raise NotImplementedError
