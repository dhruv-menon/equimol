from __future__ import annotations

from dataclasses import dataclass

import torch

# Per-residue backbone atom order in flattened [R * 4] atom indexing.
ATOM_N: int = 0
ATOM_CA: int = 1
ATOM_C: int = 2
ATOM_O: int = 3


@dataclass(frozen=True)
class ProteinBackboneTopology:
    """Backbone bonded topology for flattened [R, 4, 3] coordinates.

    Shapes:
        - bond_index: [2, B]
        - angle_index: [3, A]
        - torsion_index: [4, T]
    """

    bond_index: torch.Tensor
    angle_index: torch.Tensor
    torsion_index: torch.Tensor


def _validate_num_residues(num_residues: int) -> None:
    if num_residues < 0:
        raise ValueError(f"num_residues must be non-negative, got {num_residues}.")


def backbone_atom_index(
    residue_index: torch.Tensor,
    atom_offset: int,
) -> torch.Tensor:
    """Map residue ids and backbone atom offsets to flattened atom ids.

    Shapes:
        - residue_index: [...]
        - output: [...]

    Convention:
        - per residue atom order is [N, CA, C, O]
        - flattened index = 4 * residue_index + atom_offset
    """
    if atom_offset not in (ATOM_N, ATOM_CA, ATOM_C, ATOM_O):
        raise ValueError(f"atom_offset must be one of 0, 1, 2, 3, got {atom_offset}.")

    residue_index = torch.as_tensor(residue_index)
    if (
        torch.is_floating_point(residue_index)
        or torch.is_complex(residue_index)
        or residue_index.dtype == torch.bool
    ):
        raise TypeError(f"residue_index must be an integer tensor, got {residue_index.dtype}.")
    if residue_index.numel() > 0 and residue_index.min() < 0:
        raise ValueError("residue_index cannot contain negative residue ids.")

    return 4 * residue_index.long() + atom_offset


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
        - output: [2, B], with B = 4R - 1, doubled when directed=True

    Convention:
        - directed=True returns both i -> j and j -> i for every bond, matching
          infer_covalent_bonds and backbone_atom_bond_graph.

    Complexity:
        - O(R)
    """
    _validate_num_residues(num_residues)
    residues = torch.arange(num_residues, dtype=torch.long, device=device)  # [R]
    current = residues[:-1]  # [R - 1]
    following = residues[1:]  # [R - 1]

    src = torch.cat([
        backbone_atom_index(residues, ATOM_N),
        backbone_atom_index(residues, ATOM_CA),
        backbone_atom_index(residues, ATOM_C),
        backbone_atom_index(current, ATOM_C),
    ])
    dst = torch.cat([
        backbone_atom_index(residues, ATOM_CA),
        backbone_atom_index(residues, ATOM_C),
        backbone_atom_index(residues, ATOM_O),
        backbone_atom_index(following, ATOM_N),
    ])

    if directed:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])

    return torch.stack([src, dst], dim=0)


def protein_backbone_angle_index(
    num_residues: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build standard backbone angle triplets for flattened backbone atoms.

    Angles (grouped by type, in this order):
        - N_i - CA_i - C_i            for residues 0..R-1
        - CA_i - C_i - O_i            for residues 0..R-1
        - CA_i - C_i - N_{i+1}        for residues 0..R-2
        - C_i - N_{i+1} - CA_{i+1}    for residues 0..R-2

    Shape:
        - output: [3, A], with A = 2R + 2(R - 1) for R >= 1

    The grouping matches backbone_bond_angles, so bond_angles_from_index on
    flattened coordinates equals concatenating its outputs.

    Complexity:
        - O(R)
    """
    _validate_num_residues(num_residues)
    residues = torch.arange(num_residues, dtype=torch.long, device=device)  # [R]
    current = residues[:-1]  # [R - 1]
    following = residues[1:]  # [R - 1]

    angles = [
        (backbone_atom_index(residues, ATOM_N), backbone_atom_index(residues, ATOM_CA), backbone_atom_index(residues, ATOM_C)),
        (backbone_atom_index(residues, ATOM_CA), backbone_atom_index(residues, ATOM_C), backbone_atom_index(residues, ATOM_O)),
        (backbone_atom_index(current, ATOM_CA), backbone_atom_index(current, ATOM_C), backbone_atom_index(following, ATOM_N)),
        (backbone_atom_index(current, ATOM_C), backbone_atom_index(following, ATOM_N), backbone_atom_index(following, ATOM_CA)),
    ]
    return torch.stack([torch.cat(column) for column in zip(*angles)], dim=0)


def protein_backbone_torsion_index(
    num_residues: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build phi/psi/omega torsion quartets for flattened backbone atoms.

    Torsions (grouped by type, in this order):
        - phi_i:   C_{i-1} - N_i - CA_i - C_i          for residues 1..R-1
        - psi_i:   N_i - CA_i - C_i - N_{i+1}          for residues 0..R-2
        - omega_i: CA_i - C_i - N_{i+1} - CA_{i+1}     for residues 0..R-2

    Shape:
        - output: [4, T], with T = 3(R - 1) for R >= 1

    The grouping matches backbone_torsions, so dihedral_features_from_index on
    flattened coordinates equals concatenating its phi, psi, omega outputs.

    Complexity:
        - O(R)
    """
    _validate_num_residues(num_residues)
    residues = torch.arange(num_residues, dtype=torch.long, device=device)  # [R]
    current = residues[:-1]  # [R - 1]
    following = residues[1:]  # [R - 1]

    torsions = [
        (
            backbone_atom_index(current, ATOM_C),
            backbone_atom_index(following, ATOM_N),
            backbone_atom_index(following, ATOM_CA),
            backbone_atom_index(following, ATOM_C),
        ),
        (
            backbone_atom_index(current, ATOM_N),
            backbone_atom_index(current, ATOM_CA),
            backbone_atom_index(current, ATOM_C),
            backbone_atom_index(following, ATOM_N),
        ),
        (
            backbone_atom_index(current, ATOM_CA),
            backbone_atom_index(current, ATOM_C),
            backbone_atom_index(following, ATOM_N),
            backbone_atom_index(following, ATOM_CA),
        ),
    ]
    return torch.stack([torch.cat(column) for column in zip(*torsions)], dim=0)


def protein_backbone_topology(
    num_residues: int,
    *,
    directed: bool = True,
    device: torch.device | None = None,
) -> ProteinBackboneTopology:
    """Build bond, angle, and torsion topology for backbone coordinates.

    Shapes:
        - output.bond_index: [2, B]
        - output.angle_index: [3, A]
        - output.torsion_index: [4, T]
    """
    return ProteinBackboneTopology(
        bond_index=protein_backbone_bond_index(num_residues, directed=directed, device=device),
        angle_index=protein_backbone_angle_index(num_residues, device=device),
        torsion_index=protein_backbone_torsion_index(num_residues, device=device),
    )
