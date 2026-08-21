"""Protein-specific graph construction helpers."""

from __future__ import annotations

from typing import Optional

import torch


def backbone_atom_bond_graph(
    num_residues: int,
    *,
    directed: bool = True,
    loop: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build atom-level backbone bond edges for flattened N, CA, C, O atoms

    Shapes:
        - flattened atom nodes: [R * 4]
        - output edge_index: [2, E]

    Atom order per residue:
        - 0: N
        - 1: CA
        - 2: C
        - 3: O

    Intended edges:
        - N_i - CA_i
        - CA_i - C_i
        - C_i - O_i
        - C_i - N_{i+1}

    Complexity:
        - O(R)
    """
    _ATOM_TYPE_N: int = 0
    _ATOM_TYPE_CA: int = 1
    _ATOM_TYPE_C: int = 2
    _ATOM_TYPE_O: int = 3

    if num_residues < 0:
        raise ValueError(f"num_residues must be non-negative, got {num_residues}.")
    if num_residues == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    residues = torch.arange(num_residues, device=device, dtype=torch.long)  # [R]
    n_atoms = residues * 4 + _ATOM_TYPE_N  # [R]
    ca_atoms = residues * 4 + _ATOM_TYPE_CA  # [R]
    c_atoms = residues * 4 + _ATOM_TYPE_C  # [R]
    o_atoms = residues * 4 + _ATOM_TYPE_O  # [R]

    if directed:
        src = [
            n_atoms,
            ca_atoms,
            c_atoms,
            c_atoms[:-1],
            ca_atoms,
            c_atoms,
            o_atoms,
            n_atoms[1:],
        ]
        dst = [
            ca_atoms,
            c_atoms,
            o_atoms,
            n_atoms[1:],
            n_atoms,
            ca_atoms,
            c_atoms,
            c_atoms[:-1],
        ]
    else:
        src = [n_atoms, ca_atoms, c_atoms, c_atoms[:-1]]
        dst = [ca_atoms, c_atoms, o_atoms, n_atoms[1:]]

    row = torch.cat(src)
    col = torch.cat(dst)

    if loop:
        nodes = torch.arange(num_residues * 4, device=device, dtype=torch.long)
        row = torch.cat([row, nodes])
        col = torch.cat([col, nodes])

    return torch.stack([row, col], dim=0)


def residue_sequential_graph(
    num_residues: int,
    *,
    batch: Optional[torch.Tensor] = None,
    window: int = 1,
    directed: bool = False,
    loop: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build residue-level sequence-local edges.

    Shapes:
        - residue nodes: [R]
        - batch: [R] or None
        - output edge_index: [2, E]

    Complexity:
        - O(R * window)
    """
    raise NotImplementedError


def ca_radius_graph(
    coordinates: torch.Tensor,
    radius: float,
    *,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
) -> torch.Tensor:
    """Build residue-level spatial edges using CA coordinates.

    Shapes:
        - coordinates: [R, 4, 3]
        - batch: [R] or None
        - output edge_index: [2, E]

    Atom order:
        - CA is coordinates[:, 1, :]

    Complexity:
        - Uses the current radius graph backend.
    """
    raise NotImplementedError
