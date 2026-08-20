from __future__ import annotations

import torch

from equimol.geometry.distances import distance
from equimol.geometry.angles import bond_angle
from equimol.geometry.torsions import dihedral_angle


def _validate_backbone_coordinates(coordinates: torch.Tensor) -> None:
    if coordinates.ndim != 3:
        raise ValueError(
            f"Expected backbone coordinates with shape [R, 4, 3], "
            f"got {tuple(coordinates.shape)}."
        )
    if coordinates.shape[1:] != (4, 3):
        raise ValueError(
            f"Expected backbone coordinates with shape [R, 4, 3], "
            f"got {tuple(coordinates.shape)}."
        )


def backbone_bond_lengths(coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return standard protein backbone bond lengths.

    Shapes:
        - coordinates: [R, 4, 3]
        - n_ca: [R]
        - ca_c: [R]
        - c_o: [R]
        - c_n_next: [R - 1]
        - ca_ca_next: [R - 1]

    Atom order:
        - 0: N
        - 1: CA
        - 2: C
        - 3: O

    Complexity:
        - O(R)
    """
    _validate_backbone_coordinates(coordinates)

    n_coords = coordinates[:, 0, :]  # [R, 3]
    ca_coords = coordinates[:, 1, :]  # [R, 3]
    c_coords = coordinates[:, 2, :]  # [R, 3]
    o_coords = coordinates[:, 3, :]  # [R, 3]

    n_ca_distances = distance(n_coords, ca_coords)  # [R]
    ca_c_distances = distance(ca_coords, c_coords)  # [R]
    c_o_distances = distance(c_coords, o_coords)  # [R]
    c_n_next_distances = distance(c_coords[:-1], n_coords[1:])  # [R - 1]
    ca_ca_next_distances = distance(ca_coords[:-1], ca_coords[1:])  # [R - 1]

    return {
        "n_ca": n_ca_distances,
        "ca_c": ca_c_distances,
        "c_o": c_o_distances,
        "c_n_next": c_n_next_distances,
        "ca_ca_next": ca_ca_next_distances,
    }


def backbone_bond_angles(coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return standard protein backbone bending angles in radians.

    Shapes:
        - coordinates: [R, 4, 3]
        - n_ca_c: [R]
        - ca_c_o: [R]
        - ca_c_n_next: [R - 1]
        - c_n_next_ca_next: [R - 1]

    Complexity:
        - O(R)
    """
    _validate_backbone_coordinates(coordinates)

    n_coords = coordinates[:, 0, :]  # [R, 3]
    ca_coords = coordinates[:, 1, :]  # [R, 3]
    c_coords = coordinates[:, 2, :]  # [R, 3]
    o_coords = coordinates[:, 3, :]  # [R, 3]

    n_ca_c_angles = bond_angle(n_coords, ca_coords, c_coords)  # [R]
    ca_c_o_angles = bond_angle(ca_coords, c_coords, o_coords)  # [R]
    ca_c_n_next_angles = bond_angle(
        ca_coords[:-1],
        c_coords[:-1],
        n_coords[1:],
    )  # [R - 1]
    c_n_next_ca_next_angles = bond_angle(
        c_coords[:-1],
        n_coords[1:],
        ca_coords[1:],
    )  # [R - 1]

    return {
        "n_ca_c": n_ca_c_angles,
        "ca_c_o": ca_c_o_angles,
        "ca_c_n_next": ca_c_n_next_angles,
        "c_n_next_ca_next": c_n_next_ca_next_angles,
    }


def backbone_torsions(coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return standard protein backbone torsions as sin/cos pairs.

    Shapes:
        - coordinates: [R, 4, 3]
        - phi: [R - 1, 2]
        - psi: [R - 1, 2]
        - omega: [R - 1, 2]

    Convention:
        - phi_i: C_{i-1}, N_i, CA_i, C_i
        - psi_i: N_i, CA_i, C_i, N_{i+1}
        - omega_i: CA_i, C_i, N_{i+1}, CA_{i+1}
        - torsion output stores [sin(theta), cos(theta)]

    Complexity:
        - O(R)
    """
    _validate_backbone_coordinates(coordinates)

    n_coords = coordinates[:, 0, :]  # [R, 3]
    ca_coords = coordinates[:, 1, :]  # [R, 3]
    c_coords = coordinates[:, 2, :]  # [R, 3]

    phi_angle = dihedral_angle(
        c_coords[:-1],
        n_coords[1:],
        ca_coords[1:],
        c_coords[1:],
    )  # [R - 1]
    psi_angle = dihedral_angle(
        n_coords[:-1],
        ca_coords[:-1],
        c_coords[:-1],
        n_coords[1:],
    )  # [R - 1]
    omega_angle = dihedral_angle(
        ca_coords[:-1],
        c_coords[:-1],
        n_coords[1:],
        ca_coords[1:],
    )  # [R - 1]

    phi_i = torch.stack([torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
    psi_i = torch.stack([torch.sin(psi_angle), torch.cos(psi_angle)], dim=-1)
    omega_i = torch.stack([torch.sin(omega_angle), torch.cos(omega_angle)], dim=-1)

    return {
        "phi": phi_i,
        "psi": psi_i,
        "omega": omega_i,
    }


def backbone_geometry(coordinates: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """Return bond lengths, bending angles, and torsions for a backbone.

    Shapes:
        - coordinates: [R, 4, 3]
        - output["lengths"]: dictionary of length tensors
        - output["angles"]: dictionary of angle tensors
        - output["torsions"]: dictionary of sin/cos torsion tensors

    Complexity:
        - O(R)
    """
    _validate_backbone_coordinates(coordinates)

    return {
        "lengths": backbone_bond_lengths(coordinates),
        "angles": backbone_bond_angles(coordinates),
        "torsions": backbone_torsions(coordinates),
    }
