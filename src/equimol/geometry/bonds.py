from __future__ import annotations

from dataclasses import dataclass

import torch

from equimol.geometry.indexing import angle_triplets_from_edges, torsion_quartets_from_edges

RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}


@dataclass(frozen=True)
class MolecularTopology:
    bond_index: torch.Tensor
    angle_index: torch.Tensor
    torsion_index: torch.Tensor


def covalent_radii(z: torch.Tensor) -> torch.Tensor:
    """Return covalent radii in Angstrom for atomic numbers z."""
    if z.ndim != 1:
        raise ValueError(f"z must have shape [N], got {tuple(z.shape)}")
    if not torch.is_floating_point(z) and not torch.is_complex(z):
        z = z.long()
    else:
        raise TypeError(f"z must contain integer atomic numbers, got dtype {z.dtype}")

    max_z = max(RADII.keys())
    if z.numel() > 0 and (z.min() < 0 or z.max() > max_z):
        raise ValueError("z contains unsupported atomic numbers")

    radii_lookup = torch.zeros(max_z + 1, dtype=torch.float32, device=z.device)
    supported = torch.zeros(max_z + 1, dtype=torch.bool, device=z.device)

    for z_num, radius in RADII.items():
        radii_lookup[z_num] = radius
        supported[z_num] = True

    if z.numel() > 0 and not supported[z].all():
        raise ValueError("z contains unsupported atomic numbers")

    return radii_lookup[z]


def infer_covalent_bonds(
    z: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    scale: float = 1.2,
    directed: bool = True,
) -> torch.Tensor:
    """Infer covalent bonds from d_ij <= scale * (r_i + r_j)."""
    if z.ndim != 1:
        raise ValueError(f"z must have shape [N], got {tuple(z.shape)}")
    if not torch.is_floating_point(z) and not torch.is_complex(z):
        z = z.long()
    else:
        raise TypeError(f"z must contain integer atomic numbers, got dtype {z.dtype}")
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"coordinates must have shape [N, 3], got {tuple(coordinates.shape)}"
        )
    if coordinates.shape[0] != z.shape[0]:
        raise ValueError(
            f"z and coordinates must describe the same N; got {z.shape[0]} and {coordinates.shape[0]}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    radii = covalent_radii(z) # [N]
    if radii.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=coordinates.device)

    pair_distances = torch.cdist(coordinates.float(), coordinates.float())

    cutoff_ij = scale * (radii[:, None] + radii[None, :])
    mask = pair_distances <= cutoff_ij
    mask.fill_diagonal_(False)
    if not directed:
        mask = torch.triu(mask, diagonal=1)

    src, dst = torch.nonzero(mask, as_tuple=True)
    return torch.stack([src, dst], dim=0).long()

    

def molecular_topology_from_geometry(
    z: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    scale: float = 1.2,
    ordered: bool = False,
) -> MolecularTopology:
    """Infer molecular bond, angle, and torsion indices from one conformer."""
    bond_index = infer_covalent_bonds(
        z,
        coordinates,
        scale=scale,
        directed=True,
    )
    return MolecularTopology(
        bond_index=bond_index,
        angle_index=angle_triplets_from_edges(
            bond_index,
            num_nodes=z.numel(),
            ordered=ordered,
        ).T.contiguous(),
        torsion_index=torsion_quartets_from_edges(
            bond_index,
            num_nodes=z.numel(),
            ordered=ordered,
        ).T.contiguous(),
    )
