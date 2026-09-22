from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from equimol.adapters.protein import ProteinBackboneAdapter
from equimol.data.types import GeometricBatch
from equimol.graphs import backbone_atom_bond_graph
from equimol.graphs import radius_graph

ProteinGraphType = Literal["backbone", "radius", "backbone_radius"]


@dataclass(frozen=True)
class ProteinBackboneBatchConfig:
    """Configuration for atom-major protein backbone batch preparation."""

    graph: ProteinGraphType = "backbone_radius"
    radius: float = 8.0
    num_atom_types: int = 4
    num_residue_types: int = 20
    use_residue_types: bool = False
    use_residue_index: bool = True
    use_edge_type: bool = True
    use_radial_basis: bool = True


def prepare_protein_backbone_batch(
    data,
    config: ProteinBackboneBatchConfig | None = None,
    radial_basis = None,
) -> GeometricBatch:

    """Prepare a full-backbone protein diffusion batch.

    Input contract:
        data.coordinates:    [R, 4, 3]
        data.atom_mask:      [R, 4] or None
        data.residue_types:  [R] or None
        data.residue_index:  [R] or None
        data.batch:          [R] or None

    Atom order per residue:
        0 = N
        1 = CA
        2 = C
        3 = O

    Output contract:
        h:          [4R, F]
        x:          [4R, 3]
        edge_index: [2, E]
        edge_attr:  [E, A] or None
        batch:      [4R]
        mask:       [4R]

    Adapter step:
        X_backbone in R^{R x 4 x 3}
        X_atom = reshape(X_backbone, [4R, 3])

        atom_type_i = i mod 4
        atom_to_residue_i = floor(i / 4)
        batch_atom_i = batch_residue_{atom_to_residue_i}
        mask_atom_i = atom_mask.reshape([4R])

    Node features:
        h_i = concat(
            one_hot(atom_type_i),
            optional one_hot(residue_type_{atom_to_residue_i}),
            optional positional/residue-index feature
        )

    Backbone covalent graph:
        For residue r:
            N_r  <-> CA_r
            CA_r <-> C_r
            C_r  <-> O_r
        For adjacent residues:
            C_r  <-> N_{r+1}

    Radius graph:
        edge i -> j exists if:
            ||x_i - x_j||_2 <= radius
            batch_i == batch_j
            i != j

    Combined graph:
        edge_index = union(backbone_edges, radius_edges)

    Edge features:
        distance:
            d_ij = ||x_i - x_j||_2
            radial_ij = RBF(d_ij)

        optional edge type:
            backbone intra-residue bond
            peptide bond
            radius contact

        edge_attr_ij = concat(radial_ij, edge_type_ij)

    Diffusion usage:
        x_t, eps = q_sample_coordinates(x, t, schedule, batch=batch)
        eps_hat = model(h, x_t, t, edge_index, batch, edge_attr)
        loss = coordinate_noise_mse(eps_hat, eps, node_mask=mask)

    TODO:
        1. Use ProteinBackboneAdapter.to_atom_tensors(data).
        2. Build atom one-hot features.
        3. Add optional residue-type features.
        4. Add optional residue-index/position feature.
        5. Build backbone covalent edges.
        6. Build radius edges.
        7. Merge/deduplicate edges.
        8. Build radial and edge-type edge_attr.
        9. Return GeometricBatch.
    """
    config = config or ProteinBackboneBatchConfig()
    protein_atom_tensor = ProteinBackboneAdapter().to_atom_tensors(data)
    x = protein_atom_tensor.coordinates
    batch = protein_atom_tensor.batch
    mask = protein_atom_tensor.atom_mask
    num_nodes = x.shape[0]

    # Node features: atom type, optional residue type, optional residue index.
    atom_types = protein_atom_tensor.atom_types
    atom_one_hot = F.one_hot(atom_types, num_classes=config.num_atom_types).float()
    features = [atom_one_hot]

    atom_to_residue = protein_atom_tensor.atom_to_residue

    residue_types = protein_atom_tensor.residue_types
    if config.use_residue_types and residue_types is not None:
        residue_types_atom = residue_types[atom_to_residue]
        residue_one_hot = F.one_hot(
            residue_types_atom,
            num_classes=config.num_residue_types,
        ).float()
        features.append(residue_one_hot)

    residue_index = protein_atom_tensor.residue_index
    if config.use_residue_index and residue_index is not None:
        residue_index_atom = residue_index[atom_to_residue]
        pos = residue_index_atom.float()
        pos = pos / pos.max().clamp_min(1.0)
        features.append(pos.unsqueeze(-1))

    h = torch.cat(features, dim=-1)

    if config.graph not in ("backbone", "radius", "backbone_radius"):
        raise ValueError(f"Unsupported protein graph type: {config.graph}")

    # Build edges per protein to avoid covalent/radius links across graphs.
    edge_parts = []
    backbone_codes = []
    radius_codes = []
    for graph_id in batch.unique(sorted=True).tolist():
        atom_ids = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        if atom_ids.numel() == 0:
            continue

        if config.graph in ("backbone", "backbone_radius"):
            if atom_ids.numel() % 4 != 0:
                raise ValueError("Protein backbone atom count must be divisible by 4")
            local_backbone = backbone_atom_bond_graph(
                int(atom_ids.numel() // 4),
                directed=True,
                device=x.device,
            )
            global_backbone = atom_ids[local_backbone]
            edge_parts.append(global_backbone)
            backbone_codes.append(global_backbone[0] * num_nodes + global_backbone[1])

        if config.graph in ("radius", "backbone_radius"):
            local_radius = radius_graph(
                x[atom_ids],
                radius=config.radius,
                loop=False,
            )
            global_radius = atom_ids[local_radius]
            edge_parts.append(global_radius)
            radius_codes.append(global_radius[0] * num_nodes + global_radius[1])

    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long, device=x.device)
    )
    edge_index = torch.unique(edge_index, dim=1)

    edge_attr_parts = []
    if config.use_radial_basis and radial_basis is not None:
        edge_attr_parts.append(radial_basis(x=x, edge_index=edge_index))

    if config.use_edge_type:
        edge_codes = edge_index[0] * num_nodes + edge_index[1]
        is_backbone = torch.zeros_like(edge_codes, dtype=torch.bool)
        is_radius = torch.zeros_like(edge_codes, dtype=torch.bool)
        if backbone_codes:
            is_backbone = torch.isin(edge_codes, torch.cat(backbone_codes))
        if radius_codes:
            is_radius = torch.isin(edge_codes, torch.cat(radius_codes))
        edge_attr_parts.append(torch.stack([is_backbone, is_radius], dim=-1).float())

    edge_attr = torch.cat(edge_attr_parts, dim=-1) if edge_attr_parts else None

    return GeometricBatch(
        h=h,
        x=x,
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        mask=mask,
    )
