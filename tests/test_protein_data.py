import torch

from equimol.data import ProteinBackboneBatchConfig
from equimol.data import build_radial_basis
from equimol.data import prepare_protein_backbone_batch


def test_prepare_protein_backbone_batch_builds_atom_major_graph():
    coordinates = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    data = {
        "coordinates": coordinates,
        "atom_mask": torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ]
        ),
        "residue_types": torch.tensor([1, 2, 3, 4]),
        "batch": torch.tensor([0, 0, 1, 1]),
    }

    batch = prepare_protein_backbone_batch(
        data,
        config=ProteinBackboneBatchConfig(
            graph="backbone_radius",
            radius=100.0,
            use_residue_types=True,
            use_residue_index=True,
        ),
        radial_basis=build_radial_basis(num_basis=4, cutoff=100.0),
    )

    assert batch.h.shape == torch.Size([16, 25])
    assert batch.x.shape == torch.Size([16, 3])
    assert torch.equal(batch.x, coordinates.reshape(-1, 3))
    assert torch.equal(
        batch.batch,
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert torch.equal(
        batch.mask,
        torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ]
        ),
    )

    assert batch.edge_index.shape[0] == 2
    assert batch.edge_attr is not None
    assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
    assert batch.edge_attr.shape[1] == 6

    src_batch = batch.batch[batch.edge_index[0]]
    dst_batch = batch.batch[batch.edge_index[1]]
    assert torch.equal(src_batch, dst_batch)

    edge_codes = batch.edge_index[0] * batch.x.shape[0] + batch.edge_index[1]
    assert (2 * batch.x.shape[0] + 4) in edge_codes
    assert (6 * batch.x.shape[0] + 8) not in edge_codes
