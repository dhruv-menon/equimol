import pytest
import torch
from torch_geometric.data import Batch, Data

from equimol.data import build_radial_basis
from equimol.data.md17 import prepare_md17_batch, split_md17


pytestmark = pytest.mark.data


def test_split_md17_returns_deterministic_non_empty_splits():
    train_idx, val_idx, test_idx = split_md17(100, train_frac=0.8, val_frac=0.1, seed=0)

    assert train_idx.numel() == 80
    assert val_idx.numel() == 10
    assert test_idx.numel() == 10

    train_idx_2, val_idx_2, test_idx_2 = split_md17(100, train_frac=0.8, val_frac=0.1, seed=0)
    assert torch.equal(train_idx, train_idx_2)
    assert torch.equal(val_idx, val_idx_2)
    assert torch.equal(test_idx, test_idx_2)


def test_prepare_md17_batch_builds_energy_force_inputs():
    frame0 = Data(
        z=torch.tensor([6, 1, 1]),
        pos=torch.randn(3, 3),
        energy=torch.tensor([-10.0]),
        force=torch.randn(3, 3),
    )
    frame1 = Data(
        z=torch.tensor([8, 1, 1]),
        pos=torch.randn(3, 3),
        energy=torch.tensor([-12.0]),
        force=torch.randn(3, 3),
    )
    data = Batch.from_data_list([frame0, frame1])
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0)

    batch = prepare_md17_batch(
        data,
        radial_basis=radial_basis,
        graph="fully_connected",
        num_atom_types=100,
    )

    assert batch.h.shape == torch.Size([6, 100])
    assert batch.x.shape == torch.Size([6, 3])
    assert batch.edge_index.shape == torch.Size([2, 12])
    assert batch.edge_attr.shape == torch.Size([12, 4])
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(batch.y, torch.tensor([-10.0, -12.0]))
    assert batch.force.shape == torch.Size([6, 3])
