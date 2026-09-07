import pytest
import torch
from torch_geometric.data import Batch, Data

from equimol.data.qm9 import build_radial_basis, prepare_qm9_batch, split_qm9


pytestmark = pytest.mark.data


def test_split_qm9_returns_deterministic_non_empty_splits():
    train_idx, val_idx, test_idx = split_qm9(100, train_frac=0.8, val_frac=0.1, seed=0)

    assert train_idx.numel() == 80
    assert val_idx.numel() == 10
    assert test_idx.numel() == 10

    train_idx_2, val_idx_2, test_idx_2 = split_qm9(100, train_frac=0.8, val_frac=0.1, seed=0)
    assert torch.equal(train_idx, train_idx_2)
    assert torch.equal(val_idx, val_idx_2)
    assert torch.equal(test_idx, test_idx_2)


def test_prepare_qm9_batch_builds_model_inputs():
    mol0 = Data(
        z=torch.tensor([6, 1]),
        pos=torch.randn(2, 3),
        y=torch.arange(19, dtype=torch.float32).reshape(1, 19),
    )
    mol1 = Data(
        z=torch.tensor([8, 9, 1]),
        pos=torch.randn(3, 3),
        y=(100 + torch.arange(19, dtype=torch.float32)).reshape(1, 19),
    )
    data = Batch.from_data_list([mol0, mol1])
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0)

    batch = prepare_qm9_batch(
        data,
        target="gap",
        radial_basis=radial_basis,
        num_atom_types=100,
    )

    assert batch.h.shape == torch.Size([5, 100])
    assert batch.h.dtype == torch.float32
    assert batch.x.shape == torch.Size([5, 3])
    assert batch.edge_index.shape == torch.Size([2, 8])
    assert batch.edge_attr.shape == torch.Size([8, 4])
    assert torch.equal(batch.batch, torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(batch.y, torch.tensor([4.0, 104.0]))
