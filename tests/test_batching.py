import pytest
import torch

from equimol.batching import dense_to_sparse_nodes

pytestmark = pytest.mark.batching


@pytest.fixture()
def dense_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [0.0, 0.0]],
            [[3.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    x = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
        ]
    )
    return h, x, mask


def test_dense_to_sparse_nodes_shapes(dense_batch):
    h, x, mask = dense_batch

    h_sparse, x_sparse, batch, local_index = dense_to_sparse_nodes(h, x, mask)

    assert h_sparse.shape == torch.Size([3, 2])
    assert x_sparse.shape == torch.Size([3, 3])
    assert batch.shape == torch.Size([3])
    assert local_index.shape == torch.Size([3])


def test_dense_to_sparse_nodes_values_are_preserved(dense_batch):
    h, x, mask = dense_batch

    h_sparse, x_sparse, _, _ = dense_to_sparse_nodes(h, x, mask)

    expected_h = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    expected_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(h_sparse, expected_h)
    assert torch.equal(x_sparse, expected_x)


def test_dense_to_sparse_nodes_returns_batch_and_local_index(dense_batch):
    h, x, mask = dense_batch

    _, _, batch, local_index = dense_to_sparse_nodes(h, x, mask)

    assert torch.equal(batch, torch.tensor([0, 0, 1]))
    assert torch.equal(local_index, torch.tensor([0, 1, 0]))


def test_dense_to_sparse_nodes_removes_padding(dense_batch):
    h, x, mask = dense_batch

    h_sparse, x_sparse, _, _ = dense_to_sparse_nodes(h, x, mask)

    assert h_sparse.size(0) == int(mask.sum().item())
    assert x_sparse.size(0) == int(mask.sum().item())
    assert not torch.any(torch.all(h_sparse == 0.0, dim=-1))


def test_dense_to_sparse_nodes_reconstructs_dense_values(dense_batch):
    h, _, mask = dense_batch
    h_sparse, _, batch, local_index = dense_to_sparse_nodes(*dense_batch)

    reconstructed = torch.zeros_like(h)
    reconstructed[batch, local_index] = h_sparse

    valid = mask.bool()
    assert torch.equal(reconstructed[valid], h[valid])
    assert torch.count_nonzero(reconstructed[~valid]) == 0


def test_dense_to_sparse_nodes_accepts_bool_mask(dense_batch):
    h, x, mask = dense_batch

    h_sparse_int, x_sparse_int, batch_int, local_index_int = dense_to_sparse_nodes(h, x, mask)
    h_sparse_bool, x_sparse_bool, batch_bool, local_index_bool = dense_to_sparse_nodes(
        h,
        x,
        mask.bool(),
    )

    assert torch.equal(h_sparse_int, h_sparse_bool)
    assert torch.equal(x_sparse_int, x_sparse_bool)
    assert torch.equal(batch_int, batch_bool)
    assert torch.equal(local_index_int, local_index_bool)


def test_dense_to_sparse_nodes_handles_empty_batch():
    h = torch.zeros((2, 3, 4))
    x = torch.zeros((2, 3, 3))
    mask = torch.zeros((2, 3), dtype=torch.bool)

    h_sparse, x_sparse, batch, local_index = dense_to_sparse_nodes(h, x, mask)

    assert h_sparse.shape == torch.Size([0, 4])
    assert x_sparse.shape == torch.Size([0, 3])
    assert batch.shape == torch.Size([0])
    assert local_index.shape == torch.Size([0])


def test_dense_to_sparse_nodes_validates_ranks():
    h = torch.zeros((2, 3, 4))
    x = torch.zeros((2, 3, 3))
    mask = torch.ones((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="h with shape"):
        dense_to_sparse_nodes(h[0], x, mask)

    with pytest.raises(ValueError, match="x with shape"):
        dense_to_sparse_nodes(h, x[0], mask)

    with pytest.raises(ValueError, match="mask with shape"):
        dense_to_sparse_nodes(h, x, mask.unsqueeze(-1))


def test_dense_to_sparse_nodes_validates_leading_shapes():
    h = torch.zeros((2, 3, 4))
    x = torch.zeros((2, 4, 3))
    mask = torch.ones((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="Alignment mismatch"):
        dense_to_sparse_nodes(h, x, mask)
