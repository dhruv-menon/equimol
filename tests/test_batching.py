import pytest
import torch

from equimol.batching import dense_to_sparse_nodes
from equimol.batching import sparse_to_dense_nodes

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


def test_sparse_to_dense_nodes_reconstructs_basic_values():
    values = torch.tensor(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
    )
    batch = torch.tensor([0, 0, 1])
    local_index = torch.tensor([0, 1, 0])

    dense = sparse_to_dense_nodes(values, batch, local_index)

    expected = torch.tensor(
        [
            [[10.0, 100.0], [20.0, 200.0]],
            [[30.0, 300.0], [0.0, 0.0]],
        ]
    )
    assert torch.equal(dense, expected)


def test_sparse_to_dense_nodes_round_trips_dense_to_sparse(dense_batch):
    h, x, mask = dense_batch
    h_sparse, _, batch, local_index = dense_to_sparse_nodes(h, x, mask)

    reconstructed = sparse_to_dense_nodes(
        h_sparse,
        batch,
        local_index,
        num_graphs=h.size(0),
        max_nodes=h.size(1),
    )

    valid = mask.bool()
    assert torch.equal(reconstructed[valid], h[valid])
    assert torch.count_nonzero(reconstructed[~valid]) == 0


def test_sparse_to_dense_nodes_supports_custom_fill_value():
    values = torch.tensor([[1.0], [2.0]])
    batch = torch.tensor([0, 1])
    local_index = torch.tensor([0, 0])

    dense = sparse_to_dense_nodes(
        values,
        batch,
        local_index,
        num_graphs=2,
        max_nodes=3,
        fill_value=-1.0,
    )

    expected = torch.tensor(
        [
            [[1.0], [-1.0], [-1.0]],
            [[2.0], [-1.0], [-1.0]],
        ]
    )
    assert torch.equal(dense, expected)


def test_sparse_to_dense_nodes_uses_explicit_shape():
    values = torch.tensor([[1.0, 2.0]])
    batch = torch.tensor([0])
    local_index = torch.tensor([0])

    dense = sparse_to_dense_nodes(
        values,
        batch,
        local_index,
        num_graphs=3,
        max_nodes=4,
    )

    assert dense.shape == torch.Size([3, 4, 2])
    assert torch.equal(dense[0, 0], values[0])
    assert torch.count_nonzero(dense[1:]) == 0


def test_sparse_to_dense_nodes_handles_non_contiguous_graph_ids():
    values = torch.tensor([[1.0], [2.0]])
    batch = torch.tensor([0, 2])
    local_index = torch.tensor([0, 0])

    dense = sparse_to_dense_nodes(values, batch, local_index)

    assert dense.shape == torch.Size([3, 1, 1])
    assert torch.equal(dense[:, 0, 0], torch.tensor([1.0, 0.0, 2.0]))


def test_sparse_to_dense_nodes_handles_local_index_holes():
    values = torch.tensor([[1.0], [2.0]])
    batch = torch.tensor([0, 0])
    local_index = torch.tensor([0, 3])

    dense = sparse_to_dense_nodes(values, batch, local_index)

    expected = torch.tensor([[[1.0], [0.0], [0.0], [2.0]]])
    assert torch.equal(dense, expected)


def test_sparse_to_dense_nodes_handles_empty_sparse_input_with_explicit_shape():
    values = torch.empty((0, 2), dtype=torch.float64)
    batch = torch.empty((0,), dtype=torch.long)
    local_index = torch.empty((0,), dtype=torch.long)

    dense = sparse_to_dense_nodes(
        values,
        batch,
        local_index,
        num_graphs=2,
        max_nodes=3,
        fill_value=5.0,
    )

    assert dense.shape == torch.Size([2, 3, 2])
    assert dense.dtype == torch.float64
    assert torch.all(dense == 5.0)


def test_sparse_to_dense_nodes_empty_sparse_input_requires_explicit_shape():
    values = torch.empty((0, 2))
    batch = torch.empty((0,), dtype=torch.long)
    local_index = torch.empty((0,), dtype=torch.long)

    with pytest.raises(ValueError, match="num_graphs and max_nodes"):
        sparse_to_dense_nodes(values, batch, local_index)


def test_sparse_to_dense_nodes_preserves_dtype():
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    batch = torch.tensor([0, 1])
    local_index = torch.tensor([0, 0])

    dense = sparse_to_dense_nodes(values, batch, local_index)

    assert dense.dtype == torch.float64


def test_sparse_to_dense_nodes_validates_ranks():
    values = torch.zeros((3, 2))
    batch = torch.tensor([0, 0, 1])
    local_index = torch.tensor([0, 1, 0])

    with pytest.raises(ValueError, match="values with shape"):
        sparse_to_dense_nodes(values.unsqueeze(0), batch, local_index)

    with pytest.raises(ValueError, match="batch with shape"):
        sparse_to_dense_nodes(values, batch.unsqueeze(-1), local_index)

    with pytest.raises(ValueError, match="local_index with shape"):
        sparse_to_dense_nodes(values, batch, local_index.unsqueeze(-1))


def test_sparse_to_dense_nodes_validates_leading_dimensions():
    values = torch.zeros((3, 2))
    batch = torch.tensor([0, 0])
    local_index = torch.tensor([0, 1, 0])

    with pytest.raises(ValueError, match="Dimensional mismatch"):
        sparse_to_dense_nodes(values, batch, local_index)
