import pytest
import torch

from equimol.layers import global_add_pool
from equimol.layers import global_mean_pool

pytestmark = pytest.mark.pooling


def test_global_mean_pool_single_graph():
    h = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 5.0],
            [10.0, 20.0],
        ]
    )

    pooled = global_mean_pool(h, batch=None)

    assert pooled.shape == torch.Size([1, 2])
    assert torch.allclose(pooled, torch.tensor([[14.0 / 3.0, 28.0 / 3.0]]))


def test_global_mean_pool_batched_graphs_with_unequal_sizes():
    h = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 5.0],
            [10.0, 20.0],
            [20.0, 40.0],
            [30.0, 60.0],
        ]
    )
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    pooled = global_mean_pool(h, batch)

    expected = torch.tensor(
        [
            [2.0, 4.0],
            [20.0, 40.0],
        ]
    )
    assert pooled.shape == torch.Size([2, 2])
    assert torch.allclose(pooled, expected)


def test_global_mean_pool_preserves_dtype():
    h = torch.randn(4, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    pooled = global_mean_pool(h, batch)

    assert pooled.dtype == torch.float64


def test_global_add_and_mean_pool_single_node_graphs_match():
    h = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    batch = torch.tensor([0, 1], dtype=torch.long)

    assert torch.allclose(global_add_pool(h, batch), global_mean_pool(h, batch))
