import pytest
import torch

from equimol.graphs.sequential import sequential_edges

pytestmark = pytest.mark.graph


def test_sequential_edges_window_one_undirected():
    edge_index = sequential_edges(4, window=1, directed=False)

    expected = torch.tensor(
        [
            [0, 1, 2, 1, 2, 3],
            [1, 2, 3, 0, 1, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_sequential_edges_window_two_directed():
    edge_index = sequential_edges(4, window=2, directed=True)

    expected = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [1, 2, 3, 2, 3],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_sequential_edges_window_two_undirected():
    edge_index = sequential_edges(4, window=2, directed=False)

    expected = torch.tensor(
        [
            [0, 1, 2, 1, 2, 3, 0, 1, 2, 3],
            [1, 2, 3, 0, 1, 2, 2, 3, 0, 1],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_sequential_edges_includes_self_loops():
    edge_index = sequential_edges(3, window=1, directed=True, loop=True)

    expected = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 1, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_sequential_edges_window_zero_without_loops_returns_empty_edges():
    edge_index = sequential_edges(3, window=0, loop=False)

    assert edge_index.shape == torch.Size([2, 0])
    assert edge_index.dtype == torch.long


def test_sequential_edges_window_zero_with_loops_returns_self_edges():
    edge_index = sequential_edges(3, window=0, loop=True)

    expected = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_sequential_edges_single_node_without_loop_returns_empty_edges():
    edge_index = sequential_edges(1, window=2, loop=False)

    assert edge_index.shape == torch.Size([2, 0])


def test_sequential_edges_empty_graph_returns_empty_edges():
    edge_index = sequential_edges(0, window=1)

    assert edge_index.shape == torch.Size([2, 0])


def test_sequential_edges_respects_batches():
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    edge_index = sequential_edges(6, batch=batch, window=1, directed=False)

    expected = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 2, 0, 1, 4, 5, 3, 4],
        ]
    )
    assert torch.equal(edge_index, expected)
    assert torch.all(batch[edge_index[0]] == batch[edge_index[1]])


def test_sequential_edges_batched_single_node_graphs_without_loop_returns_empty_edges():
    batch = torch.tensor([0, 1, 2], dtype=torch.long)

    edge_index = sequential_edges(3, batch=batch, window=1, loop=False)

    assert edge_index.shape == torch.Size([2, 0])


def test_sequential_edges_validates_inputs():
    with pytest.raises(ValueError, match="num_nodes"):
        sequential_edges(-1)

    with pytest.raises(ValueError, match="window"):
        sequential_edges(3, window=-1)

    with pytest.raises(ValueError, match="batch"):
        sequential_edges(3, batch=torch.tensor([0, 0]))
