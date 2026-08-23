import pytest
import torch

from equimol.graphs import backbone_atom_bond_graph
from equimol.graphs import ca_radius_graph
from equimol.graphs import fully_connected_edges
from equimol.graphs import knn_graph
from equimol.graphs import radius_graph
from equimol.graphs import residue_sequential_graph

pytestmark = pytest.mark.graph


def test_fully_connected_edges_shape_without_self_edges():
    edge_index = fully_connected_edges(4)

    assert edge_index.shape == (2, 12)
    assert not torch.any(edge_index[0] == edge_index[1])


def test_fully_connected_edges_respects_batches():
    batch = torch.tensor([0, 0, 0, 1, 1])
    edge_index = fully_connected_edges(5, batch=batch)

    assert edge_index.shape == (2, 8)
    assert torch.all(batch[edge_index[0]] == batch[edge_index[1]])


def test_radius_and_knn_graph_shapes_are_sparse():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1])

    radius_edges = radius_graph(x, radius=1.1, batch=batch)
    knn_edges = knn_graph(x, k=1, batch=batch)

    assert radius_edges.shape[0] == 2
    assert knn_edges.shape == (2, 5)
    assert torch.all(batch[radius_edges[0]] == batch[radius_edges[1]])
    assert torch.all(batch[knn_edges[0]] == batch[knn_edges[1]])


def test_backbone_atom_bond_graph_builds_directed_flattened_atom_topology():
    edge_index = backbone_atom_bond_graph(2, directed=True)

    expected = torch.tensor(
        [
            [0, 4, 1, 5, 2, 6, 2, 1, 5, 2, 6, 3, 7, 4],
            [1, 5, 2, 6, 3, 7, 4, 0, 4, 1, 5, 2, 6, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_backbone_atom_bond_graph_supports_undirected_and_loops():
    undirected = backbone_atom_bond_graph(1, directed=False, loop=False)
    directed_loop = backbone_atom_bond_graph(1, directed=True, loop=True)

    assert torch.equal(undirected, torch.tensor([[0, 1, 2], [1, 2, 3]]))
    assert directed_loop.shape == torch.Size([2, 10])
    assert torch.equal(directed_loop[:, -4:], torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))


def test_backbone_atom_bond_graph_handles_empty_and_invalid_residue_counts():
    edge_index = backbone_atom_bond_graph(0)

    assert edge_index.shape == torch.Size([2, 0])
    assert edge_index.dtype == torch.long

    with pytest.raises(ValueError, match="num_residues"):
        backbone_atom_bond_graph(-1)


def test_residue_sequential_graph_wraps_sequential_edges():
    edge_index = residue_sequential_graph(4, window=1, directed=False)

    expected = torch.tensor(
        [
            [0, 1, 2, 1, 2, 3],
            [1, 2, 3, 0, 1, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def test_residue_sequential_graph_respects_batches():
    batch = torch.tensor([0, 0, 0, 1, 1])

    edge_index = residue_sequential_graph(5, batch=batch, window=1, directed=False)

    assert torch.all(batch[edge_index[0]] == batch[edge_index[1]])
    assert not torch.any((edge_index[0] == 2) & (edge_index[1] == 3))


def test_ca_radius_graph_uses_ca_coordinates():
    coordinates = torch.zeros(3, 4, 3)
    coordinates[:, 1, :] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    edge_index = ca_radius_graph(coordinates, radius=1.5, loop=False)

    assert torch.equal(edge_index, torch.tensor([[0, 1], [1, 0]]))


def test_ca_radius_graph_validates_backbone_coordinate_shape():
    with pytest.raises(ValueError, match="backbone coordinates"):
        ca_radius_graph(torch.randn(3, 3), radius=1.0)
