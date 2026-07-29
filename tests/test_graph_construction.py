import torch

from egnn.graph import fully_connected_edges, knn_graph, radius_graph


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
