import torch

from equimol.geometry import random_rotation, rotate, translate
from equimol.graph import fully_connected_edges
from equimol.layers import EGNNLayer
from equimol.models import EGNNRegressor


def _permute_edges(edge_index, perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    return inverse[edge_index]


def test_layer_translation_equivariance_of_coordinates():
    torch.manual_seed(0)
    h = torch.randn(5, 8)
    x = torch.randn(5, 3)
    edge_index = fully_connected_edges(5)
    shift = torch.tensor([10.0, -2.0, 0.5])
    layer = EGNNLayer(hidden_dim=8, message_dim=16).eval()

    h_out, x_out = layer(h, x, edge_index)
    h_shift, x_shift = layer(h, translate(x, shift), edge_index)

    assert torch.allclose(h_out, h_shift, atol=1e-5)
    assert torch.allclose(translate(x_out, shift), x_shift, atol=1e-5)


def test_layer_rotation_equivariance_of_coordinates():
    torch.manual_seed(1)
    h = torch.randn(5, 8)
    x = torch.randn(5, 3)
    edge_index = fully_connected_edges(5)
    rotation = random_rotation(3)
    layer = EGNNLayer(hidden_dim=8, message_dim=16).eval()

    h_out, x_out = layer(h, x, edge_index)
    h_rot, x_rot = layer(h, rotate(x, rotation), edge_index)

    assert torch.allclose(h_out, h_rot, atol=1e-5)
    assert torch.allclose(rotate(x_out, rotation), x_rot, atol=1e-5)


def test_layer_permutation_equivariance_of_node_ordering():
    torch.manual_seed(2)
    h = torch.randn(6, 8)
    x = torch.randn(6, 3)
    edge_index = fully_connected_edges(6)
    perm = torch.tensor([2, 0, 5, 1, 4, 3])
    layer = EGNNLayer(hidden_dim=8, message_dim=16).eval()

    h_out, x_out = layer(h, x, edge_index)
    h_perm, x_perm = layer(h[perm], x[perm], _permute_edges(edge_index, perm))

    assert torch.allclose(h_out[perm], h_perm, atol=1e-5)
    assert torch.allclose(x_out[perm], x_perm, atol=1e-5)


def test_graph_prediction_is_rigid_motion_and_permutation_invariant():
    torch.manual_seed(3)
    h = torch.randn(7, 4)
    x = torch.randn(7, 3)
    edge_index = fully_connected_edges(7)
    batch = torch.zeros(7, dtype=torch.long)
    rotation = random_rotation(3)
    shift = torch.tensor([1.0, 2.0, -4.0])
    perm = torch.tensor([6, 0, 2, 5, 1, 4, 3])
    model = EGNNRegressor(node_feat_dim=4, hidden_dim=16, num_layers=2, message_dim=16).eval()

    pred = model(h, x, edge_index, batch)
    pred_rigid = model(h, translate(rotate(x, rotation), shift), edge_index, batch)
    pred_perm = model(h[perm], x[perm], _permute_edges(edge_index, perm), batch[perm])

    assert torch.allclose(pred, pred_rigid, atol=1e-5)
    assert torch.allclose(pred, pred_perm, atol=1e-5)
