import torch
from e3nn import o3

from equimol.layers.irrep_egnn import IrrepEGNNBackbone, IrrepEGNNLayer
from equimol.models.irrep_egnn import IrrepEGNNRegressor


def test_irrep_egnn_layer_and_backbone_shapes():
    irreps_hidden = "4x0e + 2x1o + 1x2e"
    irreps_edge = "0e + 1o + 2e"
    hidden_dim = o3.Irreps(irreps_hidden).dim

    num_nodes, num_edges = 5, 9
    h = torch.randn(num_nodes, hidden_dim)
    x = torch.randn(num_nodes, 3, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    layer = IrrepEGNNLayer(
        irreps_hidden=irreps_hidden,
        irreps_edge=irreps_edge,
        edge_attr_dim=3,
        radial_hidden_dim=16,
        attention=True,
    )
    h_next = layer(h, x, edge_index, edge_attr)
    assert h_next.shape == h.shape

    backbone = IrrepEGNNBackbone(
        num_layers=2,
        irreps_hidden=irreps_hidden,
        irreps_edge=irreps_edge,
        edge_attr_dim=3,
        radial_hidden_dim=16,
        attention=True,
    )
    h_final = backbone(h, x, edge_index, edge_attr)
    assert h_final.shape == h.shape

    loss = h_final.pow(2).sum()
    grad = torch.autograd.grad(loss, x)[0]
    assert grad.shape == x.shape


def test_irrep_egnn_regressor_forward_and_force_shapes():
    num_nodes, num_edges = 6, 12
    node_feat_dim, edge_attr_dim = 10, 5

    h = torch.randn(num_nodes, node_feat_dim)
    x = torch.randn(num_nodes, 3, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    model = IrrepEGNNRegressor(
        node_feat_dim=node_feat_dim,
        num_layers=2,
        edge_attr_dim=edge_attr_dim,
        irreps_hidden="4x0e + 2x1o + 1x2e",
        irreps_edge="0e + 1o + 2e",
        radial_hidden_dim=16,
    )

    energy = model(h, x, edge_index, batch, edge_attr)
    force = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

    assert energy.shape == (2,)
    assert force.shape == x.shape
