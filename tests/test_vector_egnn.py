import torch
import pytest

from equimol.models import VectorEGNNRegressor


@pytest.mark.parametrize("vector_gate", [False, True])
def test_vector_egnn_regressor_forward_and_force_shapes(vector_gate):
    num_nodes, num_edges = 6, 12
    node_feat_dim, edge_attr_dim = 10, 5

    h = torch.randn(num_nodes, node_feat_dim)
    x = torch.randn(num_nodes, 3, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    model = VectorEGNNRegressor(
        node_feat_dim=node_feat_dim,
        num_layers=2,
        hidden_dim=16,
        edge_attr_dim=edge_attr_dim,
        message_dim=12,
        vector_dim=8,
        vector_gate=vector_gate,
    )

    energy = model(h, x, edge_index, batch, edge_attr)
    force = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

    assert energy.shape == (2,)
    assert force.shape == x.shape
