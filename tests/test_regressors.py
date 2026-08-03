import pytest
import torch

from equimol.graphs import fully_connected_edges
from equimol.models import AttentiveEGNNRegressor
from equimol.models import EGNNRegressor

pytestmark = pytest.mark.regressor


def _permute_edges(edge_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse[edge_index]


def _rotation_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = torch.tensor(torch.pi / 3, dtype=dtype, device=device)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    return torch.stack(
        [
            torch.stack([cos, -sin, zero]),
            torch.stack([sin, cos, zero]),
            torch.stack([zero, zero, one]),
        ]
    )


@pytest.fixture()
def single_graph_input() -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    num_nodes = 8
    node_feat_dim = 6
    h = torch.randn((num_nodes, node_feat_dim), dtype=torch.float32)
    x = torch.randn((num_nodes, 3), dtype=torch.float32)
    edge_index = fully_connected_edges(num_nodes)
    return num_nodes, node_feat_dim, h, x, edge_index


@pytest.fixture()
def batched_graph_input() -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    num_nodes = batch.numel()
    node_feat_dim = 6
    h = torch.randn((num_nodes, node_feat_dim), dtype=torch.float32)
    x = torch.randn((num_nodes, 3), dtype=torch.float32)
    edge_index = fully_connected_edges(num_nodes, batch=batch)
    return 2, node_feat_dim, h, x, edge_index, batch


@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_egnn_regressor_returns_one_scalar_per_graph(single_graph_input, pooling):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = EGNNRegressor(
        node_feat_dim=node_feat_dim,
        hidden_dim=16,
        num_layers=2,
        pooling=pooling,
    )
    model.eval()

    y = model(h, x, edge_index)

    assert y.shape == torch.Size([1])


@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_attentive_egnn_regressor_returns_one_scalar_per_graph(single_graph_input, pooling):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = AttentiveEGNNRegressor(
        node_feat_dim=node_feat_dim,
        hidden_dim=16,
        num_layers=2,
        pooling=pooling,
    )
    model.eval()

    y = model(h, x, edge_index)

    assert y.shape == torch.Size([1])


@pytest.mark.parametrize("model_cls", [EGNNRegressor, AttentiveEGNNRegressor])
def test_regressors_reject_unknown_pooling(model_cls):
    with pytest.raises(ValueError, match="pooling type"):
        model_cls(node_feat_dim=6, pooling="max")


def test_egnn_regressor_translation_invariant(single_graph_input):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = EGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    translation = torch.tensor([2.0, -1.0, 0.5], dtype=x.dtype, device=x.device)
    y_shift = model(h, x + translation, edge_index)

    assert torch.allclose(y, y_shift, atol=1e-5, rtol=1e-5)


def test_attentive_egnn_regressor_translation_invariant(single_graph_input):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = AttentiveEGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    translation = torch.tensor([2.0, -1.0, 0.5], dtype=x.dtype, device=x.device)
    y_shift = model(h, x + translation, edge_index)

    assert torch.allclose(y, y_shift, atol=1e-5, rtol=1e-5)


def test_egnn_regressor_rotation_invariant(single_graph_input):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = EGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    rotation = _rotation_matrix(dtype=x.dtype, device=x.device)
    y_rot = model(h, x @ rotation.T, edge_index)

    assert torch.allclose(y, y_rot, atol=1e-5, rtol=1e-5)


def test_attentive_egnn_regressor_rotation_invariant(single_graph_input):
    _, node_feat_dim, h, x, edge_index = single_graph_input
    model = AttentiveEGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    rotation = _rotation_matrix(dtype=x.dtype, device=x.device)
    y_rot = model(h, x @ rotation.T, edge_index)

    assert torch.allclose(y, y_rot, atol=1e-5, rtol=1e-5)


def test_egnn_regressor_permutation_invariant(single_graph_input):
    num_nodes, node_feat_dim, h, x, edge_index = single_graph_input
    model = EGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    perm = torch.randperm(num_nodes)
    y_perm = model(h[perm], x[perm], _permute_edges(edge_index, perm))

    assert torch.allclose(y, y_perm, atol=1e-5, rtol=1e-5)


def test_attentive_egnn_regressor_permutation_invariant(single_graph_input):
    num_nodes, node_feat_dim, h, x, edge_index = single_graph_input
    model = AttentiveEGNNRegressor(node_feat_dim=node_feat_dim, hidden_dim=16, num_layers=2)
    model.eval()
    y = model(h, x, edge_index)

    perm = torch.randperm(num_nodes)
    y_perm = model(h[perm], x[perm], _permute_edges(edge_index, perm))

    assert torch.allclose(y, y_perm, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model_cls", [EGNNRegressor, AttentiveEGNNRegressor])
@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_regressors_handle_batched_graphs(batched_graph_input, model_cls, pooling):
    num_graphs, node_feat_dim, h, x, edge_index, batch = batched_graph_input
    model = model_cls(
        node_feat_dim=node_feat_dim,
        hidden_dim=16,
        num_layers=2,
        pooling=pooling,
    )
    model.eval()

    y = model(h, x, edge_index, batch=batch)

    assert y.shape == torch.Size([num_graphs])
