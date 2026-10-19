import pytest
import torch
from torch import nn

from equimol.graphs import fully_connected_edges
from equimol.layers import AttentiveEGNNLayer, EGNNLayer, VectorEGNNLayer
from equimol.models import (
    AttentiveEGNNRegressor,
    EGNNCoordinateDenoiser,
    EGNNRegressor,
    IrrepEGNNRegressor,
    VectorEGNNCoordinateDenoiser,
    VectorEGNNRegressor,
)
from equimol.testing import assert_equivariant, check_equivariance, random_orthogonal

pytestmark = pytest.mark.equivariance

NUM_NODES, NODE_DIM = 7, 5


@pytest.fixture
def graph():
    torch.manual_seed(0)
    h = torch.randn(NUM_NODES, NODE_DIM, dtype=torch.float64)
    x = torch.randn(NUM_NODES, 3, dtype=torch.float64)
    return h, x, fully_connected_edges(NUM_NODES)


@pytest.mark.parametrize("proper", [True, False])
def test_random_orthogonal_has_requested_determinant(proper):
    q = random_orthogonal(3, proper=proper)
    assert torch.allclose(q @ q.T, torch.eye(3, dtype=q.dtype), atol=1e-12)
    assert torch.det(q).item() == pytest.approx(1.0 if proper else -1.0)


# ----- negative controls: the harness must catch violations -----

def test_harness_rejects_absolute_coordinate_dependence(graph):
    _, x, _ = graph
    report = check_equivariance(lambda x: x.sum(dim=-1, keepdim=True), x, output="invariant")
    assert not report.passed


def test_harness_distinguishes_rotations_from_reflections(graph):
    _, x, _ = graph

    def chirality(x):
        rel = x[1:4] - x[0]
        return torch.det(rel).reshape(1)

    assert_equivariant(chirality, x, output="invariant", group="SE3")
    with pytest.raises(AssertionError, match="reflection"):
        assert_equivariant(chirality, x, output="invariant", group="E3")


def test_harness_catches_pointwise_activation_on_vector_channels(graph):
    # Regression test for the original VectorEGNN vector_mlp (bias + SiLU on
    # Cartesian components), which broke rotation equivariance.
    h, x, edge_index = graph
    model = VectorEGNNCoordinateDenoiser(
        NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, vector_dim=8
    ).double().eval()
    for layer in model.backbone.layers:
        layer.vector_mlp = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8)).double()

    report = check_equivariance(lambda x: model(h, x, torch.tensor(10), edge_index), x)
    assert not report.passed


# ----- layers -----

@pytest.mark.parametrize("layer_cls", [EGNNLayer, AttentiveEGNNLayer])
def test_scalar_layers_are_e3_equivariant(graph, layer_cls):
    h, x, edge_index = graph
    layer = layer_cls(hidden_dim=NODE_DIM, message_dim=16).double().eval()
    assert_equivariant(lambda x: layer(h, x, edge_index), x, output=["invariant", "coordinates"])


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("vector_gate", [False, True])
def test_vector_layer_is_e3_equivariant(graph, attention, vector_gate):
    h, x, edge_index = graph
    layer = VectorEGNNLayer(
        hidden_dim=NODE_DIM,
        message_dim=16,
        vector_dim=8,
        attention=attention,
        vector_gate=vector_gate,
    ).double().eval()
    # Non-zero vector states exercise vector_mlp. v is an input that must be
    # rotated together with x, so this checks one transform by hand instead of
    # using the coordinate-only harness.
    v = torch.randn(NUM_NODES, 3, 8, dtype=torch.float64)

    def run(x_in, v_in):
        h_out, v_out, x_out = layer(h, x_in, v_in, edge_index)
        return h_out, v_out.transpose(1, 2), x_out  # [N, V, 3]: coordinate dim last

    reference = run(x, v)
    rotation = random_orthogonal(3, proper=False, dtype=torch.float64)
    shift = torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    rotated = run(x @ rotation.T + shift, torch.einsum("ij,njv->niv", rotation, v))

    assert torch.allclose(rotated[0], reference[0], atol=1e-9)
    assert torch.allclose(rotated[1], reference[1] @ rotation.T, atol=1e-9)
    assert torch.allclose(rotated[2], reference[2] @ rotation.T + shift, atol=1e-9)


# ----- graph regressors (invariant outputs, equivariant forces) -----

REGRESSORS = {
    "egnn": lambda: EGNNRegressor(NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16),
    "attentive-egnn": lambda: AttentiveEGNNRegressor(NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16),
    "vector-egnn": lambda: VectorEGNNRegressor(NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, vector_dim=8),
    "vector-egnn-gated": lambda: VectorEGNNRegressor(
        NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, vector_dim=8, vector_gate=True
    ),
    "irrep-egnn": lambda: IrrepEGNNRegressor(NODE_DIM, num_layers=2, irreps_hidden="8x0e + 4x1o + 2x2e"),
}


@pytest.mark.parametrize("name", REGRESSORS)
def test_regressor_energy_invariant_and_force_equivariant(graph, name):
    h, x, edge_index = graph
    model = REGRESSORS[name]().double().eval()

    assert_equivariant(lambda x: model(h, x, edge_index), x, output="invariant")

    def force(x_in):
        with torch.enable_grad():
            x_in = x_in.clone().requires_grad_(True)
            return -torch.autograd.grad(model(h, x_in, edge_index).sum(), x_in)[0]

    assert_equivariant(force, x, output="equivariant")


# ----- coordinate denoisers (equivariant noise prediction) -----

DENOISERS = {
    "egnn": lambda: EGNNCoordinateDenoiser(NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16),
    "attentive-egnn": lambda: EGNNCoordinateDenoiser(
        NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, attention=True
    ),
    "vector-egnn": lambda: VectorEGNNCoordinateDenoiser(
        NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, vector_dim=8, vector_gate=False
    ),
    "vector-egnn-gated-attention": lambda: VectorEGNNCoordinateDenoiser(
        NODE_DIM, num_layers=2, hidden_dim=16, message_dim=16, vector_dim=8, vector_gate=True, attention=True
    ),
}


@pytest.mark.parametrize("name", DENOISERS)
def test_denoiser_noise_prediction_is_e3_equivariant(graph, name):
    h, x, edge_index = graph
    model = DENOISERS[name]().double().eval()
    assert_equivariant(lambda x: model(h, x, torch.tensor(10), edge_index), x, output="equivariant")
