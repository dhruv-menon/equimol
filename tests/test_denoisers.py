import pytest
import torch

from equimol.graphs.fully_connected import fully_connected_edges
from equimol.models import MolecularEGNNDenoiser

pytestmark = pytest.mark.denoiser


def _rotation_matrix(dtype=torch.float32, device=None):
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


def _permute_edges(edge_index, perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse[edge_index]


@pytest.fixture()
def denoiser_input():
    num_nodes = 6
    node_dim = 5
    hidden_dim = 16
    h = torch.randn(num_nodes, node_dim)
    x_t = torch.randn(num_nodes, 3)
    edge_index = fully_connected_edges(num_nodes)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    t = torch.tensor([3, 7])
    model = MolecularEGNNDenoiser(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        message_dim=hidden_dim,
        time_embedding_dim=hidden_dim,
        num_layers=2,
    )
    model.eval()
    return model, h, x_t, t, edge_index, batch


def test_molecular_egnn_denoiser_returns_node_coordinate_noise(denoiser_input):
    model, h, x_t, t, edge_index, batch = denoiser_input

    eps_hat = model(h, x_t, t, edge_index, batch=batch)

    assert eps_hat.shape == x_t.shape


def test_molecular_egnn_denoiser_accepts_scalar_timestep(denoiser_input):
    model, h, x_t, _, edge_index, _ = denoiser_input

    eps_hat = model(h, x_t, torch.tensor(3), edge_index)

    assert eps_hat.shape == x_t.shape


def test_molecular_egnn_denoiser_translation_invariance(denoiser_input):
    model, h, x_t, t, edge_index, batch = denoiser_input
    translation = torch.tensor([2.0, -1.0, 0.5])

    eps_hat = model(h, x_t, t, edge_index, batch=batch)
    shifted_eps_hat = model(h, x_t + translation, t, edge_index, batch=batch)

    assert torch.allclose(eps_hat, shifted_eps_hat, atol=1e-5)


def test_molecular_egnn_denoiser_rotation_equivariance(denoiser_input):
    model, h, x_t, t, edge_index, batch = denoiser_input
    rotation = _rotation_matrix(dtype=x_t.dtype, device=x_t.device)

    eps_hat = model(h, x_t, t, edge_index, batch=batch)
    rotated_eps_hat = model(h, x_t @ rotation.T, t, edge_index, batch=batch)

    assert torch.allclose(eps_hat @ rotation.T, rotated_eps_hat, atol=1e-5)


def test_molecular_egnn_denoiser_permutation_equivariance(denoiser_input):
    model, h, x_t, t, edge_index, batch = denoiser_input
    perm = torch.randperm(h.shape[0])

    eps_hat = model(h, x_t, t, edge_index, batch=batch)
    permuted_eps_hat = model(
        h[perm],
        x_t[perm],
        t,
        _permute_edges(edge_index, perm),
        batch=batch[perm],
    )

    assert torch.allclose(eps_hat[perm], permuted_eps_hat, atol=1e-5)


def test_molecular_egnn_denoiser_accepts_edge_attr(denoiser_input):
    _, h, x_t, t, edge_index, batch = denoiser_input
    edge_attr = torch.randn(edge_index.shape[1], 2)
    model = MolecularEGNNDenoiser(
        node_dim=h.shape[-1],
        hidden_dim=16,
        edge_attr_dim=2,
        message_dim=16,
        time_embedding_dim=16,
        num_layers=2,
    )
    model.eval()

    eps_hat = model(h, x_t, t, edge_index, batch=batch, edge_attr=edge_attr)

    assert eps_hat.shape == x_t.shape


def test_molecular_egnn_denoiser_validates_constructor_inputs():
    with pytest.raises(ValueError, match="node_dim"):
        MolecularEGNNDenoiser(node_dim=0)

    with pytest.raises(ValueError, match="hidden_dim"):
        MolecularEGNNDenoiser(node_dim=4, hidden_dim=0)

    with pytest.raises(ValueError, match="num_layers"):
        MolecularEGNNDenoiser(node_dim=4, num_layers=0)

    with pytest.raises(ValueError, match="edge_attr_dim"):
        MolecularEGNNDenoiser(node_dim=4, edge_attr_dim=-1)

    with pytest.raises(ValueError, match="message_dim"):
        MolecularEGNNDenoiser(node_dim=4, message_dim=0)

    with pytest.raises(ValueError, match="time_embedding_dim"):
        MolecularEGNNDenoiser(node_dim=4, time_embedding_dim=0)


def test_molecular_egnn_denoiser_validates_forward_inputs(denoiser_input):
    model, h, x_t, t, edge_index, batch = denoiser_input

    with pytest.raises(ValueError, match="feature dim"):
        model(torch.randn(h.shape[0], h.shape[1] + 1), x_t, t, edge_index, batch=batch)

    with pytest.raises(ValueError, match="coordinate dim"):
        model(h, torch.randn(h.shape[0], 2), t, edge_index, batch=batch)

    with pytest.raises(ValueError, match="graph-wise timesteps require batch"):
        model(h, x_t, t, edge_index)

    with pytest.raises(ValueError, match="batch"):
        model(h, x_t, t, edge_index, batch=torch.tensor([0, 0]))

    with pytest.raises(TypeError, match="edge_index"):
        model(h, x_t, t, edge_index.float(), batch=batch)

    with pytest.raises(ValueError, match="outside"):
        bad_edges = edge_index.clone()
        bad_edges[0, 0] = h.shape[0]
        model(h, x_t, t, bad_edges, batch=batch)

    with pytest.raises(ValueError, match="edge_attr"):
        model(h, x_t, t, edge_index, batch=batch, edge_attr=torch.randn(2, 2))
