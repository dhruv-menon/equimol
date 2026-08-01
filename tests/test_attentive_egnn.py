import torch
import pytest
from equimol.layers.attentive_egnn import AttentiveEGNNLayer
from equimol.graphs import fully_connected_edges

pytestmark = pytest.mark.attentive_egnn


def _permute_edges(edge_index, perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    return inverse[edge_index]

@pytest.fixture()
def generate_input() -> tuple[int, 
                              int, 
                              int, 
                              torch.Tensor, 
                              torch.Tensor, 
                              torch.Tensor]:
    N: int = 8
    H: int = 32
    M: int = 16
    
    h = torch.randn((N, H), dtype = torch.float32)
    x = torch.randn((N, 3), dtype = torch.float32)
    edge_index = fully_connected_edges(N)
    return (N, H, M, h, x, edge_index)

def test_attentive_layer_shapes(generate_input):

    _, H, M, h, x, edge_index = generate_input

    layer = AttentiveEGNNLayer(hidden_dim = H, message_dim = M)
    layer.eval()
    h_out, x_out = layer(h, x, edge_index)

    assert h_out.shape == h.shape
    assert x_out.shape == x.shape

def test_attentive_layer_shapes_with_edge_attr(generate_input):
    _, H, M, h, x, edge_index = generate_input
    E = edge_index.shape[-1]
    A: int = 8
    edge_attr = torch.randn((E, A), dtype = torch.float32)
    
    layer = AttentiveEGNNLayer(hidden_dim = H, message_dim = M, edge_attr_dim = A)
    layer.eval()
    h_out, x_out = layer(h, x, edge_index, edge_attr)
    
    assert h_out.shape == h.shape
    assert x_out.shape == x.shape
    
def test_attentive_layer_translation_equivariance(generate_input):
    _, H, M, h, x, edge_index = generate_input
    layer = AttentiveEGNNLayer(hidden_dim = H, message_dim = M)
    layer.eval()
    h_out, x_out = layer(h, x, edge_index)

    translation = torch.tensor([2.0, 1.8, 0.92], dtype = x.dtype, device = x.device)
    h_shift, x_shift = layer(h, x + translation, edge_index)

    assert torch.allclose(
        h_shift, 
        h_out, 
        atol = 1e-6,
        rtol = 1e-5)

    assert torch.allclose(
        x_shift, 
        x_out + translation,
        atol = 1e-6,
        rtol = 1e-5)
    
def test_attentive_layer_rotation_equivariance(generate_input):
    _, H, M, h, x, edge_index = generate_input
    layer = AttentiveEGNNLayer(hidden_dim = H, message_dim = M)
    layer.eval()
    h_out, x_out = layer(h, x, edge_index)

    theta = torch.tensor(
        torch.pi / 3,
        dtype = x.dtype,
        device = x.device)

    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    Q = torch.stack(
        [
            torch.stack([cos, -sin, zero]),
            torch.stack([sin, cos, zero]),
            torch.stack([zero, zero, one])
        ]
    ) # [3 x 3] rotation matrix

    rotated_x = x @ Q.T
    h_rot, x_rot = layer(h, rotated_x, edge_index)

    assert torch.allclose(
        h_out,
        h_rot,
        atol = 1e-6,
        rtol = 1e-5)

    assert torch.allclose(
        x_out @ Q.T,
        x_rot,
        atol = 1e-6,
        rtol = 1e-5)

def test_attentive_layer_permutation_equivariance(generate_input):
    N, H, M, h, x, edge_index = generate_input
    layer = AttentiveEGNNLayer(hidden_dim = H, message_dim = M)
    layer.eval()
    h_out, x_out = layer(h, x, edge_index)

    shuffled_indices = torch.randperm(N)
    h_perm, x_perm = layer(h[shuffled_indices], 
                           x[shuffled_indices],
                           _permute_edges(edge_index, shuffled_indices))
    assert torch.allclose(h_out[shuffled_indices], h_perm, atol = 1e-5)
    assert torch.allclose(x_out[shuffled_indices], x_perm, atol = 1e-5)
