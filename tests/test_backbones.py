import pytest
import torch

from equimol.graphs.fully_connected import fully_connected_edges
from equimol.models.backbones import EGNNBackbone
from equimol.models.backbones import AttentiveEGNNBackbone

pytestmark = pytest.mark.backbone

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


def test_egnn_backbone_shapes(generate_input):
    _, H, _, h, x, edge_index = generate_input
    backbone = EGNNBackbone(num_layers = 2, hidden_dim = H)
    backbone.eval()
    h_out, x_out = backbone(h, x, edge_index)
    assert h_out.shape == h.shape
    assert x_out.shape == x.shape
    

def test_attentive_egnn_backbone_shapes(generate_input):
    _, H, _, h, x, edge_index = generate_input
    attentive_backbone = AttentiveEGNNBackbone(num_layers = 2, hidden_dim = H)
    attentive_backbone.eval()
    h_out, x_out = attentive_backbone(h, x, edge_index)
    assert h_out.shape == h.shape
    assert x_out.shape == x.shape
    

def test_egnn_backbone_translation_equivariance(generate_input):
    _, H, _, h, x, edge_index = generate_input
    backbone = EGNNBackbone(num_layers = 2, hidden_dim = H)
    backbone.eval()
    h_out, x_out = backbone(h, x, edge_index)

    translation = torch.tensor([1.9, 0.3, -0.92], dtype = x.dtype, device = x.device)    
    h_shift, x_shift = backbone(h, x + translation, edge_index)

    assert torch.allclose(h_out, h_shift, atol = 1e-5)
    assert torch.allclose(x_out + translation, x_shift, atol = 1e-5)


def test_attentive_egnn_backbone_translation_equivariance(generate_input):
    _, H, _, h, x, edge_index = generate_input
    attentive_backbone = AttentiveEGNNBackbone(num_layers = 2, hidden_dim = H)
    attentive_backbone.eval()
    h_out, x_out = attentive_backbone(h, x, edge_index)

    translation = torch.tensor([1.9, 0.3, -0.92], dtype = x.dtype, device = x.device)    
    h_shift, x_shift = attentive_backbone(h, x + translation, edge_index)

    assert torch.allclose(h_out, h_shift, atol = 1e-5)
    assert torch.allclose(x_out + translation, x_shift, atol = 1e-5)


def test_attentive_egnn_backbone_rotation_equivariance(generate_input):
    _, H, _, h, x, edge_index = generate_input
    attentive_backbone = AttentiveEGNNBackbone(num_layers = 2, hidden_dim = H)
    attentive_backbone.eval()
    h_out, x_out = attentive_backbone(h, x, edge_index)

    theta = torch.tensor(torch.pi / 3, dtype = x.dtype, device = x.device)
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
    ) # [3, 3] rotation matrix

    h_rot, x_rot = attentive_backbone(h, x @ Q.T, edge_index)

    assert torch.allclose(h_out, h_rot, atol = 1e-5)
    assert torch.allclose(x_out @ Q.T, x_rot, atol = 1e-5)


def test_attentive_egnn_backbone_permutation_equivariance(generate_input):
    N, H, _, h, x, edge_index = generate_input
    attentive_backbone = AttentiveEGNNBackbone(num_layers = 2, hidden_dim = H)
    attentive_backbone.eval()
    h_out, x_out = attentive_backbone(h, x, edge_index)

    shuffled_indices = torch.randperm(N)
    h_perm, x_perm = attentive_backbone(
        h[shuffled_indices],
        x[shuffled_indices],
        _permute_edges(edge_index, shuffled_indices))

    assert torch.allclose(h_out[shuffled_indices], h_perm, atol = 1e-5)
    assert torch.allclose(x_out[shuffled_indices], x_perm, atol = 1e-5)
