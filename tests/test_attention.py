import torch

from egnn.geometry import random_rotation, rotate, translate
from egnn.graph import fully_connected_edges
from egnn.layers import InvariantEdgeAttention, segmented_softmax


def _permute_edges(edge_index, perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    return inverse[edge_index]


def test_segmented_softmax_normalizes_per_destination_node():
    scores = torch.tensor([[1.0], [2.0], [0.0], [4.0], [-1.0]])
    dst = torch.tensor([0, 0, 1, 1, 1])

    alpha = segmented_softmax(scores, dst, num_nodes=2)
    sums = torch.zeros(2, 1).index_add_(0, dst, alpha)

    assert alpha.shape == (5, 1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_segmented_softmax_is_stable_for_large_scores():
    scores = torch.tensor([[1000.0], [1001.0], [-1000.0], [-999.0]])
    dst = torch.tensor([0, 0, 1, 1])

    alpha = segmented_softmax(scores, dst, num_nodes=2)

    assert torch.isfinite(alpha).all()
    assert torch.allclose(alpha[:2].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(alpha[2:].sum(), torch.tensor(1.0), atol=1e-6)


def test_attention_is_rigid_motion_invariant():
    torch.manual_seed(0)
    h = torch.randn(5, 8)
    x = torch.randn(5, 3)
    edge_index = fully_connected_edges(5)
    rotation = random_rotation(3)
    shift = torch.tensor([2.0, -1.0, 0.5])
    attention = InvariantEdgeAttention(hidden_dim=8, attention_dim=16).eval()

    alpha = attention(h, x, edge_index)
    alpha_rigid = attention(h, translate(rotate(x, rotation), shift), edge_index)

    assert torch.allclose(alpha, alpha_rigid, atol=1e-6)


def test_attention_is_permutation_equivariant_over_edges():
    torch.manual_seed(1)
    h = torch.randn(6, 8)
    x = torch.randn(6, 3)
    edge_index = fully_connected_edges(6)
    perm = torch.tensor([2, 0, 5, 1, 4, 3])
    attention = InvariantEdgeAttention(hidden_dim=8, attention_dim=16).eval()

    alpha = attention(h, x, edge_index)
    alpha_perm = attention(h[perm], x[perm], _permute_edges(edge_index, perm))

    assert torch.allclose(alpha, alpha_perm, atol=1e-6)
