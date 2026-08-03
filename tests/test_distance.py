import pytest
import torch

from equimol.layers import PairwiseDistance

pytestmark = pytest.mark.distance


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


def test_pairwise_distance_shape():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    distance = PairwiseDistance(eps=0.0)

    out = distance(x, edge_index)

    assert out.shape == torch.Size([4, 1])


def test_pairwise_distance_matches_manual_values():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1], [1, 2]])
    distance = PairwiseDistance(eps=0.0)

    out = distance(x, edge_index)

    expected = torch.tensor([[5.0], [torch.sqrt(torch.tensor(20.0))]])
    assert torch.allclose(out, expected)


def test_pairwise_squared_distance_matches_manual_values():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1], [1, 2]])
    distance = PairwiseDistance(squared=True)

    out = distance(x, edge_index)

    expected = torch.tensor([[25.0], [20.0]])
    assert torch.allclose(out, expected)


def test_pairwise_distance_translation_invariant():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    distance = PairwiseDistance(eps=0.0)
    out = distance(x, edge_index)

    translation = torch.tensor([2.0, -1.0, 0.5], dtype=x.dtype, device=x.device)
    shifted_out = distance(x + translation, edge_index)

    assert torch.allclose(out, shifted_out, atol=1e-6, rtol=1e-6)


def test_pairwise_distance_rotation_invariant():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    distance = PairwiseDistance(eps=0.0)
    out = distance(x, edge_index)

    rotation = _rotation_matrix(dtype=x.dtype, device=x.device)
    rotated_out = distance(x @ rotation.T, edge_index)

    assert torch.allclose(out, rotated_out, atol=1e-6, rtol=1e-6)


def test_pairwise_distance_validates_eps():
    with pytest.raises(ValueError, match="eps must be non-negative"):
        PairwiseDistance(eps=-1e-8)
