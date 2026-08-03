import pytest
import torch

from equimol.layers import GaussianRadialBasis

pytestmark = pytest.mark.radial


def _rotation_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = torch.tensor(torch.pi / 4, dtype=dtype, device=device)
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


def test_gaussian_radial_basis_shape():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    radial = GaussianRadialBasis(num_basis=8, cutoff=5.0)

    edge_attr = radial(x, edge_index)

    assert edge_attr.shape == torch.Size([4, 8])


def test_gaussian_radial_basis_matches_manual_values():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0], [1]])
    radial = GaussianRadialBasis(num_basis=3, cutoff=2.0, gamma=1.0, eps=0.0)

    edge_attr = radial(x, edge_index)

    centers = torch.tensor([0.0, 1.0, 2.0])
    expected = torch.exp(-(torch.tensor([[1.0]]) - centers).pow(2))
    assert torch.allclose(edge_attr, expected)


def test_gaussian_radial_basis_translation_invariant():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    radial = GaussianRadialBasis(num_basis=8, cutoff=5.0)
    edge_attr = radial(x, edge_index)

    translation = torch.tensor([2.0, -1.0, 0.5])
    shifted_edge_attr = radial(x + translation, edge_index)

    assert torch.allclose(edge_attr, shifted_edge_attr, atol=1e-6, rtol=1e-6)


def test_gaussian_radial_basis_rotation_invariant():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    radial = GaussianRadialBasis(num_basis=8, cutoff=5.0)
    edge_attr = radial(x, edge_index)

    rotation = _rotation_matrix(dtype=x.dtype, device=x.device)
    rotated_edge_attr = radial(x @ rotation.T, edge_index)

    assert torch.allclose(edge_attr, rotated_edge_attr, atol=1e-6, rtol=1e-6)


def test_gaussian_radial_basis_masks_distances_beyond_cutoff():
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0], [1]])
    radial = GaussianRadialBasis(num_basis=4, cutoff=1.0, eps=0.0)

    edge_attr = radial(x, edge_index)

    assert torch.count_nonzero(edge_attr) == 0


@pytest.mark.parametrize(
    ("num_basis", "cutoff", "gamma"),
    [
        (0, 1.0, None),
        (4, 0.0, None),
        (4, 1.0, 0.0),
    ],
)
def test_gaussian_radial_basis_validates_inputs(num_basis, cutoff, gamma):
    with pytest.raises(ValueError):
        GaussianRadialBasis(num_basis=num_basis, cutoff=cutoff, gamma=gamma)
