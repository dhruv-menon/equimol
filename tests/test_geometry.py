import torch
import pytest

from equimol.geometry import bond_angle
from equimol.utils import random_rotation, rotate, squared_distances, translate
from equimol.graphs import fully_connected_edges

pytestmark = pytest.mark.geometry


def test_squared_distances_are_rigid_motion_invariant():
    torch.manual_seed(0)
    x = torch.randn(6, 3)
    edge_index = fully_connected_edges(x.size(0))
    rotation = random_rotation(3)
    shift = torch.tensor([2.0, -1.0, 0.5])

    before = squared_distances(x, edge_index)
    after = squared_distances(translate(rotate(x, rotation), shift), edge_index)

    assert torch.allclose(before, after, atol=1e-5)


def test_bond_angle_returns_right_angle_in_radians():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])

    angle = bond_angle(a, b, c)

    assert torch.allclose(angle, torch.tensor(torch.pi / 2), atol=1e-5)


def test_bond_angle_returns_straight_angle_in_radians():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([-1.0, 0.0, 0.0])

    angle = bond_angle(a, b, c)

    assert torch.allclose(angle, torch.tensor(torch.pi), atol=1e-5)


def test_bond_angle_supports_batched_inputs():
    a = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    b = torch.zeros((2, 3))
    c = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    angle = bond_angle(a, b, c)

    expected = torch.tensor([torch.pi / 2, torch.pi])
    assert angle.shape == torch.Size([2])
    assert torch.allclose(angle, expected, atol=1e-5)


def test_bond_angle_supports_broadcasted_center():
    a = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    angle = bond_angle(a, b, c)

    expected = torch.tensor([torch.pi / 2, torch.pi])
    assert angle.shape == torch.Size([2])
    assert torch.allclose(angle, expected, atol=1e-5)


def test_bond_angle_is_translation_invariant():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    shift = torch.tensor([2.0, -1.0, 0.5])

    before = bond_angle(a, b, c)
    after = bond_angle(a + shift, b + shift, c + shift)

    assert torch.allclose(before, after, atol=1e-5)


def test_bond_angle_is_rotation_invariant():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    rotation = random_rotation(3)

    before = bond_angle(a, b, c)
    after = bond_angle(rotate(a, rotation), rotate(b, rotation), rotate(c, rotation))

    assert torch.allclose(before, after, atol=1e-5)


def test_bond_angle_validates_coordinate_dimension():
    a = torch.zeros(2)
    b = torch.zeros(2)
    c = torch.zeros(2)

    with pytest.raises(ValueError, match="coordinate dimension"):
        bond_angle(a, b, c)
