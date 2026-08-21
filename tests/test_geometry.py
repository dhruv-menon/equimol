import torch
import pytest

from equimol.geometry import backbone_bond_angles
from equimol.geometry import backbone_bond_lengths
from equimol.geometry import backbone_geometry
from equimol.geometry import backbone_torsions
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


def test_backbone_bond_lengths_return_expected_shapes_and_values():
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
            [[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [4.0, 2.0, 0.0]],
            [[4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [5.0, 3.0, 0.0], [6.0, 3.0, 0.0]],
        ]
    )

    lengths = backbone_bond_lengths(coordinates)

    assert torch.allclose(lengths["n_ca"], torch.ones(3), atol=1e-5)
    assert torch.allclose(lengths["ca_c"], torch.ones(3), atol=1e-5)
    assert torch.allclose(lengths["c_o"], torch.ones(3), atol=1e-5)
    assert torch.allclose(lengths["c_n_next"], torch.ones(2), atol=1e-5)
    assert torch.allclose(
        lengths["ca_ca_next"],
        torch.tensor([torch.sqrt(torch.tensor(5.0)), torch.sqrt(torch.tensor(5.0))]),
        atol=1e-5,
    )


def test_backbone_bond_angles_return_expected_shapes_and_values():
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
            [[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [4.0, 2.0, 0.0]],
            [[4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [5.0, 3.0, 0.0], [6.0, 3.0, 0.0]],
        ]
    )

    angles = backbone_bond_angles(coordinates)

    assert torch.allclose(angles["n_ca_c"], torch.full((3,), torch.pi / 2), atol=1e-5)
    assert torch.allclose(
        angles["ca_c_o"],
        torch.tensor([torch.pi, torch.pi, torch.pi]),
        atol=1e-5,
    )
    assert torch.allclose(
        angles["ca_c_n_next"],
        torch.tensor([torch.pi / 2, torch.pi]),
        atol=1e-5,
    )
    assert torch.allclose(
        angles["c_n_next_ca_next"],
        torch.full((2,), torch.pi / 2),
        atol=1e-5,
    )


def test_backbone_torsions_return_sin_cos_pairs():
    torch.manual_seed(0)
    coordinates = torch.randn(5, 4, 3)

    torsions = backbone_torsions(coordinates)

    assert torsions["phi"].shape == torch.Size([4, 2])
    assert torsions["psi"].shape == torch.Size([4, 2])
    assert torsions["omega"].shape == torch.Size([4, 2])
    for value in torsions.values():
        assert torch.allclose(torch.linalg.norm(value, dim=-1), torch.ones(4), atol=1e-5)


def test_backbone_geometry_returns_expected_groups():
    coordinates = torch.randn(4, 4, 3)

    geometry = backbone_geometry(coordinates)

    assert set(geometry) == {"lengths", "angles", "torsions"}
    assert set(geometry["lengths"]) == {"n_ca", "ca_c", "c_o", "c_n_next", "ca_ca_next"}
    assert set(geometry["angles"]) == {
        "n_ca_c",
        "ca_c_o",
        "ca_c_n_next",
        "c_n_next_ca_next",
    }
    assert set(geometry["torsions"]) == {"phi", "psi", "omega"}


def test_backbone_geometry_handles_single_residue():
    coordinates = torch.randn(1, 4, 3)

    geometry = backbone_geometry(coordinates)

    assert geometry["lengths"]["n_ca"].shape == torch.Size([1])
    assert geometry["lengths"]["c_n_next"].shape == torch.Size([0])
    assert geometry["angles"]["n_ca_c"].shape == torch.Size([1])
    assert geometry["angles"]["ca_c_n_next"].shape == torch.Size([0])
    assert geometry["torsions"]["phi"].shape == torch.Size([0, 2])
    assert geometry["torsions"]["psi"].shape == torch.Size([0, 2])
    assert geometry["torsions"]["omega"].shape == torch.Size([0, 2])


def test_backbone_geometry_is_rigid_motion_invariant():
    torch.manual_seed(0)
    coordinates = torch.randn(6, 4, 3)
    rotation = random_rotation(3)
    shift = torch.tensor([2.0, -1.0, 0.5])
    transformed = translate(
        rotate(coordinates.reshape(-1, 3), rotation),
        shift,
    ).reshape_as(coordinates)

    before = backbone_geometry(coordinates)
    after = backbone_geometry(transformed)

    for group in before:
        for key in before[group]:
            assert torch.allclose(before[group][key], after[group][key], atol=1e-5)


def test_backbone_geometry_validates_coordinate_shape():
    with pytest.raises(ValueError, match="backbone coordinates"):
        backbone_bond_lengths(torch.randn(4, 3))

    with pytest.raises(ValueError, match="backbone coordinates"):
        backbone_bond_angles(torch.randn(4, 3, 3))

    with pytest.raises(ValueError, match="backbone coordinates"):
        backbone_torsions(torch.randn(4, 4, 2))
