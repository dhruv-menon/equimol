import pytest
import torch

from equimol.adapters import MoleculeAdapter
from equimol.adapters import ProteinBackboneAdapter
from equimol.features import MolecularEdgeGeometryConfig
from equimol.features import edge_angle_summary_features
from equimol.features import edge_bond_features
from equimol.features import edge_torsion_summary_features
from equimol.features import molecule_atom_features
from equimol.features import molecule_edge_features
from equimol.features import molecular_edge_geometry_features
from equimol.features import molecule_geometry_features
from equimol.features import protein_atom_features
from equimol.features import protein_edge_features
from equimol.features import protein_residue_features
from equimol.features.molecule import MolecularGraphTensors
from equimol.graphs import backbone_atom_bond_graph

pytestmark = pytest.mark.feature


def test_molecule_atom_features_use_z_as_node_ids_and_preserve_batch():
    molecule = MoleculeAdapter()(
        {
            "z": [6, 8, 1],
            "coordinates": torch.randn(3, 3),
            "batch": [0, 0, 1],
        }
    )

    features = molecule_atom_features(molecule)

    assert torch.equal(features.node_ids, torch.tensor([6, 8, 1]))
    assert torch.equal(features.batch, torch.tensor([0, 0, 1]))


def test_molecule_atom_features_validate_z_shape():
    molecule = MolecularGraphTensors(
        z=torch.tensor([6]),
        coordinates=torch.randn(2, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="Expected z"):
        molecule_atom_features(molecule)


def test_molecule_edge_features_return_distance_as_edge_attr_when_no_attrs():
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1], [1, 2]])

    features = molecule_edge_features(coordinates, edge_index)

    expected = torch.tensor([[5.0], [torch.sqrt(torch.tensor(18.0))]])
    assert torch.allclose(features.edge_distance, expected, atol=1e-5)
    assert torch.allclose(features.edge_attr, expected, atol=1e-5)


def test_molecule_edge_features_concat_provided_attrs_with_distance():
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    features = molecule_edge_features(coordinates, edge_index, edge_attr=edge_attr)

    expected = torch.tensor([[2.0, 3.0, 1.0], [4.0, 5.0, 1.0]])
    assert torch.allclose(features.edge_attr, expected, atol=1e-5)
    assert features.edge_distance.shape == torch.Size([2, 1])


def test_molecule_edge_features_handle_empty_edges_and_validate_ranges():
    coordinates = torch.randn(2, 3)

    features = molecule_edge_features(
        coordinates,
        torch.empty((2, 0), dtype=torch.long),
    )

    assert features.edge_distance.shape == torch.Size([0, 1])
    assert features.edge_attr.shape == torch.Size([0, 1])

    with pytest.raises(ValueError, match="negative"):
        molecule_edge_features(coordinates, torch.tensor([[0], [-1]]))

    with pytest.raises(ValueError, match="num_nodes=2"):
        molecule_edge_features(coordinates, torch.tensor([[0], [2]]))


def test_molecule_edge_features_validate_edge_attr_length():
    coordinates = torch.randn(2, 3)
    edge_index = torch.tensor([[0], [1]])

    with pytest.raises(ValueError, match="edge_attr"):
        molecule_edge_features(coordinates, edge_index, edge_attr=torch.randn(2, 4))


def test_molecule_geometry_features_return_bond_angle_torsion_features():
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    bond_index = torch.tensor([[0, 1], [1, 2]])
    angle_index = torch.tensor([[0, 1], [1, 2], [2, 3]])
    torsion_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])

    features = molecule_geometry_features(
        coordinates,
        bond_index=bond_index,
        angle_index=angle_index,
        torsion_index=torsion_index,
    )

    assert torch.equal(features.bond_index, bond_index)
    assert torch.equal(features.angle_index, angle_index)
    assert torch.equal(features.torsion_index, torsion_index)
    assert features.bond_lengths.shape == torch.Size([2, 1])
    assert torch.allclose(features.bond_lengths, torch.ones(2, 1), atol=1e-5)
    assert features.angle_features.shape == torch.Size([2, 1])
    assert features.torsion_features.shape == torch.Size([2, 2])
    assert torch.allclose(
        torch.linalg.norm(features.torsion_features, dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


def test_edge_bond_features_map_bonds_to_edges():
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    bond_index = torch.tensor([[0], [1]])
    bond_lengths = torch.tensor([[1.5]])

    features = edge_bond_features(edge_index, bond_index, bond_lengths)

    expected = torch.tensor(
        [
            [1.0, 1.5],
            [1.0, 1.5],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    assert torch.allclose(features, expected)


def test_edge_angle_and_torsion_summary_features_map_to_edges():
    edge_index = torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]])
    angle_index = torch.tensor([[0], [1], [2]])
    angle_features = torch.tensor([[0.25]])
    torsion_index = torch.tensor([[0], [1], [2], [3]])
    torsion_features = torch.tensor([[0.5, 0.75]])

    angle_out = edge_angle_summary_features(edge_index, angle_index, angle_features)
    torsion_out = edge_torsion_summary_features(edge_index, torsion_index, torsion_features)

    assert torch.allclose(angle_out, torch.tensor([[0.25], [0.25], [0.0], [0.0]]))
    assert torch.allclose(
        torsion_out,
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.5, 0.75], [0.5, 0.75]]),
    )


def test_molecular_edge_geometry_features_concatenates_selected_features():
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    bond_index = torch.tensor([[0], [1]])
    bond_lengths = torch.tensor([[1.5]])

    empty = molecular_edge_geometry_features(edge_index)
    features = molecular_edge_geometry_features(
        edge_index,
        bond_index=bond_index,
        bond_lengths=bond_lengths,
        config=MolecularEdgeGeometryConfig(use_bond_features=True),
    )

    assert empty.shape == torch.Size([4, 0])
    assert features.shape == torch.Size([4, 2])


def test_protein_residue_features_use_residue_types_and_mask():
    backbone = ProteinBackboneAdapter().to_backbone_tensors(
        {
            "coordinates": torch.randn(3, 4, 3),
            "residue_types": [5, 7, 9],
            "atom_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                ]
            ),
            "batch": [0, 0, 1],
        }
    )

    features = protein_residue_features(backbone)

    assert torch.equal(features.node_ids, torch.tensor([5, 7, 9]))
    assert torch.equal(features.node_mask, torch.tensor([True, False, True]))
    assert torch.equal(features.batch, torch.tensor([0, 0, 1]))


def test_protein_residue_features_fallback_to_unknown_ids():
    backbone = ProteinBackboneAdapter().to_backbone_tensors(
        {"coordinates": torch.randn(2, 4, 3)}
    )

    features = protein_residue_features(backbone)

    assert torch.equal(features.node_ids, torch.tensor([0, 0]))
    assert torch.equal(features.node_mask, torch.tensor([True, True]))
    assert torch.equal(features.batch, torch.tensor([0, 0]))


def test_protein_atom_features_lift_residue_types_to_atoms():
    atoms = ProteinBackboneAdapter().to_atom_tensors(
        {
            "coordinates": torch.randn(2, 4, 3),
            "residue_types": [5, 7],
            "atom_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "batch": [0, 1],
        }
    )

    features = protein_atom_features(atoms)

    assert torch.equal(features.node_ids, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    assert torch.equal(features.residue_ids, torch.tensor([5, 5, 5, 5, 7, 7, 7, 7]))
    assert torch.equal(
        features.node_mask,
        torch.tensor([True, True, True, True, True, True, False, False]),
    )
    assert torch.equal(features.batch, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))


def test_protein_edge_features_return_distances_offsets_and_edge_types():
    atoms = ProteinBackboneAdapter().to_atom_tensors(
        {"coordinates": torch.randn(2, 4, 3)}
    )
    edge_index = backbone_atom_bond_graph(2, directed=False)
    edge_type = torch.arange(edge_index.shape[1])

    features = protein_edge_features(
        atoms.coordinates,
        edge_index,
        residue_index=atoms.atom_to_residue,
        edge_type=edge_type,
    )

    assert features.edge_distance.shape == torch.Size([7, 1])
    assert features.sequence_offset.shape == torch.Size([7, 1])
    assert features.edge_attr.shape == torch.Size([7, 2])
    assert torch.equal(features.edge_type, edge_type)
    assert torch.equal(features.sequence_offset[-1], torch.tensor([1.0]))


def test_protein_edge_features_validate_residue_index_and_edge_type_shapes():
    coordinates = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]])

    with pytest.raises(ValueError, match="residue_index"):
        protein_edge_features(coordinates, edge_index, residue_index=torch.ones(2))

    with pytest.raises(ValueError, match="edge_type"):
        protein_edge_features(coordinates, edge_index, edge_type=torch.ones(1))
