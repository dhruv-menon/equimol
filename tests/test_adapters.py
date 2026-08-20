import pytest
import torch

from equimol.adapters import MoleculeAdapter
from equimol.adapters import ProteinBackboneAdapter

pytestmark = pytest.mark.adapter


def test_molecule_adapter_builds_minimal_molecule_contract():
    adapter = MoleculeAdapter()
    molecule = {
        "z": [6, 8, 1],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    }

    out = adapter(molecule)

    assert torch.equal(out.z, torch.tensor([6, 8, 1]))
    assert out.z.dtype == torch.long
    assert out.coordinates.shape == torch.Size([3, 3])
    assert out.coordinates.dtype == torch.float32
    assert out.edge_index.shape == torch.Size([2, 0])
    assert out.edge_index.dtype == torch.long
    assert out.edge_attr is None
    assert torch.equal(out.batch, torch.tensor([0, 0, 0]))


def test_molecule_adapter_preserves_provided_edges_attrs_and_batch():
    adapter = MoleculeAdapter()
    molecule = {
        "z": torch.tensor([6, 8, 1]),
        "coordinates": torch.randn(3, 3),
        "edge_index": [[0, 1, 1], [1, 0, 2]],
        "edge_attr": [1.0, 2.0, 3.0],
        "batch": [0, 0, 0],
    }

    out = adapter(molecule)

    assert torch.equal(out.edge_index, torch.tensor([[0, 1, 1], [1, 0, 2]]))
    assert out.edge_attr.shape == torch.Size([3, 1])
    assert out.edge_attr.dtype == torch.float32
    assert torch.equal(out.batch, torch.tensor([0, 0, 0]))


def test_molecule_adapter_validates_coordinate_shape():
    adapter = MoleculeAdapter()

    with pytest.raises(ValueError, match="coordinates with shape"):
        adapter({"z": [6, 8], "coordinates": torch.randn(2, 4, 3)})


def test_molecule_adapter_validates_z_shape():
    adapter = MoleculeAdapter()

    with pytest.raises(ValueError, match="Expected z"):
        adapter({"z": [[6, 8]], "coordinates": torch.randn(2, 3)})

    with pytest.raises(ValueError, match="Expected z"):
        adapter({"z": [6], "coordinates": torch.randn(2, 3)})


def test_molecule_adapter_validates_edge_index_range():
    adapter = MoleculeAdapter()

    with pytest.raises(ValueError, match="negative"):
        adapter(
            {
                "z": [6, 8],
                "coordinates": torch.randn(2, 3),
                "edge_index": [[0], [-1]],
            }
        )

    with pytest.raises(ValueError, match="num_nodes=2"):
        adapter(
            {
                "z": [6, 8],
                "coordinates": torch.randn(2, 3),
                "edge_index": [[0], [2]],
            }
        )


def test_molecule_adapter_validates_edge_attr_length():
    adapter = MoleculeAdapter()

    with pytest.raises(ValueError, match="edge_attr"):
        adapter(
            {
                "z": [6, 8],
                "coordinates": torch.randn(2, 3),
                "edge_index": [[0], [1]],
                "edge_attr": torch.randn(2, 4),
            }
        )


def test_protein_backbone_adapter_builds_minimal_backbone_contract():
    adapter = ProteinBackboneAdapter()
    coordinates = torch.randn(3, 4, 3)

    out = adapter.to_backbone_tensors({"coordinates": coordinates})

    assert out.coordinates.shape == torch.Size([3, 4, 3])
    assert out.coordinates.dtype == torch.float32
    assert out.atom_mask.shape == torch.Size([3, 4])
    assert out.atom_mask.dtype == torch.bool
    assert torch.all(out.atom_mask)
    assert torch.equal(out.residue_index, torch.tensor([0, 1, 2]))
    assert torch.equal(out.batch, torch.tensor([0, 0, 0]))
    assert out.residue_types is None


def test_protein_backbone_adapter_preserves_optional_fields():
    adapter = ProteinBackboneAdapter()
    protein = {
        "coordinates": torch.randn(3, 4, 3),
        "atom_mask": torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]
        ),
        "residue_types": [0, 5, 19],
        "residue_index": [10, 11, 14],
        "batch": [0, 0, 1],
    }

    out = adapter.to_backbone_tensors(protein)

    assert out.atom_mask.dtype == torch.bool
    assert torch.equal(
        out.atom_mask,
        torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
            ]
        ),
    )
    assert torch.equal(out.residue_types, torch.tensor([0, 5, 19]))
    assert torch.equal(out.residue_index, torch.tensor([10, 11, 14]))
    assert torch.equal(out.batch, torch.tensor([0, 0, 1]))


def test_protein_backbone_adapter_validates_shapes():
    adapter = ProteinBackboneAdapter()

    with pytest.raises(ValueError, match="coordinates with shape"):
        adapter.to_backbone_tensors({"coordinates": torch.randn(3, 3)})

    with pytest.raises(ValueError, match="atom_mask"):
        adapter.to_backbone_tensors(
            {
                "coordinates": torch.randn(3, 4, 3),
                "atom_mask": torch.ones(3, 3),
            }
        )

    with pytest.raises(ValueError, match="residue_types"):
        adapter.to_backbone_tensors(
            {
                "coordinates": torch.randn(3, 4, 3),
                "residue_types": torch.ones(2),
            }
        )


def test_protein_backbone_adapter_builds_atom_major_contract():
    adapter = ProteinBackboneAdapter()
    coordinates = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    protein = {
        "coordinates": coordinates,
        "atom_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "residue_types": torch.tensor([5, 7]),
        "residue_index": torch.tensor([10, 11]),
        "batch": torch.tensor([0, 1]),
    }

    out = adapter.to_atom_tensors(protein)

    assert out.coordinates.shape == torch.Size([8, 3])
    assert torch.equal(out.coordinates, coordinates.reshape(-1, 3))
    assert torch.equal(out.atom_types, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    assert torch.equal(out.atom_to_residue, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.equal(out.atom_mask, torch.tensor([True, True, True, True, True, True, False, False]))
    assert torch.equal(out.batch, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.equal(out.residue_types, torch.tensor([5, 7]))
    assert torch.equal(out.residue_index, torch.tensor([10, 11]))
