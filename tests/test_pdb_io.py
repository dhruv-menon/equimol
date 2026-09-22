import pytest
import torch

from equimol.io import read_backbone_pdb, write_backbone_pdb


def test_read_backbone_pdb_parses_backbone_atoms(tmp_path):
    pdb = tmp_path / "backbone.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   7      11.000  12.000  13.000  1.00  0.00           N",
                "ATOM      2  CA  ALA A   7      14.000  15.000  16.000  1.00  0.00           C",
                "ATOM      3  C   ALA A   7      17.000  18.000  19.000  1.00  0.00           C",
                "ATOM      4  O   ALA A   7      20.000  21.000  22.000  1.00  0.00           O",
                "ATOM      5  CB  ALA A   7      99.000  99.000  99.000  1.00  0.00           C",
                "ATOM      6  N   GLY A   8       1.000   2.000   3.000  1.00  0.00           N",
                "ATOM      7  CA  GLY A   8       4.000   5.000   6.000  1.00  0.00           C",
                "ATOM      8  C   GLY A   8       7.000   8.000   9.000  1.00  0.00           C",
                "ATOM      9  OB  GLY A   8      10.000  11.000  12.000  1.00  0.00           O",
                "HETATM   10  O   HOH A   9       0.000   0.000   0.000  1.00  0.00           O",
            ]
        )
        + "\n"
    )

    out = read_backbone_pdb(pdb)

    assert out["coordinates"].shape == torch.Size([2, 4, 3])
    assert out["atom_mask"].shape == torch.Size([2, 4])
    assert torch.equal(out["residue_types"], torch.tensor([0, 7]))
    assert torch.equal(out["residue_index"], torch.tensor([7, 8]))
    assert torch.allclose(out["coordinates"][0, 1], torch.tensor([14.0, 15.0, 16.0]))
    assert torch.equal(out["atom_mask"][0], torch.tensor([True, True, True, True]))
    assert torch.equal(out["atom_mask"][1], torch.tensor([True, True, True, False]))


def test_read_backbone_pdb_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_backbone_pdb(tmp_path / "missing.pdb")


def test_write_backbone_pdb_roundtrips_backbone_atoms(tmp_path):
    path = tmp_path / "generated.pdb"
    coordinates = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    atom_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    residue_types = torch.tensor([0, 7])
    residue_index = torch.tensor([10, 11])

    write_backbone_pdb(
        path,
        coordinates,
        atom_mask=atom_mask,
        residue_types=residue_types,
        residue_index=residue_index,
    )
    out = read_backbone_pdb(path)

    assert torch.allclose(out["coordinates"], coordinates.masked_fill(~atom_mask.unsqueeze(-1), 0.0))
    assert torch.equal(out["atom_mask"], atom_mask)
    assert torch.equal(out["residue_types"], residue_types)
    assert torch.equal(out["residue_index"], residue_index)
