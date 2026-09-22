from __future__ import annotations

from pathlib import Path

import torch

BACKBONE_ATOMS = ("N", "CA", "C", "O")
BACKBONE_ATOM_TO_SLOT = {atom: idx for idx, atom in enumerate(BACKBONE_ATOMS)}

RESIDUE_TO_ID = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

ID_TO_RESIDUE = {idx: residue for residue, idx in RESIDUE_TO_ID.items()}


def read_backbone_pdb(path: str | Path) -> dict[str, torch.Tensor]:
    """Read N, CA, C, O ATOM records from a PDB file.

    Returns:
        coordinates: [R, 4, 3]
        atom_mask: [R, 4]
        residue_types: [R]
        residue_index: [R]
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Path specified: {path} does not exist")

    residues: dict[tuple[str, int, str], dict] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line[0:6].strip() != "ATOM":
                continue

            atom_name = line[12:16].strip()
            if atom_name not in BACKBONE_ATOM_TO_SLOT:
                continue

            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue

            resname = line[17:20].strip().upper()
            if resname not in RESIDUE_TO_ID:
                raise ValueError(f"Unsupported residue {resname!r} on line {line_number}")

            chain_id = line[21:22].strip()
            insertion_code = line[26:27].strip()
            try:
                residue_number = int(line[22:26])
                xyz = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            except ValueError as exc:
                raise ValueError(f"Invalid PDB coordinate fields on line {line_number}") from exc

            key = (chain_id, residue_number, insertion_code)
            if key not in residues:
                residues[key] = {
                    "resname": resname,
                    "residue_number": residue_number,
                    "coordinates": torch.zeros((4, 3), dtype=torch.float32),
                    "atom_mask": torch.zeros(4, dtype=torch.bool),
                }

            slot = BACKBONE_ATOM_TO_SLOT[atom_name]
            if residues[key]["atom_mask"][slot]:
                continue
            residues[key]["coordinates"][slot] = torch.tensor(xyz, dtype=torch.float32)
            residues[key]["atom_mask"][slot] = True

    if not residues:
        raise ValueError(f"No backbone ATOM records found in {path}")

    residue_values = list(residues.values())
    return {
        "coordinates": torch.stack(
            [residue["coordinates"] for residue in residue_values],
            dim=0,
        ),
        "atom_mask": torch.stack(
            [residue["atom_mask"] for residue in residue_values],
            dim=0,
        ),
        "residue_types": torch.tensor(
            [RESIDUE_TO_ID[residue["resname"]] for residue in residue_values],
            dtype=torch.long,
        ),
        "residue_index": torch.tensor(
            [residue["residue_number"] for residue in residue_values],
            dtype=torch.long,
        ),
    }


def write_backbone_pdb(
    path: str | Path,
    coordinates: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    residue_types: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    chain_id: str = "A",
) -> None:
    """Write backbone coordinates to a PDB file."""
    path = Path(path)

    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 3):
        raise ValueError(f"coordinates must have shape [R, 4, 3], got {tuple(coordinates.shape)}")
    if not torch.is_floating_point(coordinates):
        coordinates = coordinates.float()

    num_residues = coordinates.shape[0]
    device = coordinates.device

    if atom_mask is None:
        atom_mask = torch.ones((num_residues, 4), dtype=torch.bool, device=device)
    else:
        atom_mask = torch.as_tensor(atom_mask, dtype=torch.bool, device=device)
        if atom_mask.shape != (num_residues, 4):
            raise ValueError(f"atom_mask must have shape [{num_residues}, 4]")

    if residue_types is None:
        residue_types = torch.full((num_residues,), RESIDUE_TO_ID["GLY"], dtype=torch.long)
    else:
        residue_types = torch.as_tensor(residue_types, dtype=torch.long)
        if residue_types.shape != (num_residues,):
            raise ValueError(f"residue_types must have shape [{num_residues}]")

    if residue_index is None:
        residue_index = torch.arange(1, num_residues + 1, dtype=torch.long)
    else:
        residue_index = torch.as_tensor(residue_index, dtype=torch.long)
        if residue_index.shape != (num_residues,):
            raise ValueError(f"residue_index must have shape [{num_residues}]")

    if len(chain_id) != 1:
        raise ValueError("chain_id must be a single character")

    path.parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    lines = []
    coords_cpu = coordinates.detach().cpu()

    for residue_i in range(num_residues):
        res_id = int(residue_types[residue_i].item())
        resname = ID_TO_RESIDUE.get(res_id, "GLY")
        resid = int(residue_index[residue_i].item())

        for atom_i, atom_name in enumerate(BACKBONE_ATOMS):
            if not bool(atom_mask[residue_i, atom_i].item()):
                continue
            x, y, z = coords_cpu[residue_i, atom_i].tolist()
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
                f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
            serial += 1

    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
