"""Specification tests for the revised MD17 (rMD17) benchmark protocol.

These tests define the API that equimol.data should provide; they fail until it
is implemented. The protocol follows NequIP/MACE on rMD17:

    - official split k (k = 1..5) gives 1000 train and 1000 test indices
    - the 1000 train indices are split into 950 train + 50 validation
    - the official test indices are the test set
    - energies in kcal/mol, forces in kcal/mol/A; report MAE in meV, meV/A

Files on disk, after torch_geometric.datasets.MD17(root, "revised aspirin")
has downloaded and extracted the archive:

    <root>/raw/rmd17/npz_data/rmd17_aspirin.npz
        nuclear_charges [N], coords [M, N, 3], energies [M], forces [M, N, 3]
    <root>/raw/rmd17/splits/index_train_0k.csv, index_test_0k.csv
        one integer index per line (verify against the real download)

API under test:

    KCAL_MOL_TO_MEV(energy) -> energy in meV (numbers or tensors)
    read_rmd17_split(splits_dir, split=1) -> (train_idx, test_idx)
    split_rmd17(train_idx, num_val=50, seed=42) -> (train_idx, val_idx)
    load_rmd17(npz_path) -> indexable dataset of Data(z, pos, energy, force)
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from equimol.data import KCAL_MOL_TO_MEV, load_rmd17, read_rmd17_split, split_rmd17
from equimol.data.md17 import prepare_md17_batch

pytestmark = pytest.mark.data

# rMD17 aspirin total energies are ~ -4e5 kcal/mol, where float32 spacing is
# 2^-5 = 0.03125 kcal/mol (~1.4 meV), comparable to state-of-the-art energy MAE.
ENERGY_OFFSET = -406_750.0
SMALL_ENERGY_GAP = 0.004  # kcal/mol, below float32 resolution at ENERGY_OFFSET


def _write_split(splits_dir: Path, split: int, train: list[int], test: list[int]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / f"index_train_{split:02d}.csv").write_text("".join(f"{i}\n" for i in train))
    (splits_dir / f"index_test_{split:02d}.csv").write_text("".join(f"{i}\n" for i in test))


def _write_rmd17_npz(path: Path, num_frames: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays = {
        "nuclear_charges": np.array([6, 6, 8, 1, 1], dtype=np.int64),
        "coords": rng.normal(size=(num_frames, 5, 3)),
        "energies": ENERGY_OFFSET + SMALL_ENERGY_GAP * np.arange(num_frames, dtype=np.float64),
        "forces": rng.normal(size=(num_frames, 5, 3)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


# ----- units -----

def test_kcal_mol_to_mev_converts_numbers_and_tensors():
    # 1 kcal/mol = 4184 J/mol / (N_A * e) = 43.364 meV
    assert KCAL_MOL_TO_MEV(1.0) == pytest.approx(43.3641, rel=1e-5)
    assert KCAL_MOL_TO_MEV(2) == pytest.approx(86.7282, rel=1e-5)

    errors = torch.tensor([0.1, 0.5])
    converted = KCAL_MOL_TO_MEV(errors)
    assert isinstance(converted, torch.Tensor)
    assert torch.allclose(converted, errors * 43.3641)


@pytest.mark.parametrize("bad", [None, "1.0", True])
def test_kcal_mol_to_mev_rejects_non_numeric_inputs(bad):
    with pytest.raises(TypeError):
        KCAL_MOL_TO_MEV(bad)


# ----- official splits -----

def test_read_rmd17_split_returns_official_train_and_test_indices(tmp_path):
    _write_split(tmp_path, 3, train=[5, 1, 9], test=[0, 2, 7, 8])

    train_idx, test_idx = read_rmd17_split(tmp_path, split=3)

    assert train_idx.dtype == torch.long and test_idx.dtype == torch.long
    assert train_idx.tolist() == [5, 1, 9]
    assert test_idx.tolist() == [0, 2, 7, 8]


@pytest.mark.parametrize("split", [0, 6])
def test_read_rmd17_split_rejects_unknown_split_ids(tmp_path, split):
    with pytest.raises(ValueError):
        read_rmd17_split(tmp_path, split=split)


def test_read_rmd17_split_raises_for_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rmd17_split(tmp_path, split=1)


def test_read_rmd17_split_rejects_train_test_overlap(tmp_path):
    _write_split(tmp_path, 1, train=[0, 1, 2], test=[2, 3])

    with pytest.raises(ValueError, match="overlap"):
        read_rmd17_split(tmp_path, split=1)


# ----- train / validation split -----

def test_split_rmd17_partitions_official_train_indices():
    official_train = torch.arange(100, 1100)

    train_idx, val_idx = split_rmd17(official_train, num_val=50, seed=0)

    assert train_idx.numel() == 950
    assert val_idx.numel() == 50
    assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(official_train.tolist())


def test_split_rmd17_is_deterministic_per_seed():
    official_train = torch.arange(1000)

    first = split_rmd17(official_train, num_val=50, seed=0)
    repeat = split_rmd17(official_train, num_val=50, seed=0)
    other = split_rmd17(official_train, num_val=50, seed=1)

    assert torch.equal(first[0], repeat[0]) and torch.equal(first[1], repeat[1])
    assert not torch.equal(first[1], other[1])


@pytest.mark.parametrize("num_val", [0, 1000, 1001])
def test_split_rmd17_requires_non_empty_train_and_val(num_val):
    with pytest.raises(ValueError):
        split_rmd17(torch.arange(1000), num_val=num_val)


def test_full_protocol_gives_950_50_1000_disjoint_sets(tmp_path):
    perm = torch.randperm(3000, generator=torch.Generator().manual_seed(0)).tolist()
    _write_split(tmp_path, 1, train=perm[:1000], test=perm[1000:2000])

    official_train, test_idx = read_rmd17_split(tmp_path, split=1)
    train_idx, val_idx = split_rmd17(official_train, num_val=50, seed=42)

    assert (train_idx.numel(), val_idx.numel(), test_idx.numel()) == (950, 50, 1000)
    train, val, test = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


# ----- loading -----

def test_load_rmd17_returns_frames_with_md17_fields(tmp_path):
    npz_path = tmp_path / "rmd17_aspirin.npz"
    arrays = _write_rmd17_npz(npz_path, num_frames=4)

    dataset = load_rmd17(npz_path)

    assert len(dataset) == 4
    frame = dataset[2]
    assert frame.z.dtype == torch.long
    assert torch.equal(frame.z, torch.from_numpy(arrays["nuclear_charges"]))
    assert frame.pos.shape == (5, 3)
    assert frame.force.shape == (5, 3)
    assert torch.allclose(frame.pos.double(), torch.from_numpy(arrays["coords"][2]), atol=1e-6)
    assert torch.allclose(frame.force.double(), torch.from_numpy(arrays["forces"][2]), atol=1e-6)


def test_load_rmd17_preserves_energy_differences_below_float32_resolution(tmp_path):
    # Only energy differences matter for training. Storing ~ -4e5 kcal/mol in
    # float32 rounds away gaps smaller than 0.03 kcal/mol.
    npz_path = tmp_path / "rmd17_aspirin.npz"
    _write_rmd17_npz(npz_path, num_frames=4)

    dataset = load_rmd17(npz_path)

    gaps = [float(dataset[i + 1].energy) - float(dataset[i].energy) for i in range(3)]
    assert gaps == pytest.approx([SMALL_ENERGY_GAP] * 3, abs=1e-6)


def test_rmd17_frames_batch_through_prepare_md17_batch_without_losing_precision(tmp_path):
    npz_path = tmp_path / "rmd17_aspirin.npz"
    _write_rmd17_npz(npz_path, num_frames=2)
    dataset = load_rmd17(npz_path)

    batch = prepare_md17_batch(Batch.from_data_list([dataset[0], dataset[1]]), graph="fully_connected")

    assert batch.y.shape == (2,)
    assert batch.force.shape == (10, 3)
    assert float(batch.y[1] - batch.y[0]) == pytest.approx(SMALL_ENERGY_GAP, abs=1e-6)
