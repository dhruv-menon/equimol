"""Specification tests for the rMD17 mode of examples/trainers/md17_regression.py.

These tests define the behaviour of `--dataset rmd17`; they fail until it is
implemented. The default `--dataset md17` keeps the current behaviour.

CLI under test:

    --dataset {md17,rmd17}   default md17
    --datapath DIR           for rmd17, the extracted rMD17 directory:
                                 DIR/npz_data/rmd17_<molecule>.npz
                                 DIR/splits/index_{train,test}_0k.csv
    --splits K [K ...]       official split ids to run, default 1
    --num-val N              validation frames held out of the official train set, default 50

Per split k: official train indices -> split_rmd17(num_val, seed) -> train/val,
official test indices -> test. Energy stats come from the train frames only.
Each split writes its own checkpoint, `<stem>_split{k}<suffix>` (e.g.
ckpt.pt -> ckpt_split1.pt).

main(argv) returns, for rmd17:

    {
        "per_split": [{"split": k, "energy_mae_mev": float, "force_mae_mev": float}, ...],
        "energy_mae_mev": (mean, std),   # over splits, population std (0.0 for one split)
        "force_mae_mev": (mean, std),
    }

where per-split values are the test MAEs returned by trainer(), converted
from kcal/mol (kcal/mol/A) with KCAL_MOL_TO_MEV.

Training loss: energy and force errors are both measured in units of
energy_std (forces are -dE/dx, so they share the energy scale). Changing the
energy unit then leaves the loss unchanged.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from equimol.data import KCAL_MOL_TO_MEV, build_radial_basis
from equimol.models import EGNNRegressor

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "trainers" / "md17_regression.py"
_SPEC = importlib.util.spec_from_file_location("md17_regression_rmd17", _SCRIPT_PATH)
md17_regression = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(md17_regression)

NUM_FRAMES = 30
NUM_TRAIN, NUM_TEST, NUM_VAL = 8, 6, 2
ENERGY_BASE, ENERGY_STEP = -406_000.0, 0.5  # frame i has energy BASE + STEP * i

SMALL_MODEL_ARGS = [
    "--graph", "fully_connected",
    "--num-atom-types", "10",
    "--edge-featurizer-num-basis", "4",
    "--egnn-num-layers", "1",
    "--egnn-hidden-dim", "8",
    "--egnn-message-dim", "8",
    "--batch-size", "4",
    "--epochs", "1",
]


def _write_fake_rmd17(root: Path, molecule: str = "aspirin") -> dict[int, tuple[list[int], list[int]]]:
    """Write a tiny rMD17 directory; return {split: (train, test)} official indices."""
    rng = np.random.default_rng(0)
    (root / "npz_data").mkdir(parents=True)
    np.savez(
        root / "npz_data" / f"rmd17_{molecule}.npz",
        nuclear_charges=np.array([6, 6, 8, 1, 1], dtype=np.int64),
        coords=rng.normal(size=(NUM_FRAMES, 5, 3)),
        energies=ENERGY_BASE + ENERGY_STEP * np.arange(NUM_FRAMES, dtype=np.float64),
        forces=rng.normal(size=(NUM_FRAMES, 5, 3)),
    )

    splits = {}
    (root / "splits").mkdir()
    for k in range(1, 6):
        perm = rng.permutation(NUM_FRAMES).tolist()
        train, test = perm[:NUM_TRAIN], perm[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
        (root / "splits" / f"index_train_{k:02d}.csv").write_text("".join(f"{i}\n" for i in train))
        (root / "splits" / f"index_test_{k:02d}.csv").write_text("".join(f"{i}\n" for i in test))
        splits[k] = (train, test)
    return splits


def _frame_ids(loader: DataLoader) -> list[int]:
    """Recover npz frame indices from the energies of the frames in a loader."""
    ids = []
    for frame in loader.dataset:
        absolute = float(frame.energy.double().sum() + frame.energy_offset.double().sum())
        ids.append(round((absolute - ENERGY_BASE) / ENERGY_STEP))
    return ids


@pytest.fixture
def fake_trainer(monkeypatch):
    """Replace trainer() so main's wiring can be checked without training."""
    calls = []

    def trainer(**kwargs):
        k = len(calls) + 1
        calls.append(kwargs)
        return 1.0 * k, 0.01 * k, 0.2 * k  # best_val_mae, test_energy_mae, test_force_mae (kcal)

    monkeypatch.setattr(md17_regression, "trainer", trainer)
    return calls


def _rmd17_argv(root: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dataset", "rmd17",
        "--datapath", str(root),
        "--num-val", str(NUM_VAL),
        "--checkpoint-path", str(tmp_path / "ckpt.pt"),
        *SMALL_MODEL_ARGS,
        *extra,
    ]


# ----- data wiring -----

def test_rmd17_mode_uses_official_split_with_validation_carved_from_train(tmp_path, fake_trainer):
    splits = _write_fake_rmd17(tmp_path / "rmd17")

    md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path, "--splits", "3"))

    assert len(fake_trainer) == 1
    call = fake_trainer[0]
    train, val, test = (_frame_ids(call[name]) for name in ("train_loader", "val_loader", "test_loader"))
    official_train, official_test = splits[3]

    assert sorted(test) == sorted(official_test)
    assert len(train) == NUM_TRAIN - NUM_VAL and len(val) == NUM_VAL
    assert sorted(train + val) == sorted(official_train)


def test_rmd17_energy_stats_come_from_train_frames_only(tmp_path, fake_trainer):
    splits = _write_fake_rmd17(tmp_path / "rmd17")

    md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path))

    call = fake_trainer[0]
    train_energies = torch.cat([frame.energy.double() for frame in call["train_loader"].dataset])
    assert float(call["energy_mean"]) == pytest.approx(float(train_energies.mean()), abs=1e-4)
    assert float(call["energy_std"]) == pytest.approx(float(train_energies.std(unbiased=False)), abs=1e-4)
    # The default split is 1.
    assert sorted(_frame_ids(call["test_loader"])) == sorted(splits[1][1])


def test_rmd17_validation_split_follows_seed(tmp_path, fake_trainer):
    _write_fake_rmd17(tmp_path / "rmd17")
    argv = _rmd17_argv(tmp_path / "rmd17", tmp_path)

    md17_regression.main([*argv, "--seed", "0"])
    md17_regression.main([*argv, "--seed", "0"])
    md17_regression.main([*argv, "--seed", "1"])

    vals = [sorted(_frame_ids(call["val_loader"])) for call in fake_trainer]
    assert vals[0] == vals[1]
    assert vals[0] != vals[2]


def test_rmd17_each_split_gets_its_own_checkpoint_path(tmp_path, fake_trainer):
    _write_fake_rmd17(tmp_path / "rmd17")

    md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path, "--splits", "1", "4"))

    paths = [Path(call["checkpoint_path"]) for call in fake_trainer]
    assert paths == [tmp_path / "ckpt_split1.pt", tmp_path / "ckpt_split4.pt"]


# ----- reporting -----

def test_rmd17_reports_test_maes_in_mev_with_mean_and_std_over_splits(tmp_path, fake_trainer):
    splits = _write_fake_rmd17(tmp_path / "rmd17")

    results = md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path, "--splits", "1", "2", "5"))

    # fake_trainer returns test MAEs of 0.01k kcal/mol and 0.2k kcal/mol/A on call k.
    energy_kcal = np.array([0.01, 0.02, 0.03])
    force_kcal = np.array([0.2, 0.4, 0.6])
    assert [row["split"] for row in results["per_split"]] == [1, 2, 5]
    assert [row["energy_mae_mev"] for row in results["per_split"]] == pytest.approx(
        list(KCAL_MOL_TO_MEV(1.0) * energy_kcal)
    )
    assert [row["force_mae_mev"] for row in results["per_split"]] == pytest.approx(
        list(KCAL_MOL_TO_MEV(1.0) * force_kcal)
    )

    energy_mev = KCAL_MOL_TO_MEV(1.0) * energy_kcal
    force_mev = KCAL_MOL_TO_MEV(1.0) * force_kcal
    assert results["energy_mae_mev"] == pytest.approx((energy_mev.mean(), energy_mev.std()))
    assert results["force_mae_mev"] == pytest.approx((force_mev.mean(), force_mev.std()))
    # Different splits really use different test sets.
    tests = [sorted(_frame_ids(call["test_loader"])) for call in fake_trainer]
    assert tests == [sorted(splits[k][1]) for k in (1, 2, 5)]


def test_rmd17_single_split_has_zero_std(tmp_path, fake_trainer):
    _write_fake_rmd17(tmp_path / "rmd17")

    results = md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path))

    assert results["energy_mae_mev"][1] == 0.0
    assert results["force_mae_mev"][1] == 0.0


# ----- loss -----

def _unit_scaled_loader(scale: float) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    frames = [
        Data(
            z=torch.tensor([6, 1, 1, 8]),
            pos=torch.randn(4, 3, generator=generator),
            energy=torch.tensor([float(i)]) * scale,
            force=torch.randn(4, 3, generator=generator) * scale,
        )
        for i in range(4)
    ]
    return DataLoader(frames, batch_size=4, shuffle=False)


def test_train_loss_does_not_depend_on_energy_units():
    # Measuring energies in different units (e.g. kcal/mol vs meV) multiplies
    # energies, forces, energy_mean and energy_std by the same factor. A loss
    # with both terms in units of energy_std is unchanged; a force loss in raw
    # units is not, which silently reweights forces against energies.
    torch.manual_seed(0)
    model = EGNNRegressor(
        node_feat_dim=10,
        num_layers=1,
        hidden_dim=16,
        edge_attr_dim=4,
        message_dim=16,
        update_coords=False,
        pooling="mean",
    )
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0)

    losses = []
    for scale in (1.0, 43.3641):
        scaled_model = copy.deepcopy(model)
        losses.append(
            md17_regression.train_epoch(
                model=scaled_model,
                optimizer=torch.optim.SGD(scaled_model.parameters(), lr=0.0),
                loader=_unit_scaled_loader(scale),
                radial_basis=radial_basis,
                device=torch.device("cpu"),
                energy_mean=torch.tensor(1.5 * scale),
                energy_std=torch.tensor(1.2 * scale),
                graph_type="fully_connected",
                num_atom_types=10,
                lambda_energy=1.0,
                lambda_force=10.0,
            )
        )

    assert losses[1] == pytest.approx(losses[0], rel=1e-4)


# ----- end to end -----

def test_rmd17_mode_trains_two_splits_end_to_end(tmp_path):
    _write_fake_rmd17(tmp_path / "rmd17")

    results = md17_regression.main(_rmd17_argv(tmp_path / "rmd17", tmp_path, "--splits", "1", "2"))

    assert [row["split"] for row in results["per_split"]] == [1, 2]
    for row in results["per_split"]:
        assert np.isfinite(row["energy_mae_mev"]) and row["energy_mae_mev"] >= 0.0
        assert np.isfinite(row["force_mae_mev"]) and row["force_mae_mev"] >= 0.0
    assert (tmp_path / "ckpt_split1.pt").is_file()
    assert (tmp_path / "ckpt_split2.pt").is_file()
