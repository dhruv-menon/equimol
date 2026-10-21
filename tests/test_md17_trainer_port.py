"""Specification tests for porting examples/trainers/md17_regression.py onto
equimol.training.Trainer and equimol.tasks.EnergyForceTask.

The script keeps its CLI and its results; what changes is who runs the loop.
Data loading, splits, loaders, the radial basis, the model, the optimizer
and the scheduler stay in the script. Training, validation, checkpoints, EMA,
resume and testing move to Trainer; energies, forces, loss and metrics move
to EnergyForceTask. The script's own training loop (train_epoch, evaluate,
trainer) goes away.

Wiring the script must do, per run (one rMD17 split, or the md17 split):

    task = EnergyForceTask(featurize, energy_mean, energy_std,
                           lambda_energy=--lambda-energy, lambda_force=--lambda-force,
                           per_atom_energy=--energy-loss == "per-atom")   # default per-atom
        featurize builds the graph and edge features from the CLI settings
        (--graph, --radius, --k, --num-atom-types, edge featurizer, bond/angle/
        torsion features); energy stats come from the train frames only.

    Trainer(model, task, optimizer, TrainerConfig(...), scheduler=..., device=..., metadata=...)
        TrainerConfig:
            epochs         --epochs
            checkpoint_path  per-split path, <stem>_split{k}<suffix> for rmd17
            eval_every     --eval-every
            select_metric  "energy_mae" | "force_mae"  from --select-metric energy|force
            grad_clip      --grad-clip, with 0 meaning no clipping (None)
            ema_decay      --ema-decay
        scheduler: ReduceLROnPlateau(factor, patience, min_lr) or None (--lr-scheduler none)
        device: cuda if available, else cpu
        metadata: the CLI arguments plus "energy_mean" and "energy_std" as
            floats, so a saved checkpoint can turn the model's normalised
            output back into energies.

    trainer.fit(train_loader, val_loader, resume_from=--resume-from-checkpoint)
    trainer.test(test_loader) -> test MAEs reported by main in meV (rmd17).

Progress: during fit, one line per validation is printed with the epoch
number and the validation force MAE (long GPU runs need it).

main(argv) still returns, for rmd17:
    {
        "per_split": [{"split": k, "energy_mae_mev": float, "force_mae_mev": float}, ...],
        "energy_mae_mev": (mean, std),   # population std over splits
        "force_mae_mev": (mean, std),
    }

These tests replace tests/test_md17_regression_example.py,
tests/test_rmd17_trainer.py and tests/test_md17_training_improvements.py,
which tested the script's own loop. Loss and metric behaviour is covered by
tests/test_energy_force_task.py, loop behaviour by tests/test_training.py.
"""

import importlib.util
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from equimol.data import KCAL_MOL_TO_MEV
from equimol.tasks import EnergyForceTask
from equimol.training import FitResult, Trainer, TrainerConfig

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "trainers" / "md17_regression.py"
_SPEC = importlib.util.spec_from_file_location("md17_regression_port", _SCRIPT_PATH)
md17_regression = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(md17_regression)

NUM_FRAMES = 30
NUM_TRAIN, NUM_TEST, NUM_VAL = 8, 6, 2
NUM_ATOMS = 5
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
        coords=rng.normal(size=(NUM_FRAMES, NUM_ATOMS, 3)),
        energies=ENERGY_BASE + ENERGY_STEP * np.arange(NUM_FRAMES, dtype=np.float64),
        forces=rng.normal(size=(NUM_FRAMES, NUM_ATOMS, 3)),
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


def _frame_ids(loader) -> list[int]:
    """Recover npz frame indices from the energies of the frames in a loader."""
    ids = []
    for frame in loader.dataset:
        absolute = float(frame.energy.double().sum() + frame.energy_offset.double().sum())
        ids.append(round((absolute - ENERGY_BASE) / ENERGY_STEP))
    return ids


def _official_split(tmp_path: Path, k: int) -> tuple[list[int], list[int]]:
    """Official (train, test) frame indices of split k in the fake rMD17 directory."""
    splits_dir = tmp_path / "rmd17" / "splits"

    def read(name: str) -> list[int]:
        return [int(line) for line in (splits_dir / name).read_text().split()]

    return read(f"index_train_{k:02d}.csv"), read(f"index_test_{k:02d}.csv")


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    root = tmp_path / "rmd17"
    if not root.exists():
        _write_fake_rmd17(root)
    return [
        "--dataset", "rmd17",
        "--datapath", str(root),
        "--num-val", str(NUM_VAL),
        "--checkpoint-path", str(tmp_path / "ckpt.pt"),
        *SMALL_MODEL_ARGS,
        *extra,
    ]


@pytest.fixture
def fake_trainer(monkeypatch):
    """Replace the script's Trainer with a recorder, so main's wiring can be
    checked without training.

    Each run records the Trainer arguments (by name, however they were passed)
    plus train_loader, val_loader, resume_from (from fit) and test_loader
    (from test). Run k (1-based) returns test MAEs of 0.01k kcal/mol and
    0.2k kcal/mol/A.
    """
    runs = []
    signature = inspect.signature(Trainer)

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.run = dict(bound.arguments)
            runs.append(self.run)

        def fit(self, train_loader, val_loader, resume_from=None):
            self.run.update(train_loader=train_loader, val_loader=val_loader, resume_from=resume_from)
            history = [{"epoch": 1, "train_loss": 0.0, "energy_mae": 1.0, "force_mae": 1.0}]
            return FitResult(best_metric=1.0, best_epoch=1, history=history)

        def test(self, loader):
            self.run["test_loader"] = loader
            k = next(i for i, run in enumerate(runs, start=1) if run is self.run)
            return {"energy_mae": 0.01 * k, "force_mae": 0.2 * k}

    monkeypatch.setattr(md17_regression, "Trainer", FakeTrainer)
    return runs


# ----- the script delegates to the library -----


def test_script_no_longer_has_its_own_training_loop():
    for name in ("train_epoch", "evaluate", "trainer"):
        assert not hasattr(md17_regression, name), f"{name} should be replaced by Trainer/EnergyForceTask"


def test_main_builds_one_trainer_with_an_energy_force_task_per_split(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--splits", "1", "3"))

    assert len(fake_trainer) == 2
    for run in fake_trainer:
        assert isinstance(run["task"], EnergyForceTask)
        assert isinstance(run["config"], TrainerConfig)
        assert isinstance(run["model"], torch.nn.Module)
        assert {"train_loader", "val_loader", "test_loader"} <= set(run), "fit() and test() must both be called"
    assert fake_trainer[0]["model"] is not fake_trainer[1]["model"], "each split trains a fresh model"


# ----- data wiring -----


def test_rmd17_uses_official_split_with_validation_carved_from_train(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--splits", "3"))

    run = fake_trainer[0]
    official_train, official_test = _official_split(tmp_path, 3)
    train, val, test = (_frame_ids(run[name]) for name in ("train_loader", "val_loader", "test_loader"))
    assert sorted(test) == sorted(official_test)
    assert len(train) == NUM_TRAIN - NUM_VAL and len(val) == NUM_VAL
    assert sorted(train + val) == sorted(official_train)


def test_task_energy_stats_come_from_train_frames_only(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path))

    run = fake_trainer[0]
    train_energies = torch.cat([frame.energy.double() for frame in run["train_loader"].dataset])
    assert run["task"].energy_mean == pytest.approx(float(train_energies.mean()), abs=1e-4)
    assert run["task"].energy_std == pytest.approx(float(train_energies.std(unbiased=False)), abs=1e-4)
    # The default split is 1.
    assert sorted(_frame_ids(run["test_loader"])) == sorted(_official_split(tmp_path, 1)[1])


def test_validation_split_follows_seed(tmp_path, fake_trainer):
    argv = _argv(tmp_path)

    md17_regression.main([*argv, "--seed", "0"])
    md17_regression.main([*argv, "--seed", "0"])
    md17_regression.main([*argv, "--seed", "1"])

    vals = [sorted(_frame_ids(run["val_loader"])) for run in fake_trainer]
    assert vals[0] == vals[1]
    assert vals[0] != vals[2]


@pytest.mark.parametrize(
    ("graph_args", "edges_per_molecule"),
    [
        ((), NUM_ATOMS * (NUM_ATOMS - 1)),  # fully_connected: every ordered pair
        (("--graph", "knn", "--k", "2"), NUM_ATOMS * 2),  # 2 neighbours per atom
    ],
)
def test_task_featurize_follows_the_graph_and_edge_feature_options(
    tmp_path, fake_trainer, graph_args, edges_per_molecule
):
    md17_regression.main(_argv(tmp_path, *graph_args))

    run = fake_trainer[0]
    batch = next(iter(run["train_loader"]))
    geometric_batch = run["task"].featurize(batch)
    num_graphs = int(batch.batch.max()) + 1
    assert geometric_batch.edge_index.shape[1] == num_graphs * edges_per_molecule
    # --edge-featurizer-num-basis 4; --num-atom-types 10.
    assert geometric_batch.edge_attr.shape[1] == 4
    assert geometric_batch.h.shape[1] == 10


def test_task_loss_weights_come_from_the_cli(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--lambda-energy", "0.5", "--lambda-force", "20"))

    task = fake_trainer[0]["task"]
    assert task.lambda_energy == pytest.approx(0.5)
    assert task.lambda_force == pytest.approx(20.0)


def test_energy_loss_defaults_to_per_atom_and_can_use_totals(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path))
    md17_regression.main(_argv(tmp_path, "--energy-loss", "total"))

    assert fake_trainer[0]["task"].per_atom_energy is True
    assert fake_trainer[1]["task"].per_atom_energy is False


# ----- Trainer settings -----


def test_each_split_gets_its_own_checkpoint_path(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--splits", "1", "4"))

    paths = [Path(run["config"].checkpoint_path) for run in fake_trainer]
    assert paths == [tmp_path / "ckpt_split1.pt", tmp_path / "ckpt_split4.pt"]


def test_default_trainer_settings(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--lr", "1e-3"))

    run = fake_trainer[0]
    config = run["config"]
    assert config.epochs == 1
    assert config.eval_every == 1
    assert config.select_metric == "force_mae"
    assert config.grad_clip == pytest.approx(1.0)
    assert config.ema_decay is None
    assert run["device"] == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert run["resume_from"] is None

    scheduler = run["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.optimizer is run["optimizer"]
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 10
    assert scheduler.min_lrs[0] == pytest.approx(1e-6)
    assert run["optimizer"].param_groups[0]["lr"] == pytest.approx(1e-3)


def test_cli_options_reach_the_trainer(tmp_path, fake_trainer):
    md17_regression.main(
        _argv(
            tmp_path,
            "--epochs", "7",
            "--eval-every", "3",
            "--select-metric", "energy",
            "--grad-clip", "5.0",
            "--ema-decay", "0.99",
            "--lr-factor", "0.8",
            "--lr-patience", "3",
            "--min-lr", "1e-5",
        )
    )

    run = fake_trainer[0]
    config = run["config"]
    assert config.epochs == 7
    assert config.eval_every == 3
    assert config.select_metric == "energy_mae"
    assert config.grad_clip == pytest.approx(5.0)
    assert config.ema_decay == pytest.approx(0.99)
    assert run["scheduler"].factor == pytest.approx(0.8)
    assert run["scheduler"].patience == 3
    assert run["scheduler"].min_lrs[0] == pytest.approx(1e-5)


def test_grad_clip_zero_disables_clipping(tmp_path, fake_trainer):
    # TrainerConfig rejects grad_clip=0; the CLI keeps "0 means off".
    md17_regression.main(_argv(tmp_path, "--grad-clip", "0"))

    assert fake_trainer[0]["config"].grad_clip is None


def test_cli_can_disable_the_scheduler(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--lr-scheduler", "none"))

    assert fake_trainer[0]["scheduler"] is None


def test_metadata_records_cli_args_and_energy_stats(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--lr", "2e-3"))

    run = fake_trainer[0]
    metadata = run["metadata"]
    assert metadata["molecule"] == "aspirin"
    assert metadata["lr"] == pytest.approx(2e-3)
    assert metadata["energy_mean"] == pytest.approx(run["task"].energy_mean)
    assert metadata["energy_std"] == pytest.approx(run["task"].energy_std)
    assert isinstance(metadata["energy_mean"], float) and isinstance(metadata["energy_std"], float)


def test_resume_checkpoint_is_passed_to_fit(tmp_path, fake_trainer):
    md17_regression.main(_argv(tmp_path, "--resume-from-checkpoint", str(tmp_path / "ckpt_split1.last.pt")))

    assert Path(fake_trainer[0]["resume_from"]) == tmp_path / "ckpt_split1.last.pt"


def test_resume_rejects_several_splits(tmp_path, fake_trainer):
    with pytest.raises(ValueError):
        md17_regression.main(
            _argv(tmp_path, "--splits", "1", "2", "--resume-from-checkpoint", str(tmp_path / "x.pt"))
        )


# ----- reporting -----


def test_reports_test_maes_in_mev_with_mean_and_std_over_splits(tmp_path, fake_trainer):
    results = md17_regression.main(_argv(tmp_path, "--splits", "1", "2", "5"))

    # Run k's test() returns 0.01k kcal/mol and 0.2k kcal/mol/A.
    energy_mev = KCAL_MOL_TO_MEV(1.0) * np.array([0.01, 0.02, 0.03])
    force_mev = KCAL_MOL_TO_MEV(1.0) * np.array([0.2, 0.4, 0.6])
    assert [row["split"] for row in results["per_split"]] == [1, 2, 5]
    assert [row["energy_mae_mev"] for row in results["per_split"]] == pytest.approx(list(energy_mev))
    assert [row["force_mae_mev"] for row in results["per_split"]] == pytest.approx(list(force_mev))
    assert results["energy_mae_mev"] == pytest.approx((energy_mev.mean(), energy_mev.std()))
    assert results["force_mae_mev"] == pytest.approx((force_mev.mean(), force_mev.std()))
    tests = [sorted(_frame_ids(run["test_loader"])) for run in fake_trainer]
    assert tests == [sorted(_official_split(tmp_path, k)[1]) for k in (1, 2, 5)]


def test_single_split_has_zero_std(tmp_path, fake_trainer):
    results = md17_regression.main(_argv(tmp_path))

    assert results["energy_mae_mev"][1] == 0.0
    assert results["force_mae_mev"][1] == 0.0


# ----- end to end, with the real Trainer -----


def test_trains_two_splits_end_to_end_with_trainer_checkpoints(tmp_path):
    results = md17_regression.main(_argv(tmp_path, "--splits", "1", "2", "--epochs", "2"))

    assert [row["split"] for row in results["per_split"]] == [1, 2]
    for k, row in zip((1, 2), results["per_split"]):
        best = torch.load(tmp_path / f"ckpt_split{k}.pt", weights_only=False)
        last = torch.load(tmp_path / f"ckpt_split{k}.last.pt", weights_only=False)

        assert best["select_metric"] == "force_mae"
        assert [h["epoch"] for h in last["history"]] == [1, 2]
        assert {"energy_mean", "energy_std"} <= set(best["metadata"])
        # The reported numbers are the ones recorded in the best checkpoint.
        assert row["energy_mae_mev"] == pytest.approx(KCAL_MOL_TO_MEV(best["test_metrics"]["energy_mae"]))
        assert row["force_mae_mev"] == pytest.approx(KCAL_MOL_TO_MEV(best["test_metrics"]["force_mae"]))
        assert np.isfinite(row["energy_mae_mev"]) and np.isfinite(row["force_mae_mev"])


def test_resuming_continues_from_the_last_checkpoint(tmp_path):
    md17_regression.main(_argv(tmp_path, "--epochs", "1"))
    md17_regression.main(
        _argv(tmp_path, "--epochs", "3", "--resume-from-checkpoint", str(tmp_path / "ckpt_split1.last.pt"))
    )

    last = torch.load(tmp_path / "ckpt_split1.last.pt", weights_only=False)
    assert last["epoch"] == 3
    assert [h["epoch"] for h in last["history"]] == [1, 2, 3]


def test_prints_progress_once_per_validation(tmp_path, capsys):
    md17_regression.main(_argv(tmp_path, "--epochs", "4", "--eval-every", "2"))

    out = capsys.readouterr().out
    progress = [line for line in out.splitlines() if "force_mae" in line and re.search(r"\bepoch\W{0,3}\d", line)]
    epochs = [int(re.search(r"\bepoch\W{0,3}(\d+)", line).group(1)) for line in progress]
    assert epochs == [2, 4]
