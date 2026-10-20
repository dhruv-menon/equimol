"""Specification tests for training improvements in examples/trainers/md17_regression.py.

These tests define three additions to trainer(); they fail until implemented.

    select_metric: "energy" | "force" = "force"
        Which validation MAE picks the best checkpoint (and drives the LR
        scheduler). checkpoint["select_metric"] records it, and
        checkpoint["best_val_mae"] / the first return value are the best value
        of that metric.

    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        Stepped once per validation with the selected metric, before that
        validation's checkpoints are written. Its state is saved as
        checkpoint["scheduler"] (None without a scheduler).

    ema_decay: float | None = None, in (0, 1)
        Keep an exponential moving average of the weights, updated after every
        optimizer step. Validation, best-checkpoint selection and the final
        test use the EMA weights. checkpoint["model"] keeps the raw trained
        weights and checkpoint["ema"] the EMA weights as a plain model
        state_dict, loadable with model.load_state_dict (None without EMA).

CLI (main):

    --select-metric {energy,force}   default force
    --lr-scheduler {none,plateau}    default plateau
    --lr-factor F                    default 0.5
    --lr-patience N                  default 10 (validations, not epochs)
    --min-lr LR                      default 1e-6
    --ema-decay D                    default None (EMA off)
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from equimol.data import build_radial_basis
from equimol.models import EGNNRegressor

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "trainers" / "md17_regression.py"
_SPEC = importlib.util.spec_from_file_location("md17_regression_improvements", _SCRIPT_PATH)
md17_regression = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(md17_regression)

NUM_ATOM_TYPES = 10


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    frames = [
        Data(
            z=torch.tensor([6, 1, 1, 8]),
            pos=torch.randn(4, 3, generator=generator),
            energy=torch.tensor([float(i)]),
            force=torch.randn(4, 3, generator=generator),
        )
        for i in range(4)
    ]
    return DataLoader(frames, batch_size=2, shuffle=False)


def _model() -> EGNNRegressor:
    torch.manual_seed(0)
    return EGNNRegressor(
        node_feat_dim=NUM_ATOM_TYPES,
        num_layers=1,
        hidden_dim=16,
        edge_attr_dim=4,
        message_dim=16,
        update_coords=False,
        pooling="mean",
    )


def _trainer_kwargs(tmp_path: Path, model, optimizer, epochs: int, **extra) -> dict:
    loader = _loader()
    return dict(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        radial_basis=build_radial_basis(num_basis=4, cutoff=5.0),
        device=torch.device("cpu"),
        energy_mean=torch.tensor(1.5),
        energy_std=torch.tensor(1.2),
        epochs=epochs,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        graph_type="fully_connected",
        num_atom_types=NUM_ATOM_TYPES,
        eval_every=1,
        **extra,
    )


@pytest.fixture
def scripted_validation(monkeypatch):
    """Skip training and make validation return scripted metrics.

    Call script(metrics) with one {"energy_mae", "force_mae"} dict per
    validation; the final test evaluation returns the last dict again.
    """

    def script(metrics: list[dict[str, float]]) -> None:
        calls = iter(metrics)

        def evaluate(**kwargs):
            return dict(next(calls, metrics[-1]))

        monkeypatch.setattr(md17_regression, "train_epoch", lambda **kwargs: 0.0)
        monkeypatch.setattr(md17_regression, "evaluate", evaluate)

    return script


# ----- checkpoint selection -----

VAL_METRICS = [
    {"energy_mae": 1.0, "force_mae": 5.0},  # best energy
    {"energy_mae": 2.0, "force_mae": 3.0},  # best force
    {"energy_mae": 3.0, "force_mae": 4.0},
]


@pytest.mark.parametrize(("select_metric", "best_epoch", "best_value"), [("energy", 1, 1.0), ("force", 2, 3.0)])
def test_best_checkpoint_follows_select_metric(tmp_path, scripted_validation, select_metric, best_epoch, best_value):
    scripted_validation(VAL_METRICS)
    model = _model()

    best_val, _, _ = md17_regression.trainer(
        **_trainer_kwargs(
            tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.0), epochs=3, select_metric=select_metric
        )
    )

    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert checkpoint["epoch"] == best_epoch
    assert checkpoint["select_metric"] == select_metric
    assert checkpoint["best_val_mae"] == pytest.approx(best_value)
    assert best_val == pytest.approx(best_value)


def test_select_metric_defaults_to_force(tmp_path, scripted_validation):
    scripted_validation(VAL_METRICS)
    model = _model()

    md17_regression.trainer(
        **_trainer_kwargs(tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.0), epochs=3)
    )

    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["epoch"] == 2


def test_unknown_select_metric_is_rejected(tmp_path):
    model = _model()
    with pytest.raises(ValueError):
        md17_regression.trainer(
            **_trainer_kwargs(
                tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.0), epochs=1, select_metric="total"
            )
        )


# ----- learning-rate schedule -----

@pytest.mark.parametrize(("select_metric", "expect_reduced"), [("force", True), ("energy", False)])
def test_plateau_scheduler_steps_on_the_selected_metric(tmp_path, scripted_validation, select_metric, expect_reduced):
    # Energy keeps improving while force is stuck: only a scheduler watching
    # force should cut the learning rate.
    scripted_validation([{"energy_mae": 10.0 - epoch, "force_mae": 2.0} for epoch in range(6)])
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    md17_regression.trainer(
        **_trainer_kwargs(
            tmp_path, model, optimizer, epochs=6, select_metric=select_metric, scheduler=scheduler
        )
    )

    assert (optimizer.param_groups[0]["lr"] < 0.1) is expect_reduced
    # The last checkpoint is written after the final scheduler step, so the
    # scheduler's best value is the best of the selected metric so far.
    checkpoint = torch.load(tmp_path / "ckpt.last.pt", weights_only=False)
    assert checkpoint["scheduler"]["best"] == pytest.approx(2.0 if select_metric == "force" else 5.0)


def test_checkpoint_scheduler_is_none_without_scheduler(tmp_path, scripted_validation):
    scripted_validation(VAL_METRICS)
    model = _model()

    md17_regression.trainer(
        **_trainer_kwargs(tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.0), epochs=3)
    )

    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["scheduler"] is None


# ----- EMA -----

def _test_metrics(model, trainer_kwargs: dict) -> dict[str, float]:
    keys = ("radial_basis", "device", "energy_mean", "energy_std", "graph_type", "num_atom_types")
    return md17_regression.evaluate(model=model, loader=_loader(), **{k: trainer_kwargs[k] for k in keys})


def test_ema_weights_are_used_for_evaluation(tmp_path):
    # The reported test MAE must come from the EMA weights saved in the best
    # checkpoint, not from the raw weights saved next to them.
    model = _model()
    kwargs = _trainer_kwargs(
        tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.05), epochs=3, ema_decay=0.5
    )

    _, test_energy_mae, test_force_mae = md17_regression.trainer(**kwargs)

    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    ema_model, raw_model = _model(), _model()
    ema_model.load_state_dict(checkpoint["ema"])
    raw_model.load_state_dict(checkpoint["model"])
    ema_metrics = _test_metrics(ema_model, kwargs)
    raw_metrics = _test_metrics(raw_model, kwargs)

    assert test_energy_mae == pytest.approx(ema_metrics["energy_mae"], rel=1e-5)
    assert test_force_mae == pytest.approx(ema_metrics["force_mae"], rel=1e-5)
    assert raw_metrics["force_mae"] != pytest.approx(ema_metrics["force_mae"], rel=1e-5)


def test_checkpoint_stores_ema_state_only_when_enabled(tmp_path):
    model = _model()
    md17_regression.trainer(
        **_trainer_kwargs(tmp_path / "ema", model, torch.optim.SGD(model.parameters(), lr=0.01), epochs=1, ema_decay=0.99)
    )
    model = _model()
    md17_regression.trainer(
        **_trainer_kwargs(tmp_path / "raw", model, torch.optim.SGD(model.parameters(), lr=0.01), epochs=1)
    )

    assert torch.load(tmp_path / "ema" / "ckpt.pt", weights_only=False)["ema"] is not None
    assert torch.load(tmp_path / "raw" / "ckpt.pt", weights_only=False)["ema"] is None


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.5])
def test_ema_decay_must_be_in_open_unit_interval(tmp_path, decay):
    model = _model()
    with pytest.raises(ValueError):
        md17_regression.trainer(
            **_trainer_kwargs(tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.0), epochs=1, ema_decay=decay)
        )


# ----- CLI wiring -----

def _write_tiny_rmd17(root: Path) -> None:
    rng = np.random.default_rng(0)
    (root / "npz_data").mkdir(parents=True)
    np.savez(
        root / "npz_data" / "rmd17_aspirin.npz",
        nuclear_charges=np.array([6, 6, 8, 1, 1], dtype=np.int64),
        coords=rng.normal(size=(12, 5, 3)),
        energies=-406_000.0 + rng.normal(size=12),
        forces=rng.normal(size=(12, 5, 3)),
    )
    (root / "splits").mkdir()
    (root / "splits" / "index_train_01.csv").write_text("".join(f"{i}\n" for i in range(6)))
    (root / "splits" / "index_test_01.csv").write_text("".join(f"{i}\n" for i in range(6, 12)))


@pytest.fixture
def captured_trainer(monkeypatch):
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        return 1.0, 0.1, 0.2

    monkeypatch.setattr(md17_regression, "trainer", trainer)
    return calls


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    _write_tiny_rmd17(tmp_path / "rmd17")
    return [
        "--dataset", "rmd17",
        "--datapath", str(tmp_path / "rmd17"),
        "--num-val", "2",
        "--checkpoint-path", str(tmp_path / "ckpt.pt"),
        "--num-atom-types", str(NUM_ATOM_TYPES),
        *extra,
    ]


def test_cli_defaults_select_force_with_plateau_scheduler_and_no_ema(tmp_path, captured_trainer):
    md17_regression.main(_argv(tmp_path, "--lr", "1e-3"))

    call = captured_trainer[0]
    assert call["select_metric"] == "force"
    assert call["ema_decay"] is None
    scheduler = call["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.optimizer is call["optimizer"]
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 10
    assert scheduler.min_lrs[0] == pytest.approx(1e-6)


def test_cli_passes_scheduler_ema_and_select_metric_options(tmp_path, captured_trainer):
    md17_regression.main(
        _argv(
            tmp_path,
            "--select-metric", "energy",
            "--lr-factor", "0.8",
            "--lr-patience", "3",
            "--min-lr", "1e-5",
            "--ema-decay", "0.99",
        )
    )

    call = captured_trainer[0]
    assert call["select_metric"] == "energy"
    assert call["ema_decay"] == pytest.approx(0.99)
    assert call["scheduler"].factor == pytest.approx(0.8)
    assert call["scheduler"].patience == 3
    assert call["scheduler"].min_lrs[0] == pytest.approx(1e-5)


def test_cli_can_disable_the_scheduler(tmp_path, captured_trainer):
    md17_regression.main(_argv(tmp_path, "--lr-scheduler", "none"))

    assert captured_trainer[0]["scheduler"] is None
