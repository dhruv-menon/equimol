"""Specification tests for equimol.training, the task-agnostic training loop.

These tests define the API extracted from examples/trainers/*.py. The loop
knows nothing about energies, forces or diffusion: a Task supplies the loss
and the validation metrics.

    from equimol.training import FitResult, Trainer, TrainerConfig, to_device

    class Task(Protocol):
        def loss(self, model, batch) -> Tensor                 # scalar, batch already on device
        def evaluate(self, model, loader, device) -> dict[str, float]

    TrainerConfig(
        epochs: int,
        checkpoint_path: str | Path,
        eval_every: int = 1,
        select_metric: str = "loss",        # key of evaluate(); lower is better
        grad_clip: float | None = None,
        ema_decay: float | None = None,     # in (0, 1)
    )

    trainer = Trainer(model, task, optimizer, config, scheduler=None, device=None, metadata=None)
    result = trainer.fit(train_loader, val_loader, resume_from=None)   # -> FitResult
    test_metrics = trainer.test(test_loader)                           # -> dict[str, float]
    trainer.eval_model                                                 # EMA weights if enabled, else model
    trainer.model, trainer.optimizer, trainer.scheduler                # as passed in

Loop semantics:
    - Each epoch: model.train(), then per batch: to_device -> task.loss ->
      backward -> clip (grad_clip) -> optimizer.step -> EMA update.
    - Validation after every eval_every epochs and after the last epoch:
      eval_model.eval(), then task.evaluate. Autograd stays enabled, so tasks
      can compute forces as -dE/dx during evaluation.
    - ReduceLROnPlateau is stepped with the selected metric after each
      validation; any other scheduler is stepped once per epoch.
    - Checkpoints: <path> holds the best validation so far, <stem>.last.pt the
      most recent validation. Both are written after the scheduler step.
    - test() loads the best checkpoint, evaluates eval_model, and records the
      metrics in the best checkpoint; <stem>.last.pt still describes the last
      epoch afterwards.

Checkpoint keys:
    model (raw weights), optimizer, scheduler (or None), ema (plain model
    state_dict, or None), epoch, best_metric, best_epoch, select_metric,
    history (one dict per validation: epoch, train_loss, and every metric),
    config (TrainerConfig as a dict), metadata, test_metrics (None until test()).

FitResult: best_metric: float, best_epoch: int, history: list[dict].
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch, Data

from equimol.training import FitResult, Trainer, TrainerConfig, to_device

pytestmark = pytest.mark.training


# ----- toy tasks -----

def _toy_loader(num_items: int = 64, batch_size: int = 16, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_items, 3, generator=generator)
    y = x @ torch.tensor([[1.5], [-2.0], [0.5]]) + 0.3
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _toy_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(3, 1)


class ToyRegression:
    """Linear regression: loss is MSE, metrics are loss (MSE) and MAE."""

    def loss(self, model, batch):
        x, y = batch
        return F.mse_loss(model(x), y)

    def evaluate(self, model, loader, device):
        squared, absolute, count = 0.0, 0.0, 0
        for batch in loader:
            x, y = to_device(batch, device)
            error = model(x) - y
            squared += error.pow(2).sum().item()
            absolute += error.abs().sum().item()
            count += y.numel()
        return {"loss": squared / count, "mae": absolute / count}


class ScriptedTask(ToyRegression):
    """Real training loss, but validation returns scripted metrics in order."""

    def __init__(self, metrics: list[dict[str, float]]):
        self.metrics = metrics
        self.calls = 0

    def evaluate(self, model, loader, device):
        metrics = self.metrics[min(self.calls, len(self.metrics) - 1)]
        self.calls += 1
        return dict(metrics)


class GradientInEvaluation(ToyRegression):
    """Evaluation that needs autograd, like forces = -dE/dx."""

    def evaluate(self, model, loader, device):
        total, count = 0.0, 0
        for batch in loader:
            x, _ = to_device(batch, device)
            x = x.clone().requires_grad_(True)
            grad = torch.autograd.grad(model(x).sum(), x)[0]
            total += grad.abs().sum().item()
            count += grad.numel()
        return {"loss": total / count}


def _trainer(tmp_path: Path, task=None, model=None, optimizer=None, scheduler=None, **config) -> Trainer:
    model = model if model is not None else _toy_model()
    optimizer = optimizer if optimizer is not None else torch.optim.SGD(model.parameters(), lr=0.05)
    config = {"epochs": 3, "checkpoint_path": tmp_path / "ckpt.pt", **config}
    return Trainer(
        model,
        task if task is not None else ToyRegression(),
        optimizer,
        TrainerConfig(**config),
        scheduler=scheduler,
    )


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


# ----- to_device -----

def test_to_device_moves_tensors_inside_containers():
    batch = {"x": torch.ones(2), "pair": (torch.zeros(1), [torch.ones(3)]), "name": "frame", "n": 3}

    moved = to_device(batch, torch.device("meta"))

    assert moved["x"].device.type == "meta"
    assert moved["pair"][0].device.type == "meta"
    assert moved["pair"][1][0].device.type == "meta"
    assert isinstance(moved["pair"], tuple) and isinstance(moved["pair"][1], list)
    assert moved["name"] == "frame" and moved["n"] == 3


def test_to_device_uses_dot_to_for_pyg_batches():
    batch = Batch.from_data_list([Data(pos=torch.randn(3, 3)), Data(pos=torch.randn(2, 3))])

    moved = to_device(batch, torch.device("meta"))

    assert isinstance(moved, Batch)
    assert moved.pos.device.type == "meta"
    assert moved.num_graphs == 2


# ----- config validation -----

@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0},
        {"eval_every": 0},
        {"ema_decay": 0.0},
        {"ema_decay": 1.0},
        {"grad_clip": 0.0},
    ],
)
def test_trainer_config_rejects_invalid_values(tmp_path, bad):
    with pytest.raises(ValueError):
        TrainerConfig(**{"epochs": 2, "checkpoint_path": tmp_path / "ckpt.pt", **bad})


# ----- fitting -----

def test_fit_learns_a_learnable_problem(tmp_path):
    trainer = _trainer(tmp_path, epochs=40)
    before = ToyRegression().evaluate(trainer.model, _toy_loader(), torch.device("cpu"))["loss"]

    result = trainer.fit(_toy_loader(), _toy_loader())

    assert isinstance(result, FitResult)
    assert result.best_metric < 0.01 * before


def test_fit_rejects_an_empty_train_loader(tmp_path):
    trainer = _trainer(tmp_path)
    empty = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0, 1)), batch_size=4)

    with pytest.raises(ValueError, match="empty"):
        trainer.fit(empty, _toy_loader())


def test_validation_runs_every_eval_every_epochs_and_after_the_last(tmp_path):
    trainer = _trainer(tmp_path, epochs=7, eval_every=3)

    result = trainer.fit(_toy_loader(), _toy_loader())

    assert [row["epoch"] for row in result.history] == [3, 6, 7]
    for row in result.history:
        assert {"epoch", "train_loss", "loss", "mae"} <= set(row)


def test_evaluation_keeps_autograd_enabled(tmp_path):
    trainer = _trainer(tmp_path, task=GradientInEvaluation(), epochs=1, ema_decay=0.9)

    result = trainer.fit(_toy_loader(), _toy_loader())

    assert result.history[0]["loss"] > 0.0


# ----- checkpoint selection -----

SCRIPT = [
    {"loss": 3.0, "mae": 9.0},
    {"loss": 1.0, "mae": 8.0},  # best loss
    {"loss": 2.0, "mae": 7.0},  # best mae, last epoch
]


@pytest.mark.parametrize(("select_metric", "best_epoch", "best_value"), [("loss", 2, 1.0), ("mae", 3, 7.0)])
def test_best_checkpoint_follows_select_metric(tmp_path, select_metric, best_epoch, best_value):
    trainer = _trainer(tmp_path, task=ScriptedTask(SCRIPT), select_metric=select_metric)

    result = trainer.fit(_toy_loader(), _toy_loader())

    assert (result.best_epoch, result.best_metric) == (best_epoch, pytest.approx(best_value))
    best = _load(tmp_path / "ckpt.pt")
    assert best["epoch"] == best_epoch
    assert best["best_epoch"] == best_epoch
    assert best["best_metric"] == pytest.approx(best_value)
    assert best["select_metric"] == select_metric


def test_unknown_select_metric_fails_at_the_first_validation(tmp_path):
    trainer = _trainer(tmp_path, select_metric="force_mae")

    with pytest.raises(ValueError, match="force_mae"):
        trainer.fit(_toy_loader(), _toy_loader())


def test_last_checkpoint_still_describes_the_last_epoch_after_test(tmp_path):
    # Regression test: the example trainers overwrote <stem>.last.pt with the
    # best checkpoint at the end, so resuming restarted from the best epoch.
    trainer = _trainer(tmp_path, task=ScriptedTask(SCRIPT), select_metric="loss")
    trainer.fit(_toy_loader(), _toy_loader())

    trainer.test(_toy_loader())

    last = _load(tmp_path / "ckpt.last.pt")
    assert last["epoch"] == 3
    assert len(last["history"]) == 3


def test_checkpoint_contains_config_metadata_and_history(tmp_path):
    model = _toy_model()
    trainer = Trainer(
        model,
        ToyRegression(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        TrainerConfig(epochs=2, checkpoint_path=tmp_path / "ckpt.pt", select_metric="mae"),
        metadata={"dataset": "toy"},
    )

    trainer.fit(_toy_loader(), _toy_loader())

    best = _load(tmp_path / "ckpt.pt")
    assert best["metadata"] == {"dataset": "toy"}
    assert best["config"]["epochs"] == 2
    assert best["config"]["select_metric"] == "mae"
    assert best["scheduler"] is None and best["ema"] is None
    assert best["test_metrics"] is None
    assert [row["epoch"] for row in _load(tmp_path / "ckpt.last.pt")["history"]] == [1, 2]


# ----- test() -----

def test_test_evaluates_the_best_checkpoint_and_records_the_metrics(tmp_path):
    trainer = _trainer(tmp_path, epochs=5)
    trainer.fit(_toy_loader(), _toy_loader())
    best_weights = _load(tmp_path / "ckpt.pt")["model"]
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(100.0)  # anything not reloaded from the checkpoint is now wrong

    metrics = trainer.test(_toy_loader(seed=1))

    reference = _toy_model()
    reference.load_state_dict(best_weights)
    expected = ToyRegression().evaluate(reference, _toy_loader(seed=1), torch.device("cpu"))
    assert metrics == pytest.approx(expected)
    assert _load(tmp_path / "ckpt.pt")["test_metrics"] == pytest.approx(expected)


# ----- gradient clipping -----

def test_grad_clip_bounds_the_update_norm(tmp_path):
    model = _toy_model()
    start = [p.detach().clone() for p in model.parameters()]
    # Huge targets give a huge gradient while the weights stay O(1), so the
    # step is not lost to float32 rounding.
    loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.full((4, 1), 1e4)), batch_size=4)
    trainer = _trainer(
        tmp_path, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=1.0), epochs=1, grad_clip=1e-3
    )

    trainer.fit(loader, loader)

    step = torch.cat([(p.detach() - s).flatten() for p, s in zip(model.parameters(), start)])
    assert step.norm() <= 1e-3 * (1 + 1e-4)
    assert step.norm() > 0.5e-3  # clipped to the bound, not to zero


# ----- schedulers -----

@pytest.mark.parametrize(("select_metric", "expect_reduced"), [("mae", True), ("loss", False)])
def test_plateau_scheduler_steps_on_the_selected_metric(tmp_path, select_metric, expect_reduced):
    # loss keeps improving while mae is stuck: only a scheduler watching mae
    # should cut the learning rate.
    task = ScriptedTask([{"loss": 10.0 - epoch, "mae": 2.0} for epoch in range(6)])
    model = _toy_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    trainer = _trainer(
        tmp_path, task=task, model=model, optimizer=optimizer, scheduler=scheduler, epochs=6, select_metric=select_metric
    )

    trainer.fit(_toy_loader(), _toy_loader())

    assert (optimizer.param_groups[0]["lr"] < 0.1) is expect_reduced
    assert _load(tmp_path / "ckpt.last.pt")["scheduler"]["best"] == pytest.approx(2.0 if expect_reduced else 5.0)


def test_other_schedulers_step_once_per_epoch(tmp_path):
    model = _toy_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = _trainer(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler, epochs=3, eval_every=2)

    trainer.fit(_toy_loader(), _toy_loader())

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.125)


# ----- EMA -----

def test_ema_weights_are_used_for_validation_and_test(tmp_path):
    trainer = _trainer(tmp_path, epochs=3, ema_decay=0.5)
    trainer.fit(_toy_loader(), _toy_loader())

    metrics = trainer.test(_toy_loader(seed=1))

    best = _load(tmp_path / "ckpt.pt")
    ema_model, raw_model = _toy_model(), _toy_model()
    ema_model.load_state_dict(best["ema"])
    raw_model.load_state_dict(best["model"])
    ema_metrics = ToyRegression().evaluate(ema_model, _toy_loader(seed=1), torch.device("cpu"))
    raw_metrics = ToyRegression().evaluate(raw_model, _toy_loader(seed=1), torch.device("cpu"))
    assert metrics == pytest.approx(ema_metrics)
    assert raw_metrics["mae"] != pytest.approx(ema_metrics["mae"])
    assert trainer.eval_model is not trainer.model


def test_without_ema_eval_model_is_the_model(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.eval_model is trainer.model


# ----- resume -----

def test_resuming_reproduces_an_uninterrupted_run(tmp_path):
    # Adam moments, plateau scheduler and EMA all carry state between epochs;
    # 2 epochs + resume to 4 must match 4 epochs in one go.
    def build(directory: Path, epochs: int) -> Trainer:
        model = _toy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
        return _trainer(
            directory, model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs, ema_decay=0.8, select_metric="mae"
        )

    straight = build(tmp_path / "straight", epochs=4)
    straight.fit(_toy_loader(), _toy_loader())

    build(tmp_path / "split", epochs=2).fit(_toy_loader(), _toy_loader())
    resumed = build(tmp_path / "split", epochs=4)
    result = resumed.fit(_toy_loader(), _toy_loader(), resume_from=tmp_path / "split" / "ckpt.last.pt")

    assert [row["epoch"] for row in result.history] == [1, 2, 3, 4]
    for name, value in straight.model.state_dict().items():
        assert torch.allclose(resumed.model.state_dict()[name], value, atol=1e-6)
    for name, value in straight.eval_model.state_dict().items():
        assert torch.allclose(resumed.eval_model.state_dict()[name], value, atol=1e-6)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(straight.optimizer.param_groups[0]["lr"])
    assert result.best_epoch == _load(tmp_path / "straight" / "ckpt.pt")["best_epoch"]
