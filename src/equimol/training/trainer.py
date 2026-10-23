from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .config import FitResult, TrainerConfig
from .device import to_device
from .task import Task


class _EMA(AveragedModel):
    """AveragedModel that keeps buffers in sync by name rather than position.

    A deepcopy of some modules (e3nn's compiled tensor products at l=3) registers
    extra constant buffers, so the copy's buffers no longer line up with the
    model's. Parameters are unaffected and are averaged as usual.
    """

    def update_parameters(self, model: nn.Module) -> None:
        averaged = [p.detach() for p in self.module.parameters()]
        current = [p.detach().to(a.device) for a, p in zip(averaged, model.parameters(), strict=True)]
        if self.n_averaged == 0:
            for a, p in zip(averaged, current):
                a.copy_(p)
        else:
            self.multi_avg_fn(averaged, current, self.n_averaged.to(averaged[0].device))

        buffers = dict(self.module.named_buffers())
        for name, b in model.named_buffers():
            buffers[name].detach().copy_(b.detach().to(buffers[name].device))
        self.n_averaged += 1


class Trainer:
    """Generic fit/validate/test loop with checkpoints, EMA and LR schedules.

    Usage:
        trainer = Trainer(model, task, optimizer, TrainerConfig(epochs=100, checkpoint_path="ckpt.pt"))
        result = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.test(test_loader)

    Checkpoints (written by fit after each validation):
        <checkpoint_path>         best validation so far (by select_metric)
        <stem>.last.pt            most recent validation (used to resume)

    Checkpoint keys:
        model          raw weights, model.state_dict()
        optimizer      optimizer.state_dict()
        scheduler      scheduler.state_dict() or None
        ema            EMA weights as a plain model state_dict (the model's own
                       keys only), or None
        epoch          epoch of this validation
        best_metric    best select_metric so far
        best_epoch     epoch of best_metric
        select_metric  config.select_metric
        history        list of per-validation dicts (see FitResult)
        config         TrainerConfig as a dict, with checkpoint_path as str
        metadata       the metadata dict passed in (e.g. CLI args)
        test_metrics   None until test() fills it in (best checkpoint only)
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        optimizer: Optimizer,
        config: TrainerConfig,
        scheduler: LRScheduler | ReduceLROnPlateau | None = None,
        device: torch.device | str | None = None,
        metadata: dict[str, Any] | None = None,
        on_validation: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Set up the loop; the EMA copy of the model is made here if enabled.

        Args:
            model: Model to train, already on `device`.
            task: Supplies the training loss and validation metrics.
            optimizer: Optimizer over the model's parameters.
            config: Loop settings.
            scheduler: Optional LR scheduler. ReduceLROnPlateau is stepped with the
                selected validation metric; any other scheduler once per epoch.
            device: Device batches are moved to. Defaults to CPU.
            metadata: Extra information saved in every checkpoint.
            on_validation: Called with each validation's history row, after
                its checkpoints are written (e.g. to print progress).
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.task = task
        self.scheduler = scheduler

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.metadata = metadata if metadata is not None else {}
        self.on_validation = on_validation

        if self.config.ema_decay is not None:
            self.ema = _EMA(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))
        else:
            self.ema = None

        self.best = Path(config.checkpoint_path)
        self.last = self.best.with_suffix(".last.pt")

    @property
    def eval_model(self) -> nn.Module:
        """Weights used for validation and testing: the EMA copy when EMA is
        enabled, otherwise the model itself.
        """
        return self.ema.module if self.ema is not None else self.model

    # ----- public API -----

    def fit(self, 
            train_loader: Any, 
            val_loader: Any, 
            resume_from: str | Path | None = None
            ) -> FitResult:
        """Train for config.epochs epochs, validating every config.eval_every
        epochs and after the last one.

        After each validation, saves the best checkpoint (lowest
        config.select_metric) and the most recent one (<stem>.last.pt).

        Args:
            train_loader: Iterable of training batches.
            val_loader: Iterable of validation batches.
            resume_from: Checkpoint written by fit, normally <stem>.last.pt.
                Training continues from the epoch after it, up to config.epochs.

        Returns:
            The best metric, its epoch and the validation history.

        Raises:
            ValueError: If config.select_metric is not among the task's metrics,
                or the train loader is empty.
        """
        self.best.parent.mkdir(parents=True, exist_ok=True)

        if resume_from is None:
            best_metric = float("inf")
            best_epoch = 0
            history = []
            start_epoch = 0
        else:
            ckpt = self._load_state(resume_from)
            start_epoch = ckpt["epoch"]
            best_metric = ckpt["best_metric"]
            best_epoch = ckpt["best_epoch"]
            history = list(ckpt["history"])

        name = self.config.select_metric
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            if epoch % self.config.eval_every != 0 and epoch != self.config.epochs:
                continue

            metrics = self._validate(val_loader)
            if name not in metrics:
                raise ValueError(f"select_metric {name!r} not in validation metrics {sorted(metrics)}")
            value = metrics[name]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(value)

            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            if value < best_metric:
                best_metric = value
                best_epoch = epoch
                torch.save(self._checkpoint_state(epoch, best_metric, best_epoch, history), self.best)
            torch.save(self._checkpoint_state(epoch, best_metric, best_epoch, history), self.last)
            if self.on_validation is not None:
                self.on_validation(history[-1])

        return FitResult(best_metric=best_metric, best_epoch=best_epoch, history=history)

    def test(self, loader: Any) -> dict[str, float]:
        """Evaluate the best checkpoint on `loader` and record the metrics in it.

        Replaces the in-memory weights (and EMA weights, if enabled) with the best
        checkpoint's. <stem>.last.pt is left unchanged, so training can still be
        resumed.

        Returns:
            The task's metrics.
        """
        checkpoint = self._load_state(self.best, restore_optimizer=False)
        metrics = self._validate(loader)
        checkpoint["test_metrics"] = metrics
        torch.save(checkpoint, self.best)
        return metrics

    # ----- internals -----

    def _train_epoch(self, loader: Any) -> float:
        """Run one training epoch and return the mean batch loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            batch = to_device(batch, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.task.loss(self.model, batch)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update_parameters(self.model)
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError("train loader is empty; cannot train an epoch with no batches")
        return total_loss / num_batches

    def _validate(self, loader: Any) -> dict[str, float]:
        """Evaluate eval_model on `loader`.

        Autograd stays enabled so tasks can compute forces as -dE/dx; tasks that
        don't need gradients disable them themselves.
        """
        eval_model = self.eval_model
        eval_model.eval()
        return self.task.evaluate(eval_model, loader=loader, device=self.device)

    def _checkpoint_state(self, epoch: int, best_metric: float, best_epoch: int, history: list[dict]) -> dict:
        """Build a checkpoint dict with the keys listed in the class docstring."""
        config = dataclasses.asdict(self.config)
        config["checkpoint_path"] = str(self.config.checkpoint_path)
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "ema": self._ema_state() if self.ema is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "select_metric": self.config.select_metric,
            "history": list(history),
            "config": config,
            "metadata": self.metadata,
            "test_metrics": None,
        }

    def _ema_state(self) -> dict:
        """EMA weights restricted to the model's own state_dict keys."""
        keys = self.model.state_dict().keys()
        return {k: v for k, v in self.ema.module.state_dict().items() if k in keys}

    def _load_state(self, path: str | Path, restore_optimizer: bool = True) -> dict:
        """Load a checkpoint, restore the weights (and the optimizer and scheduler
        state if restore_optimizer), and return the checkpoint dict.

        EMA weights are restored with n_averaged set to 1: AveragedModel copies the
        raw weights on its first update, which would otherwise overwrite them.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if self.ema is not None and checkpoint["ema"] is not None:
            # The checkpoint holds the model's keys only; keep any extra buffers
            # the EMA copy gained.
            ema_state = self.ema.module.state_dict()
            ema_state.update(checkpoint["ema"])
            self.ema.module.load_state_dict(ema_state)
            self.ema.n_averaged.fill_(1)
        if restore_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None and checkpoint["scheduler"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint
