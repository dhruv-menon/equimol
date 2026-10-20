from __future__ import annotations

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


class Trainer:
    """Generic fit/validate/test loop with checkpoints, EMA and LR schedules.

    Usage:
        trainer = Trainer(model, task, optimizer, TrainerConfig(epochs=100, checkpoint_path="ckpt.pt"))
        result = trainer.fit(train_loader, val_loader)
        test_metrics = trainer.test(test_loader)

    Checkpoints (written by fit, both after the scheduler step):
        <checkpoint_path>         best validation so far (by select_metric)
        <stem>.last.pt            most recent validation (used to resume)

    Checkpoint dict keys (tests read these exact names):
        model          raw weights, model.state_dict()
        optimizer      optimizer.state_dict()
        scheduler      scheduler.state_dict() or None
        ema            EMA weights as a plain model state_dict (ema.module.state_dict()) or None
        epoch          epoch of this validation
        best_metric    best select_metric so far
        best_epoch     epoch of best_metric
        select_metric  config.select_metric
        history        list of per-validation dicts (see FitResult)
        config         dataclasses.asdict(config), with checkpoint_path as str
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
    ) -> None:
        """Store everything and build the EMA wrapper if config.ema_decay is set.

        Steps:
            1. Keep model, task, optimizer, config, scheduler as public
               attributes (tests use trainer.model / .optimizer).
            2. device: default to CPU when None. Assume the caller already
               moved the model there; the Trainer only moves batches.
            3. metadata: default to {}.
            4. EMA: if config.ema_decay is not None, wrap the model:
                   AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
               otherwise self.ema = None.
               AveragedModel deep-copies the model; the copy lives in .module.
            5. Paths: best = Path(config.checkpoint_path),
               last = best.with_suffix(".last.pt").
        """
        # TODO: implement.
        raise NotImplementedError

    @property
    def eval_model(self) -> nn.Module:
        """The weights to validate/test with: the EMA copy (self.ema.module)
        when EMA is on, otherwise self.model itself (same object)."""
        # TODO: implement.
        raise NotImplementedError

    # ----- public API -----

    def fit(self, train_loader: Any, val_loader: Any, resume_from: str | Path | None = None) -> FitResult:
        """Train for config.epochs epochs, validating and checkpointing.

        Steps:
            1. Make the checkpoint directory (parents=True, exist_ok=True).
            2. Start state: best_metric=inf, best_epoch=0, history=[], start_epoch=0.
               If resume_from: self._load_state(...) restores model/optimizer/
               scheduler/EMA and returns the saved epoch, best_metric,
               best_epoch and history; continue from epoch + 1.
            3. For epoch in start_epoch+1 .. config.epochs:
                a. train_loss = self._train_epoch(train_loader)
                b. Non-plateau scheduler: step once per epoch, here.
                c. Skip validation unless epoch % eval_every == 0 or it is
                   the last epoch.
                d. metrics = self._validate(val_loader)
                e. value = metrics[select_metric]; if the key is missing,
                   raise ValueError naming the metric and the available keys
                   (better than a bare KeyError deep in a long run).
                f. ReduceLROnPlateau: scheduler.step(value), here, so it
                   watches the same metric that picks the best checkpoint.
                g. Append {"epoch", "train_loss", **metrics} to history.
                h. If value < best_metric: update best, save the best
                   checkpoint. Always save the last checkpoint.
            4. Return FitResult(best_metric, best_epoch, history).

        Why step (b) and (f) differ: StepLR/cosine schedules count epochs,
        but ReduceLROnPlateau needs a metric and should only be told about
        validations. isinstance(self.scheduler, ReduceLROnPlateau) tells
        them apart.
        """
        # TODO: implement.
        raise NotImplementedError

    def test(self, loader: Any) -> dict[str, float]:
        """Evaluate the BEST checkpoint (not the in-memory weights) on loader.

        Steps:
            1. Load the best checkpoint; restore model weights (and EMA
               weights if EMA is on) from it.
            2. metrics = self._validate(loader)
            3. Write metrics into that checkpoint's "test_metrics" and save it
               back to the best path only. Do not touch <stem>.last.pt: it
               must keep describing the last epoch so resuming works (this
               was a bug in the example trainers).
            4. Return metrics.
        """
        # TODO: implement.
        raise NotImplementedError

    # ----- internals -----

    def _train_epoch(self, loader: Any) -> float:
        """One pass over loader; return the mean training loss.

        Per batch:
            batch = to_device(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = self.task.loss(self.model, batch); loss.backward()
            clip with torch.nn.utils.clip_grad_norm_ if config.grad_clip
            optimizer.step()
            EMA: self.ema.update_parameters(self.model) after every step

        Call self.model.train() first. Raise ValueError mentioning "empty"
        if the loader yielded no batches (otherwise you divide by zero).
        Accumulate loss.item() (a float), not the tensor, or every batch's
        graph stays alive.
        """
        # TODO: implement.
        raise NotImplementedError

    def _validate(self, loader: Any) -> dict[str, float]:
        """eval_model.eval(), then return self.task.evaluate(eval_model, loader, device).

        Do not wrap this in torch.no_grad(): energy/force tasks need autograd
        to compute forces. Tasks that don't need gradients use no_grad
        themselves.
        """
        # TODO: implement.
        raise NotImplementedError

    def _checkpoint_state(self, epoch: int, best_metric: float, best_epoch: int, history: list[dict]) -> dict:
        """Build the checkpoint dict with exactly the keys listed in the class
        docstring. Copy history (list(history)) so later appends don't mutate
        a dict you already saved. test_metrics starts as None."""
        # TODO: implement.
        raise NotImplementedError

    def _load_state(self, path: str | Path, restore_optimizer: bool = True) -> dict:
        """Load a checkpoint and restore weights (and training state).

        Steps:
            1. torch.load(path, map_location=self.device, weights_only=False)
               (weights_only=False because checkpoints hold plain Python
               objects such as history and metadata, not just tensors).
            2. self.model.load_state_dict(ckpt["model"])
            3. EMA: if self.ema and ckpt["ema"] is not None:
                   self.ema.module.load_state_dict(ckpt["ema"])
                   self.ema.n_averaged.fill_(1)
               AveragedModel copies the raw weights on its FIRST update
               (when n_averaged == 0). Without fill_(1), the first step
               after resuming would overwrite the EMA you just loaded.
            4. If restore_optimizer: load optimizer state, and scheduler
               state when both the scheduler and ckpt["scheduler"] exist.
            5. Return ckpt (fit() reads epoch/best/history from it; test()
               writes test_metrics into it).
        """
        # TODO: implement.
        raise NotImplementedError
