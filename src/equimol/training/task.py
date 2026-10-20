from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Task(Protocol):
    """What the Trainer needs from a task. This file is the contract; it has
    no implementation on purpose.

    A Protocol is structural typing: any object with these two methods is a
    Task, no inheritance needed. That is what lets new tasks (and, later, new
    architectures) plug in without touching equimol itself.

    Examples (to be written in later steps):
        EnergyForceTask          loss = energy MSE + force L1, metrics in kcal/mol
        PropertyTask             loss = MSE on one QM9 target, metric = MAE
        CoordinateDenoisingTask  loss = noise-prediction MSE at random timesteps
    """

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Scalar training loss for one batch.

        The Trainer has already moved `batch` to the device and called
        model.train(); it calls backward() on the result. Do not call
        optimizer.step() or zero_grad() here.
        """
        ...

    def evaluate(self, model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
        """Validation/test metrics over a whole loader; lower is better for
        the metric the Trainer selects on.

        The Trainer has called model.eval() but does NOT disable autograd,
        so tasks can compute forces as -dE/dx. Use torch.no_grad() inside
        when gradients are not needed. Move batches with to_device(batch, device).
        """
        ...
