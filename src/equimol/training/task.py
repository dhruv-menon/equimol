from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Task(Protocol):
    """Interface between the Trainer and a learning problem.

    Any object with these two methods is a Task (structural typing), so new
    tasks plug in without subclassing anything in equimol.
    """

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Scalar training loss for one batch.

        The Trainer has already moved `batch` to the device and called
        model.train(); it calls backward() on the result. Do not call
        optimizer.step() or zero_grad() here.
        """
        ...

    def evaluate(self, model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
        """Validation/test metrics over a whole loader; lower is better for the
        metric the Trainer selects on.

        The Trainer has called model.eval() but does not disable autograd, so
        tasks can compute forces as -dE/dx. Use torch.no_grad() inside when
        gradients are not needed, and move batches with to_device(batch, device).
        """
        ...
