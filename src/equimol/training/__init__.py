"""Task-agnostic training loop.

The loop owns everything that is the same for every model and task:
batching onto the device, optimizer steps, gradient clipping, EMA weights,
LR scheduling, validation, best/last checkpoints, resume and testing.

A Task owns everything else: how to turn a batch into a loss, and how to
turn a loader into validation metrics. Energies, forces, properties and
diffusion noise all live in tasks, never in the Trainer.
"""

from .config import FitResult, TrainerConfig
from .device import to_device
from .task import Task
from .trainer import Trainer

__all__ = [
    "FitResult",
    "Task",
    "Trainer",
    "TrainerConfig",
    "to_device",
]
