from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainerConfig:
    """Loop settings that are the same for every task.

    Kept as a plain dataclass so it can later be built from YAML
    (`equimol train config.yaml`) and saved into checkpoints with
    dataclasses.asdict(). Model, optimizer and task settings do NOT live
    here; they belong to their own configs.

    Attributes:
        epochs: Total epochs to train (also the target when resuming).
        checkpoint_path: Best checkpoint path; the last one goes to
            <stem>.last.pt next to it.
        eval_every: Validate every N epochs, and always after the last epoch.
        select_metric: Key of Task.evaluate()'s dict used to pick the best
            checkpoint and drive ReduceLROnPlateau. Lower is better.
        grad_clip: Max global gradient norm, or None for no clipping.
        ema_decay: Exponential-moving-average decay of the weights in (0, 1),
            or None to disable. Typical: 0.99-0.999.
    """

    epochs: int
    checkpoint_path: str | Path
    eval_every: int = 1
    select_metric: str = "loss"
    grad_clip: float | None = None
    ema_decay: float | None = None

    def __post_init__(self) -> None:
        """Validate eagerly, so a bad config fails before any data loads.

        Raise ValueError (with the offending value in the message) when:
            epochs <= 0
            eval_every <= 0
            grad_clip is not None and grad_clip <= 0
            ema_decay is not None and not 0 < ema_decay < 1

        Also normalise checkpoint_path to a Path here, so the Trainer can rely
        on it. (Tests: test_trainer_config_rejects_invalid_values)
        """
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"if grad_clip is enabled, it must be positive, got {self.grad_clip}")
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError(f"if ema_decay is enabled, it must be in (0, 1), got {self.ema_decay}")
        self.checkpoint_path = Path(self.checkpoint_path)


@dataclass
class FitResult:
    """What Trainer.fit() returns.

    Attributes:
        best_metric: Best (lowest) value of select_metric seen in validation.
        best_epoch: Epoch at which best_metric was reached.
        history: One dict per validation, in order:
            {"epoch": int, "train_loss": float, **metrics_from_task_evaluate}
    """

    best_metric: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)
