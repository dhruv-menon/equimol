from __future__ import annotations

import torch

from .schedules import DiffusionSchedule


def center_coordinates(
    x: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center coordinates per graph.

    Objective:
        Remove graph-wise center of mass so diffusion does not learn arbitrary
        global translations.

    Shapes:
        - x: [N, D]
        - batch: [N] or None
        - output: [N, D]

    Equation:
        - mu_g = mean_{i in graph g} x_i
        - x_centered_i = x_i - mu_{batch_i}

    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [N, D], instead got {tuple(x.shape)}")
    N, D = x.shape
    if N == 0:
        raise ValueError("x must contain at least one node")

    if batch is None:
        return x - x.mean(dim=0, keepdim=True)

    if batch.ndim != 1:
        raise ValueError(
            f"Expected batch with shape [N], instead got {tuple(batch.shape)}"
        )
    if batch.shape[0] != N:
        raise ValueError(
            f"Expected batch with length {N}, instead got {batch.shape[0]}"
        )
    if batch.device != x.device:
        raise ValueError("batch must be on the same device as x")
    if batch.dtype != torch.long:
        raise TypeError(f"Expected batch dtype torch.long, got {batch.dtype}")
    if batch.numel() > 0 and batch.min() < 0:
        raise ValueError("batch indices must be non-negative")

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    sum_per_graph = torch.zeros((num_graphs, D), device=x.device, dtype=x.dtype)
    count_per_graph = torch.zeros((num_graphs, 1), device=x.device, dtype=x.dtype)
    ones_per_node = torch.ones((N, 1), device=x.device, dtype=x.dtype)

    sum_per_graph.index_add_(0, batch, x)
    count_per_graph.index_add_(0, batch, ones_per_node)

    mean_per_graph = sum_per_graph / count_per_graph.clamp_min(1.0)

    return x - mean_per_graph[batch]


def sample_coordinate_noise(
    x: torch.Tensor,
    batch: torch.Tensor | None = None,
    *,
    center: bool = True,
) -> torch.Tensor:
    """Sample Gaussian coordinate noise.

    Objective:
        Produce the epsilon target used by coordinate denoising models.

    Shapes:
        - x: [N, D]
        - batch: [N] or None
        - output: [N, D]

    Equation:
        - eps_i ~ N(0, I_D)
        - optional: eps_i = eps_i - mean_{j in graph batch_i} eps_j

    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [N, D], instead got {tuple(x.shape)}")

    noise = torch.randn_like(x)
    if center:
        return center_coordinates(noise, batch=batch)
    return noise


def q_sample_coordinates(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    batch: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    *,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample noisy coordinates from the forward diffusion process.

    Objective:
        Map clean coordinates x0 to noisy coordinates x_t and return the noise
        used to create them.

    Shapes:
        - x0: [N, D]
        - t: [] for one timestep or [B] for graph-wise timesteps
        - schedule.alpha_bars: [T]
        - batch: [N] or None
        - noise: [N, D] or None
        - output x_t: [N, D]
        - output noise: [N, D]

    Equation:
        - x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps

    """
    if x0.ndim != 2:
        raise ValueError(
            f"Expected x0 with shape [N, D], instead got {tuple(x0.shape)}"
        )
    N = x0.shape[0]
    if N == 0:
        raise ValueError("x0 must contain at least one node")
    if schedule.alpha_bars.ndim != 1:
        raise ValueError(
            "Expected schedule.alpha_bars with shape [T], "
            f"instead got {tuple(schedule.alpha_bars.shape)}"
        )
    if not schedule.alpha_bars.dtype.is_floating_point:
        raise TypeError("schedule.alpha_bars must use a floating point dtype")
    if noise is not None and noise.shape != x0.shape:
        raise ValueError(
            f"Expected noise with shape {tuple(x0.shape)}, got {tuple(noise.shape)}"
        )

    t = torch.as_tensor(t, device=x0.device)
    if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise TypeError(f"Expected integer timesteps, got {t.dtype}")
    t = t.long()

    if batch is not None:
        if batch.ndim != 1:
            raise ValueError(
                f"Expected batch with shape [N], instead got {tuple(batch.shape)}"
            )
        if batch.shape[0] != N:
            raise ValueError(
                f"Expected batch with length {N}, instead got {batch.shape[0]}"
            )
        if batch.device != x0.device:
            raise ValueError("batch must be on the same device as x0")
        if batch.dtype != torch.long:
            raise TypeError(f"Expected batch dtype torch.long, got {batch.dtype}")
        if batch.numel() > 0 and batch.min() < 0:
            raise ValueError("batch indices must be non-negative")

    if t.ndim == 0:
        node_t = t.expand(N)
    elif t.ndim == 1 and t.numel() == 1:
        node_t = t.expand(N)
    elif t.ndim == 1 and batch is not None:
        max_batch = int(batch.max().item()) if batch.numel() > 0 else -1
        if t.numel() <= max_batch:
            raise ValueError(
                "Expected graph-wise timesteps to cover all batch indices, "
                f"got {t.numel()} timesteps for max batch index {max_batch}"
            )
        node_t = t[batch]
    else:
        raise ValueError("Expected t with shape [] or [B] when batch is provided")

    if node_t.min() < 0 or node_t.max() >= schedule.alpha_bars.shape[0]:
        raise ValueError("t contains timestep indices outside the schedule range")

    x_start = center_coordinates(x0, batch=batch) if center else x0
    eps = noise if noise is not None else sample_coordinate_noise(x0, batch, center=False)
    eps = center_coordinates(eps, batch=batch) if center else eps

    alpha_bars = schedule.alpha_bars.to(device=x0.device, dtype=x0.dtype)
    alpha_bar_t = alpha_bars[node_t].unsqueeze(-1)
    x_t = alpha_bar_t.sqrt() * x_start + (1.0 - alpha_bar_t).sqrt() * eps

    return x_t, eps
