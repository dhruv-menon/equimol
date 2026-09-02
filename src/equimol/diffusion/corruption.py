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

    TODO:
        - validate x has shape [N, D]
        - if batch is None, subtract the mean over all nodes
        - if batch is provided, subtract each graph's node mean
        - preserve input dtype and device
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [N, D], instead got {tuple(x.shape)}")
    N, D = x.shape

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

    TODO:
        - sample noise with the same shape, dtype, and device as x
        - if center is true, center noise per graph using batch
        - keep this function independent of the diffusion schedule
    """
    raise NotImplementedError


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

    TODO:
        - validate x0, t, schedule, batch, and optional noise shapes
        - sample noise if noise is None
        - gather alpha_bar_t for each node from t and batch
        - broadcast scalar coefficients to [N, D]
        - optionally center x0 and noise before corruption
        - return both x_t and the epsilon target
    """
    raise NotImplementedError
