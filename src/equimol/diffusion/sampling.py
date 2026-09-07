from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .corruption import center_coordinates, sample_coordinate_noise
from .schedules import DiffusionSchedule


def _node_timesteps(
    t: torch.Tensor,
    batch: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    if t.ndim > 1:
        raise ValueError(f"t must have shape [] or [B], got {tuple(t.shape)}")
    if t.ndim == 0:
        return t.reshape(1).expand(num_nodes)
    if t.numel() == 1:
        return t.expand(num_nodes)
    if batch is None:
        raise ValueError("graph-wise timesteps require batch")

    max_batch = int(batch.max().item()) if batch.numel() > 0 else -1
    if t.numel() <= max_batch:
        raise ValueError(
            "graph-wise timesteps must cover all batch indices, "
            f"got {t.numel()} timesteps for max batch index {max_batch}"
        )
    return t[batch]


def _validate_schedule(schedule: DiffusionSchedule) -> None:
    if schedule.betas.ndim != 1:
        raise ValueError(
            f"schedule.betas must have shape [T], got {tuple(schedule.betas.shape)}"
        )
    if schedule.betas.numel() == 0:
        raise ValueError("schedule must contain at least one timestep")
    if schedule.alphas.shape != schedule.betas.shape:
        raise ValueError("schedule.alphas must match schedule.betas shape")
    if schedule.alpha_bars.shape != schedule.betas.shape:
        raise ValueError("schedule.alpha_bars must match schedule.betas shape")
    if not (
        schedule.betas.dtype.is_floating_point
        and schedule.alphas.dtype.is_floating_point
        and schedule.alpha_bars.dtype.is_floating_point
    ):
        raise TypeError("schedule tensors must use a floating point dtype")


def _validate_graph_inputs(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor | None,
    edge_attr: torch.Tensor | None,
) -> None:
    if h.ndim != 2:
        raise ValueError(f"h must have shape [N, F], got {tuple(h.shape)}")
    if h.shape[0] == 0:
        raise ValueError("h must contain at least one node")
    if not torch.is_floating_point(h):
        raise TypeError(f"h must be floating point, got {h.dtype}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
        )
    if edge_index.dtype != torch.long:
        raise TypeError(
            f"edge_index must have dtype torch.long, got {edge_index.dtype}"
        )
    if edge_index.device != h.device:
        raise ValueError("edge_index must be on the same device as h")
    if edge_index.numel() > 0 and (
        edge_index.min() < 0 or edge_index.max() >= h.shape[0]
    ):
        raise ValueError("edge_index contains node indices outside [0, N)")

    if batch is not None:
        if batch.ndim != 1:
            raise ValueError(f"batch must have shape [N], got {tuple(batch.shape)}")
        if batch.shape[0] != h.shape[0]:
            raise ValueError(
                f"Expected batch with length {h.shape[0]}, got {batch.shape[0]}"
            )
        if batch.device != h.device:
            raise ValueError("batch must be on the same device as h")
        if batch.dtype != torch.long:
            raise TypeError(f"batch must have dtype torch.long, got {batch.dtype}")
        if batch.numel() > 0 and batch.min() < 0:
            raise ValueError("batch indices must be non-negative")

    if edge_attr is not None:
        if edge_attr.ndim != 2:
            raise ValueError(
                f"edge_attr must have shape [E, A], got {tuple(edge_attr.shape)}"
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Expected edge_attr with {edge_index.shape[1]} edges, "
                f"got {edge_attr.shape[0]}"
            )
        if edge_attr.device != h.device:
            raise ValueError("edge_attr must be on the same device as h")


def p_sample_coordinates_step(
    model: nn.Module,
    h: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    edge_index: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    *,
    center: bool = True,
) -> torch.Tensor:
    """Sample one reverse DDPM coordinate step.

    Objective:
        Move noisy coordinates one step backward from x_t to x_{t-1} using an
        epsilon-prediction coordinate denoiser.

    Shapes:
        - h: [N, F]
        - x_t: [N, D]
        - t: [] or [B]
        - schedule.betas/alphas/alpha_bars: [T]
        - edge_index: [2, E]
        - batch: [N] or None
        - edge_attr: [E, A] optional
        - output: [N, D]

    Equations:
        - eps_hat = model(h, x_t, t, edge_index, batch, edge_attr)
        - mu_t = 1 / sqrt(alpha_t) *
                 (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat)
        - x_{t-1} = mu_t + sqrt(beta_t) z, where z ~ N(0, I)
        - if t = 0, x_{t-1} = mu_t

    """
    _validate_graph_inputs(h, edge_index, batch, edge_attr)
    _validate_schedule(schedule)
    if x_t.ndim != 2:
        raise ValueError(f"x_t must have shape [N, D], got {tuple(x_t.shape)}")
    if x_t.shape[0] != h.shape[0]:
        raise ValueError(f"Expected x_t with {h.shape[0]} nodes, got {x_t.shape[0]}")
    if not torch.is_floating_point(x_t):
        raise TypeError(f"x_t must be floating point, got {x_t.dtype}")
    if x_t.device != h.device:
        raise ValueError("x_t must be on the same device as h")

    t = torch.as_tensor(t, device=h.device)
    if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise TypeError(f"t must contain integer timestep indices, got {t.dtype}")
    node_t = _node_timesteps(t.long(), batch, h.shape[0])
    if node_t.min() < 0 or node_t.max() >= schedule.betas.shape[0]:
        raise ValueError("t contains timestep indices outside the schedule range")

    eps_hat = model(
        h,
        x_t,
        t.long(),
        edge_index,
        batch=batch,
        edge_attr=edge_attr,
    )
    if eps_hat.shape != x_t.shape:
        raise ValueError(
            f"Expected model output with shape {tuple(x_t.shape)}, "
            f"got {tuple(eps_hat.shape)}"
        )

    betas = schedule.betas.to(device=x_t.device, dtype=x_t.dtype)[node_t].unsqueeze(
        -1
    )
    alphas = schedule.alphas.to(device=x_t.device, dtype=x_t.dtype)[node_t].unsqueeze(
        -1
    )
    alpha_bars = schedule.alpha_bars.to(device=x_t.device, dtype=x_t.dtype)[
        node_t
    ].unsqueeze(-1)

    denom = (1.0 - alpha_bars).clamp_min(torch.finfo(x_t.dtype).eps).sqrt()
    mean = alphas.rsqrt() * (x_t - betas / denom * eps_hat)
    has_noise = (node_t > 0).to(dtype=x_t.dtype).unsqueeze(-1)
    noise = sample_coordinate_noise(x_t, batch=batch, center=center)
    x_prev = mean + has_noise * betas.sqrt() * noise
    return center_coordinates(x_prev, batch=batch) if center else x_prev


@torch.no_grad()
def sample_coordinates_loop(
    model: nn.Module,
    h: torch.Tensor,
    edge_index: torch.Tensor,
    schedule: DiffusionSchedule,
    batch: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    *,
    coord_dim: int = 3,
    center: bool = True,
) -> torch.Tensor:
    """Run the full reverse coordinate sampling loop.

    Objective:
        Generate coordinates from Gaussian noise conditioned on fixed node
        features and graph structure.

    Shapes:
        - h: [N, F]
        - edge_index: [2, E]
        - schedule.betas/alphas/alpha_bars: [T]
        - batch: [N] or None
        - edge_attr: [E, A] optional
        - output: [N, D]

    Equations:
        - x_T ~ N(0, I)
        - for t = T - 1, ..., 0:
            x_{t-1} = p_sample_coordinates_step(model, h, x_t, t, ...)
        - return x_0

    """
    _validate_graph_inputs(h, edge_index, batch, edge_attr)
    _validate_schedule(schedule)
    if coord_dim <= 0:
        raise ValueError(f"coord_dim must be positive, got {coord_dim}")

    x_t = torch.randn((h.shape[0], coord_dim), device=h.device, dtype=h.dtype)
    if center:
        x_t = center_coordinates(x_t, batch=batch)

    for timestep in reversed(range(schedule.betas.shape[0])):
        t = torch.tensor(timestep, device=h.device, dtype=torch.long)
        x_t = p_sample_coordinates_step(
            model,
            h,
            x_t,
            t,
            schedule,
            edge_index,
            batch=batch,
            edge_attr=edge_attr,
            center=center,
        )

    return x_t
