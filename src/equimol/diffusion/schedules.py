from __future__ import annotations

from dataclasses import dataclass

import math

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar diffusion schedule tensors.

    Shapes:
        - betas: [T]
        - alphas: [T]
        - alpha_bars: [T]
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def linear_beta_schedule(
    num_timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> DiffusionSchedule:
    """Return a linear DDPM beta schedule.

    Shapes:
        - output.betas: [T]
        - output.alphas: [T]
        - output.alpha_bars: [T]

    Operation:
        - beta_t linearly interpolates from beta_start to beta_end
        - alpha_t = 1 - beta_t
        - alpha_bar_t = prod_{s <= t} alpha_s
    """
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
    if beta_start <= 0:
        raise ValueError(f"beta_start must be positive, got {beta_start}.")
    if beta_end <= 0:
        raise ValueError(f"beta_end must be positive, got {beta_end}.")
    if beta_start >= 1 or beta_end >= 1:
        raise ValueError("beta_start and beta_end must be less than 1.")
    if beta_start > beta_end:
        raise ValueError(
            f"beta_start must be <= beta_end, got {beta_start} > {beta_end}."
        )
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating point dtype, got {dtype}.")

    betas = torch.linspace(
        beta_start,
        beta_end,
        num_timesteps,
        device=device,
        dtype=dtype,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )


def cosine_beta_schedule(
    num_timesteps: int,
    *,
    s: float = 0.008,
    max_beta: float = 0.999,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> DiffusionSchedule:
    """Return a cosine DDPM beta schedule.

    Shapes:
        - output.betas: [T]
        - output.alphas: [T]
        - output.alpha_bars: [T]

    Operation:
        - define alpha_bar(t) with a shifted cosine curve
        - convert adjacent alpha_bar ratios into beta_t
        - clamp beta_t by max_beta for numerical stability
        - recompute alpha_bar_t = prod_{s <= t} alpha_s from clamped betas

    Equations:
        - t_i = i / T for i in {0, ..., T}
        - f(t) = cos^2(((t + s) / (1 + s)) * pi / 2)
        - alpha_bar_i = f(t_i) / f(t_0)
        - beta_i = 1 - alpha_bar_{i + 1} / alpha_bar_i
        - alpha_i = 1 - beta_i
        - alpha_bar_i = prod_{j <= i} alpha_j
    """
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
    if s <= 0:
        raise ValueError(f"s must be positive, got {s}.")
    if max_beta <= 0:
        raise ValueError(f"max_beta must be positive, got {max_beta}.")
    if max_beta >= 1:
        raise ValueError("max_beta must be less than 1.")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating point dtype, got {dtype}.")

    steps = torch.arange(num_timesteps + 1, device=device, dtype=dtype)
    t = steps / num_timesteps

    f_t = torch.square(torch.cos(((t + s) / (1 + s)) * math.pi / 2))
    alpha_bars = f_t / f_t[0]

    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    betas = betas.clamp(min=0.0, max=max_beta)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )
