import torch
import pytest

from equimol.diffusion import (
    DiffusionSchedule,
    cosine_beta_schedule,
    linear_beta_schedule,
)

pytestmark = pytest.mark.diffusion


def test_linear_beta_schedule_returns_expected_shapes_and_values():
    schedule = linear_beta_schedule(
        4,
        beta_start=0.1,
        beta_end=0.4,
    )

    expected_betas = torch.tensor([0.1, 0.2, 0.3, 0.4])
    expected_alphas = 1.0 - expected_betas

    assert isinstance(schedule, DiffusionSchedule)
    assert torch.allclose(schedule.betas, expected_betas)
    assert torch.allclose(schedule.alphas, expected_alphas)
    assert torch.allclose(schedule.alpha_bars, torch.cumprod(expected_alphas, dim=0))


def test_linear_beta_schedule_preserves_device_and_dtype():
    schedule = linear_beta_schedule(
        3,
        dtype=torch.float64,
    )

    assert schedule.betas.dtype == torch.float64
    assert schedule.alphas.dtype == torch.float64
    assert schedule.alpha_bars.dtype == torch.float64


def test_linear_beta_schedule_alpha_bars_are_monotonic_decreasing():
    schedule = linear_beta_schedule(100)

    assert torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1])
    assert torch.all(schedule.alpha_bars > 0)
    assert torch.all(schedule.alpha_bars <= 1)


def test_linear_beta_schedule_validates_inputs():
    with pytest.raises(ValueError, match="num_timesteps"):
        linear_beta_schedule(0)

    with pytest.raises(ValueError, match="beta_start"):
        linear_beta_schedule(10, beta_start=0.0)

    with pytest.raises(ValueError, match="beta_end"):
        linear_beta_schedule(10, beta_end=0.0)

    with pytest.raises(ValueError, match="less than 1"):
        linear_beta_schedule(10, beta_start=1.0)

    with pytest.raises(ValueError, match="<="):
        linear_beta_schedule(10, beta_start=0.2, beta_end=0.1)

    with pytest.raises(TypeError, match="floating point"):
        linear_beta_schedule(10, dtype=torch.long)


def test_cosine_beta_schedule_returns_expected_shapes():
    schedule = cosine_beta_schedule(8)

    assert isinstance(schedule, DiffusionSchedule)
    assert schedule.betas.shape == (8,)
    assert schedule.alphas.shape == (8,)
    assert schedule.alpha_bars.shape == (8,)


def test_cosine_beta_schedule_preserves_device_and_dtype():
    schedule = cosine_beta_schedule(
        8,
        dtype=torch.float64,
    )

    assert schedule.betas.dtype == torch.float64
    assert schedule.alphas.dtype == torch.float64
    assert schedule.alpha_bars.dtype == torch.float64


def test_cosine_beta_schedule_has_valid_monotonic_values():
    schedule = cosine_beta_schedule(100)

    assert torch.all(schedule.betas >= 0)
    assert torch.all(schedule.betas < 1)
    assert torch.all(schedule.alphas > 0)
    assert torch.all(schedule.alphas <= 1)
    assert torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1])
    assert torch.all(schedule.alpha_bars > 0)
    assert torch.all(schedule.alpha_bars <= 1)


def test_cosine_beta_schedule_clamps_max_beta():
    schedule = cosine_beta_schedule(10, max_beta=0.5)

    assert torch.all(schedule.betas <= 0.5)
    assert torch.isclose(schedule.betas[-1], torch.tensor(0.5))


def test_cosine_beta_schedule_alpha_bars_match_clamped_betas():
    schedule = cosine_beta_schedule(20, max_beta=0.5)

    assert torch.allclose(schedule.alpha_bars, torch.cumprod(schedule.alphas, dim=0))


def test_cosine_beta_schedule_validates_inputs():
    with pytest.raises(ValueError, match="num_timesteps"):
        cosine_beta_schedule(0)

    with pytest.raises(ValueError, match="s"):
        cosine_beta_schedule(10, s=0.0)

    with pytest.raises(ValueError, match="max_beta"):
        cosine_beta_schedule(10, max_beta=0.0)

    with pytest.raises(ValueError, match="less than 1"):
        cosine_beta_schedule(10, max_beta=1.0)

    with pytest.raises(TypeError, match="floating point"):
        cosine_beta_schedule(10, dtype=torch.long)
