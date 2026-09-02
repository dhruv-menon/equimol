import torch
import pytest

from equimol.diffusion import (
    DiffusionSchedule,
    center_coordinates,
    cosine_beta_schedule,
    linear_beta_schedule,
    q_sample_coordinates,
    sample_coordinate_noise,
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


def test_center_coordinates_centers_single_graph():
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ]
    )

    centered = center_coordinates(x)

    assert centered.shape == x.shape
    assert torch.allclose(centered.mean(dim=0), torch.zeros(3))


def test_center_coordinates_centers_each_graph_with_non_contiguous_batch_ids():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [10.0, 4.0],
            [14.0, 8.0],
        ]
    )
    batch = torch.tensor([0, 0, 2, 2])

    centered = center_coordinates(x, batch=batch)

    assert torch.allclose(centered[batch == 0].mean(dim=0), torch.zeros(2))
    assert torch.allclose(centered[batch == 2].mean(dim=0), torch.zeros(2))


def test_center_coordinates_validates_inputs():
    x = torch.randn(4, 3)

    with pytest.raises(ValueError, match="shape"):
        center_coordinates(torch.randn(4, 3, 1))

    with pytest.raises(ValueError, match="length"):
        center_coordinates(x, batch=torch.tensor([0, 0]))

    with pytest.raises(TypeError, match="torch.long"):
        center_coordinates(x, batch=torch.tensor([0.0, 0.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="non-negative"):
        center_coordinates(x, batch=torch.tensor([0, 0, 1, -1]))


def test_sample_coordinate_noise_matches_input_shape_dtype_and_device():
    x = torch.randn(6, 3, dtype=torch.float64)

    noise = sample_coordinate_noise(x, center=False)

    assert noise.shape == x.shape
    assert noise.dtype == x.dtype
    assert noise.device == x.device


def test_sample_coordinate_noise_can_center_per_graph():
    x = torch.randn(5, 3)
    batch = torch.tensor([0, 0, 1, 1, 1])

    noise = sample_coordinate_noise(x, batch=batch, center=True)

    assert torch.allclose(noise[batch == 0].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert torch.allclose(noise[batch == 1].mean(dim=0), torch.zeros(3), atol=1e-6)


def test_q_sample_coordinates_applies_forward_equation_for_scalar_timestep():
    schedule = DiffusionSchedule(
        betas=torch.tensor([0.0, 0.75]),
        alphas=torch.tensor([1.0, 0.25]),
        alpha_bars=torch.tensor([1.0, 0.25]),
    )
    x0 = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    noise = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    x_t, target_noise = q_sample_coordinates(
        x0,
        torch.tensor(1),
        schedule,
        noise=noise,
        center=False,
    )

    expected = 0.5 * x0 + torch.sqrt(torch.tensor(0.75)) * noise
    assert torch.allclose(x_t, expected)
    assert torch.equal(target_noise, noise)


def test_q_sample_coordinates_uses_graph_wise_timesteps():
    schedule = DiffusionSchedule(
        betas=torch.tensor([0.0, 0.75]),
        alphas=torch.tensor([1.0, 0.25]),
        alpha_bars=torch.tensor([1.0, 0.25]),
    )
    x0 = torch.tensor([[2.0, 0.0], [0.0, 4.0], [6.0, 8.0]])
    noise = torch.ones_like(x0)
    batch = torch.tensor([0, 0, 1])
    t = torch.tensor([0, 1])

    x_t, _ = q_sample_coordinates(
        x0,
        t,
        schedule,
        batch=batch,
        noise=noise,
        center=False,
    )

    expected_graph_0 = x0[:2]
    expected_graph_1 = 0.5 * x0[2:] + torch.sqrt(torch.tensor(0.75)) * noise[2:]

    assert torch.allclose(x_t[:2], expected_graph_0)
    assert torch.allclose(x_t[2:], expected_graph_1)


def test_q_sample_coordinates_centers_outputs_when_requested():
    schedule = DiffusionSchedule(
        betas=torch.tensor([0.5]),
        alphas=torch.tensor([0.5]),
        alpha_bars=torch.tensor([0.5]),
    )
    x0 = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 0.0, 0.0],
            [14.0, 2.0, 4.0],
        ]
    )
    noise = torch.ones_like(x0)
    batch = torch.tensor([0, 0, 1, 1])

    x_t, target_noise = q_sample_coordinates(
        x0,
        torch.tensor([0, 0]),
        schedule,
        batch=batch,
        noise=noise,
        center=True,
    )

    assert torch.allclose(x_t[batch == 0].mean(dim=0), torch.zeros(3))
    assert torch.allclose(x_t[batch == 1].mean(dim=0), torch.zeros(3))
    assert torch.allclose(target_noise, torch.zeros_like(noise))


def test_q_sample_coordinates_validates_inputs():
    schedule = linear_beta_schedule(3)
    x0 = torch.randn(4, 3)

    with pytest.raises(ValueError, match="x0"):
        q_sample_coordinates(torch.randn(4, 3, 1), torch.tensor(0), schedule)

    with pytest.raises(ValueError, match="noise"):
        q_sample_coordinates(x0, torch.tensor(0), schedule, noise=torch.randn(4, 2))

    with pytest.raises(ValueError, match="batch"):
        q_sample_coordinates(x0, torch.tensor([0, 1]), schedule)

    with pytest.raises(ValueError, match="cover all batch"):
        q_sample_coordinates(
            x0,
            torch.tensor([0, 1]),
            schedule,
            batch=torch.tensor([0, 0, 2, 2]),
        )

    with pytest.raises(ValueError, match="outside"):
        q_sample_coordinates(x0, torch.tensor(3), schedule)
