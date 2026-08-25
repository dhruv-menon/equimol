import pytest
import torch

from equimol.layers import SinusoidalTimeEmbedding, TimestepEmbedding

pytestmark = pytest.mark.time


def test_sinusoidal_time_embedding_returns_expected_shape_for_vector_timesteps():
    layer = SinusoidalTimeEmbedding(embedding_dim=6)
    t = torch.tensor([0, 1, 2])

    out = layer(t)

    assert out.shape == torch.Size([3, 6])
    assert out.dtype == torch.float32


def test_sinusoidal_time_embedding_supports_scalar_timestep():
    layer = SinusoidalTimeEmbedding(embedding_dim=4)

    out = layer(torch.tensor(3))

    assert out.shape == torch.Size([1, 4])


def test_sinusoidal_time_embedding_pads_odd_embedding_dim():
    layer = SinusoidalTimeEmbedding(embedding_dim=5)

    out = layer(torch.tensor([0, 1]))

    assert out.shape == torch.Size([2, 5])
    assert torch.equal(out[:, -1], torch.zeros(2))


def test_sinusoidal_time_embedding_is_deterministic():
    layer = SinusoidalTimeEmbedding(embedding_dim=8)
    t = torch.tensor([1, 5, 10])

    first = layer(t)
    second = layer(t)

    assert torch.equal(first, second)


def test_sinusoidal_time_embedding_validates_inputs():
    with pytest.raises(ValueError, match="embedding_dim"):
        SinusoidalTimeEmbedding(embedding_dim=0)

    with pytest.raises(ValueError, match="max_period"):
        SinusoidalTimeEmbedding(embedding_dim=4, max_period=0)

    with pytest.raises(ValueError, match="shape"):
        SinusoidalTimeEmbedding(embedding_dim=4)(torch.zeros(2, 2))


def test_timestep_embedding_returns_expected_shape():
    layer = TimestepEmbedding(
        embedding_dim=8,
        hidden_dim=16,
        output_dim=12,
    )

    out = layer(torch.tensor([0, 1, 2]))

    assert out.shape == torch.Size([3, 12])


def test_timestep_embedding_defaults_output_dim_to_embedding_dim():
    layer = TimestepEmbedding(embedding_dim=8, hidden_dim=16)

    out = layer(torch.tensor([0, 1]))

    assert out.shape == torch.Size([2, 8])


def test_timestep_embedding_supports_gradients():
    layer = TimestepEmbedding(embedding_dim=8, hidden_dim=16)

    out = layer(torch.tensor([0, 1, 2])).sum()
    out.backward()

    assert all(param.grad is not None for param in layer.parameters())


def test_timestep_embedding_validates_dimensions():
    with pytest.raises(ValueError, match="embedding_dim"):
        TimestepEmbedding(embedding_dim=0, hidden_dim=16)

    with pytest.raises(ValueError, match="hidden_dim"):
        TimestepEmbedding(embedding_dim=8, hidden_dim=0)

    with pytest.raises(ValueError, match="output_dim"):
        TimestepEmbedding(embedding_dim=8, hidden_dim=16, output_dim=0)
