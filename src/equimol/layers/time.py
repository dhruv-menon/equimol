from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embeddings.

    Shapes:
        - t: [] or [B]
        - output: [1, embedding_dim] or [B, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if max_period <= 0:
            raise ValueError(f"max_period must be positive, got {max_period}.")

        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal timestep embeddings."""
        if t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim != 1:
            raise ValueError(f"Expected t with shape [] or [B], got {tuple(t.shape)}.")

        t = t.float()
        half_dim = self.embedding_dim // 2

        if half_dim == 0:
            return torch.zeros((t.shape[0], 1), dtype=t.dtype, device=t.device)

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=t.dtype, device=t.device)
            / half_dim
        )
        angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        if self.embedding_dim % 2 != 0:
            padding = torch.zeros((t.shape[0], 1), dtype=t.dtype, device=t.device)
            embeddings = torch.cat([embeddings, padding], dim=-1)

        return embeddings

class TimestepEmbedding(nn.Module):
    """Learned timestep embedding built on sinusoidal features.

    Shapes:
        - t: [] or [B]
        - output: [1, output_dim] or [B, output_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        output_dim = output_dim or embedding_dim

        self.sinusoidal_embedding = SinusoidalTimeEmbedding(embedding_dim=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return learned timestep embeddings."""
        return self.mlp(self.sinusoidal_embedding(t))
