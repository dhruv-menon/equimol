"""Internal helpers for graph construction."""

from __future__ import annotations

import torch


def empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty(2, 0, dtype=torch.long, device=device)
