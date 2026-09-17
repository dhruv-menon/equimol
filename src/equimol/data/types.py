from __future__ import annotations

import torch

from dataclasses import dataclass

@dataclass(frozen=True)
class GeometricBatch:
    h: torch.Tensor
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    edge_attr: torch.Tensor | None = None
    y: torch.Tensor | None = None
    force: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    bond_index: torch.Tensor | None = None
    angle_index: torch.Tensor | None = None
    torsion_index: torch.Tensor | None = None
    bond_lengths: torch.Tensor | None = None
    angle_features: torch.Tensor | None = None
    torsion_features: torch.Tensor | None = None
