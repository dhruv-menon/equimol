# EGNN backbones 
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from equimol.layers.egnn import EGNNLayer
from equimol.layers.attentive_egnn import AttentiveEGNNLayer

# ----- Vanilla EGNN backbone -----
class EGNNBackbone(nn.Module):
    """Stacked vanilla E(n)-equivariant message passing layers"""
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int, 
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 residual: bool = True, 
                 update_coords: bool = True, 
                 dropout: float = 0.0, 
                 coord_step_size: float = 0.1, 
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim = hidden_dim,
                       edge_attr_dim = edge_attr_dim,
                       message_dim = message_dim,
                       residual = residual, 
                       update_coords = update_coords,
                       dropout = dropout,
                       coord_step_size = coord_step_size,
                       eps = eps)
                       for _ in range(num_layers)]
            )

    def forward(self, 
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return h, x
    
# ----- Attentive EGNN backbone -----
class AttentiveEGNNBackbone(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int, 
                 edge_attr_dim: int = 0, 
                 message_dim: int = 128, 
                 attention_dim: int = 128,
                 residual: bool = True,
                 dropout: float = 0.0, 
                 coord_step_size: float = 0.1, 
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentiveEGNNLayer(hidden_dim = hidden_dim,
                                edge_attr_dim = edge_attr_dim,
                                message_dim = message_dim,
                                attention_dim = attention_dim,
                                residual = residual,
                                dropout = dropout, 
                                coord_step_size = coord_step_size,
                                eps = eps)
                                for _ in range(num_layers)]
        )

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h, x = layer(h = h, 
                         x = x, 
                         edge_index = edge_index, 
                         edge_attr = edge_attr)
        return h, x
