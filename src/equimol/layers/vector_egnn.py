from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from equimol.utils import segment_sum
from equimol.layers.attention import InvariantEdgeAttention
from equimol.layers.distance import PairwiseDistance

# ----------------------------------------
# An Vector EGNN layer with edge attention
#   - Implements the l = 1 irrep similar to the PaiNN style architecture.
#   - Can implement with or without attention based on the flag passed
# ----------------------------------------

class VectorEGNNLayer(nn.Module):
    """Single l = 1 vector E(n)-equivariant message passing with edge attention (if passed)"""
    def __init__(self,
                 hidden_dim: int, 
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 vector_dim: int = 64,
                 attention: Optional[bool] = True,
                 attention_dim: Optional[int] = 128,
                 vector_gate: bool = False,
                 update_coords: bool = True,
                 residual: bool = True,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1, 
                 eps: float = 1e-8
                 ) -> None:
        super().__init__()
        self.residual = residual
        self.coord_step_size = coord_step_size
        self.eps = eps
        self.squared_distance = PairwiseDistance(squared = True, eps = eps)
        self.distance = PairwiseDistance(squared = False, eps = eps)
        self.attention = attention
        self.vector_gate = vector_gate
        self.update_coords = update_coords

        # compile the edge input dim
        edge_input = 2 * hidden_dim + 1 + edge_attr_dim

        # MLP for the edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, message_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim),
            nn.SiLU())

        # Invariant edge attention if passed
        if self.attention:
            if not attention_dim:
                attention_dim = message_dim
            self.edge_attention = InvariantEdgeAttention(
                hidden_dim = hidden_dim,
                edge_attr_dim = edge_attr_dim,
                attention_dim = attention_dim,
                dropout = dropout)

        # MLP for coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, 1, bias = False)
        )

        # MLP for vector message
        self.vector_message = nn.Sequential(
            nn.Linear(message_dim, vector_dim),
            nn.SiLU(),
            nn.Linear(vector_dim, vector_dim)
        )

        # MLP for vector update
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, vector_dim),
            nn.SiLU(),
            nn.Linear(vector_dim, vector_dim)
        )

        if vector_gate:
            self.vector_gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, vector_dim),
                nn.Sigmoid(),
            )

        # MLP for node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim + vector_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU())

        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.coord_mlp[0].weight, gain = 0.5)
        nn.init.zeros_(self.coord_mlp[0].bias)
        nn.init.xavier_uniform_(self.coord_mlp[2].weight, gain = 0.01)

    def forward(self, 
                h: torch.Tensor,
                x: torch.Tensor,
                v: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''One EGNN update

        Args: - h: Invariant node states with shape [N, H]
              - x: Coordinates with shape [N, D]
              - v: Vector hidden states with shape [N, 3, V]
              - edge_index: Directed edges with shape [2, E]
              - edge_attr: Optional invariant edge features with shape [E, A]
        
        Returns:
            Updated (h, v, x) with shapes [N, H], [N, D, V] and [N, D]'''

        # collect src and dst nodes
        src, dst = edge_index # src: [E]; dst: [E]
        rel = x[src] - x[dst] # rel: [E, D]

        # calculate distance (scalar) and direction (vector)
        radial = self.squared_distance(x, edge_index) # radial : [E, 1]
        dist = self.distance(x, edge_index) # [E, 1]
        direction = rel / (dist + self.eps) # [E, D]

        # prepare edge_attr
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 0, device = h.device, dtype = h.dtype) # [E, 0]

        # prepare edge_input & calculate message
        edge_input = torch.cat([h[src], h[dst], radial, edge_attr], dim = -1) # [E, 2H + 1 + A]
        message = self.edge_mlp(edge_input) # [E, M]

        if self.attention:
            alpha = self.edge_attention(h = h, x = x, edge_index = edge_index, edge_attr = edge_attr) # [E, 1]
            weighted_message = message * alpha

        # update coords
        if self.update_coords:
            coord_update = torch.tanh(self.coord_mlp(message)) * self.coord_step_size # [E, 1]
            delta_x = torch.zeros_like(x) # [N, D]

            if self.attention:
                weighted_coord_update = alpha * coord_update
                delta_x.index_add_(0, dst, direction * weighted_coord_update)
            else:
                delta_x.index_add_(0, dst, direction * coord_update) # [N, D]

            x = x + delta_x # [N, D]

        # scalar aggregation
        if self.attention:
            aggregated_scalar_message = segment_sum(weighted_message, dst, h.size(0)) # [N, M]
        else:
            aggregated_scalar_message = segment_sum(message, dst, h.size(0)) # [N, M]
        
        # calculate vector message
        if self.attention:
            vector_message = self.vector_message(weighted_message) # [E, V]
        else:
            vector_message = self.vector_message(message) # [E, V]
        vector_weight = direction.unsqueeze(-1) * vector_message.unsqueeze(1)
        aggregated_vector_message = torch.zeros_like(v) # [N, D, V]
        aggregated_vector_message.index_add_(0, dst, vector_weight)

        delta_v = self.vector_mlp(v)
        v = v + delta_v + aggregated_vector_message if self.residual else delta_v + aggregated_vector_message # [N, D, V]
        if self.vector_gate:
            v = v * self.vector_gate_mlp(h).unsqueeze(1) # [N, D, V]

        vector_to_scalar = torch.linalg.vector_norm(v, dim = 1) # [N, V]

        delta_h = self.node_mlp(torch.cat([h, aggregated_scalar_message, vector_to_scalar], dim = -1)) # [N, M + H + V]
        h = h + delta_h if self.residual else delta_h # [N, H]
        return self.norm(h), v, x


class VectorEGNNBackbone(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 vector_dim: int = 64,
                 attention: bool = True,
                 attention_dim: int = 128,
                 vector_gate: bool = False,
                 update_coords: bool = True,
                 residual: bool = True,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1,
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorEGNNLayer(hidden_dim = hidden_dim,
                             edge_attr_dim = edge_attr_dim,
                             message_dim = message_dim,
                             vector_dim = vector_dim,
                             attention = attention,
                             attention_dim = attention_dim,
                             vector_gate = vector_gate,
                             update_coords = update_coords,
                             residual = residual,
                             dropout = dropout,
                             coord_step_size = coord_step_size,
                             eps = eps)
                             for _ in range(num_layers)]
        )

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                v: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h, v, x = layer(h = h,
                            x = x,
                            v = v,
                            edge_index = edge_index,
                            edge_attr = edge_attr)
        return h, v, x
     
