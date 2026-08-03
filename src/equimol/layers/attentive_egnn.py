from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from equimol.utils import segment_sum
from equimol.layers.attention import InvariantEdgeAttention
from equimol.layers.distance import PairwiseDistance
# ----------------------------------------
# An EGNN layer with edge attention
#   - In the vanilla EGNN, each edge is assigned an equal weight.
#     This implies that each edge is equally important to updating the node state.
#     However, in realistic cases, some edges carry more meaningful information.
#     In edge attention, we calculate attention scores that weights the edge contributions during message construction.
# For detailed documentation on tensor shapes & memory requirements, 
# Refer to .egnn AND/OR .attention
# ----------------------------------------

class AttentiveEGNNLayer(nn.Module):
    """Single E(n)-equivariant message passing with edge attention"""
    def __init__(self,
                 hidden_dim: int,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 attention_dim: int = 128,
                 residual: bool = True,
                 dropout: float = 0.0, 
                 coord_step_size: float = 0.1, 
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.residual = residual 
        self.coord_step_size = coord_step_size
        self.eps = eps
        self.squared_distance = PairwiseDistance(squared = True, eps = eps)
        self.distance = PairwiseDistance(squared = False, eps = eps)

        # compile the edge input dim
        edge_input = 2 * hidden_dim + 1 + edge_attr_dim

        # MLP for the edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, message_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim),
            nn.SiLU())

        # edge attention
        self.edge_attention = InvariantEdgeAttention(
            hidden_dim = hidden_dim,
            edge_attr_dim = edge_attr_dim,
            attention_dim = attention_dim,
            dropout = dropout)

        # MLP for coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, 1, bias = False))

        # MLP for node update 
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
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
                edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        src, dst = edge_index # [E]
        h_src = h[src] # [E, hidden_dim]
        h_dst = h[dst] # [E, hidden_dim]
        x_src = x[src] # [E, 3]
        x_dst = x[dst] # [E, 3]

        relative = x_src - x_dst # [E, 3]
        radial = self.squared_distance(x, edge_index) # [E, 1]

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 0, device = h.device, dtype = h.dtype) # [E, 0]

        # ----- calculate edge messages -----
        edge_input = torch.cat([h_src, h_dst, radial, edge_attr], dim = -1) # [E, 2H + 1 + edge_attr_dim]
        message = self.edge_mlp(edge_input) # [E, message_dim]
        alpha = self.edge_attention(h = h, x = x, edge_index = edge_index, edge_attr = edge_attr) # [E, 1]
        weighted_message = message * alpha # [E, message_dim]

        # ----- update coords -----
        distance = self.distance(x, edge_index) # [E, 1]
        direction = relative / distance # [E, 3]
        coord_update = torch.tanh(self.coord_mlp(message)) # [E, 1]
        weighted_coord_update = alpha * coord_update * self.coord_step_size
        delta_x = torch.zeros_like(x) # [N, 3]
        delta_x.index_add_(0, dst, direction * weighted_coord_update) # [N, 3]
        x = x + delta_x # [N, 3]

        # ----- updated h -----
        aggregated = segment_sum(weighted_message, dst, h.size(0)) # [N, message_dim]
        node_input = torch.cat([h, aggregated], dim = -1) # [N, hidden_dim + message_dim]
        node_update = self.node_mlp(node_input) # [N, hidden_dim]
        h = h + node_update if self.residual else node_update
        return self.norm(h), x
    
