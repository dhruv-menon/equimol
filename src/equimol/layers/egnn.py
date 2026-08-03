from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from equimol.layers.distance import PairwiseDistance
from equimol.utils import segment_sum

# ----------------------------------------
# Core EGNN layer.
#    - Update invariant node features and equivariant coordinates on a graph
#
# Shapes:
#    h: [N, H] invariant scalar node states.
#    x: [N, D] coordinates.
#    edge_index: [2, E] directed edges; row 0 is source, row 1 is target.
#    edge_attr: optional [E, A] invariant edge features.
#
# Mathematics:
#    m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
#    x_j <- x_j + sum_i (x_i - x_j) / ||x_i - x_j|| * phi_x(m_ij)
#    h_j <- phi_h(h_j, sum_i m_ij)
#
# Why equivariant:
#   - Distances are invariant to translations and rotations. Coordinate updates
#     are scalar-weighted sums of relative vectors, so they transform exactly like
#     coordinates under E(n) transforms.
#
# Complexity:
#    - O(EH + ED) per layer. Dense graphs have E = N(N-1).
# ----------------------------------------


class EGNNLayer(nn.Module):
    '''Single E(n)-equivariant message-passing layer'''
    def __init__(self,
                 hidden_dim: int,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 residual: bool = True,
                 update_coords: bool = True,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1,
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.residual = residual
        self.update_coords = update_coords
        self.coord_step_size = coord_step_size
        self.eps = eps
        self.squared_distance = PairwiseDistance(squared = True, eps = eps)
        self.distance = PairwiseDistance(squared = False, eps = eps)

        # compile the edge input dim
        edge_input_dim = 2 * hidden_dim + 1 + edge_attr_dim

        # MLP for the edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
        )

        # MLP for the coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, 1, bias=False),
        )

        # MLP for updating node states
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.coord_mlp[0].weight, gain = 0.5)
        nn.init.zeros_(self.coord_mlp[0].bias)
        nn.init.xavier_uniform_(self.coord_mlp[2].weight, gain = 0.01)

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        '''One EGNN update

        Args: - h: Invariant node states with shape [N, H]
              - x: Coordinates with shape [N, D]
              - edge_index: Directed edges with shape [2, E]
              - edge_attr: Optional invariant edge features with shape [E, A]

        Returns:
            Updated (h, x) with shapes [N, H] and [N, D]'''

        # collect src & dst nodes
        src, dst = edge_index # src: [E], dst: [E]
        rel = x[src] - x[dst] # rel: [E, D]
        radial = self.squared_distance(x, edge_index) # radial: [E, 1]

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 0, device = h.device, dtype = h.dtype) # [E, 0]

        edge_input = torch.cat([h[src], h[dst], radial, edge_attr], dim = -1) # [E, 2H + 1 + A]
        messages = self.edge_mlp(edge_input) # [E, M]

        if self.update_coords:
            dist = self.distance(x, edge_index) # [E, 1]
            direction = rel / dist # [E, D]
            coord_weight = torch.tanh(self.coord_mlp(messages)) * self.coord_step_size # [E, 1]
            delta_x = torch.zeros_like(x) # [N, D]
            delta_x.index_add_(0, dst, direction * coord_weight) # [N, D]
            x = x + delta_x # [N, D]

        # Notice how segment_sum converts edge features into node features.
        
        aggregated = segment_sum(messages, dst, h.size(0)) # [N, M]
        delta_h = self.node_mlp(torch.cat([h, aggregated], dim = -1)) # [N, H]
        h = h + delta_h if self.residual else delta_h # [N, H]
        return self.norm(h), x
