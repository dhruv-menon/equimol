
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from egnn.layers import EGNNLayer, global_add_pool

# ----------------------------------------
# A regressor built on an EGNN backbone that supports a single readout.
#   - Predicts one scalar per graph from invariant atom features and equivariant
#     coordinates
# 
# Shapes:
#    h: [N, F] atom/node scalar features
#    x: [N, D] coordinates
#    edge_index: [2, E] graph edges
#    output: [B] scalar graph predictions
#    batch: [N] graph id per node, or None for one graph.
#
# The model is invariant
#   - EGNN layers keep node states invariant when inputs are invariant scalars.
#     Sum pooling is permutation invariant within each graph. The readout sees no
#     absolute coordinates, so graph predictions are invariant to E(n) transforms.
#
# Complexity:
#    - O(LE) message passing for L layers and E edges.
# ----------------------------------------

class EGNNRegressor(nn.Module):
    def __init__(self,
                 node_feat_dim: int,
                 edge_attr_dim: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 message_dim: int = 128,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1,
                 eps: float = 1e-8) -> None:
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.layers = nn.ModuleList(
            [
                EGNNLayer(hidden_dim = hidden_dim,
                          edge_attr_dim = edge_attr_dim,
                          message_dim = message_dim,
                          dropout = dropout,
                          coord_step_size = coord_step_size,
                          eps = eps,)
                for _ in range(num_layers)
            ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Predict one scalar per graph.

        Args:
            h: Invariant node features with shape [N, F]
            x: Coordinates with shape [N, D]
            edge_index: Directed edges with shape [2, E]
            batch: Optional graph ids with shape [N]
            edge_attr: Optional invariant edge features with shape [E, A]

        Returns:
            Tensor with shape [B]"""

        h = self.node_encoder(h.float())
        x = x.float()
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        graph_state = global_add_pool(h, batch)
        return self.readout(graph_state).squeeze(-1)
