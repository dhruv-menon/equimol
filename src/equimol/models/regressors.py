
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from equimol.layers.pooling import global_add_pool, global_mean_pool
from equimol.models.backbones import EGNNBackbone, AttentiveEGNNBackbone

# ----------------------------------------
# A regressor built on an EGNN backbone that supports a single readout.
#   - Predicts one scalar per graph from invariant atom features and equivariant
#     coordinates
# 
# Shapes:
#    h: [N, F] input atom/node scalar features
#    x: [N, D] coordinates
#    edge_index: [2, E] directed graph edges
#    edge_attr: [E, A] optional invariant edge features
#    batch: [N] graph id per node, or None for one graph
#    encoded h: [N, H] hidden node states after node_encoder
#    updated_h: [N, H] hidden node states after EGNNBackbone
#    graph_state: [B, H] pooled graph representations
#    output: [B] scalar graph predictions
#
# The model is invariant
#   - EGNN layers keep node states invariant when inputs are invariant scalars.
#     Sum pooling is permutation invariant within each graph. The readout sees no
#     absolute coordinates, so graph predictions are invariant to E(n) transforms.
#
# Complexity:
#    - O(LE) message passing for L layers and E edges.
# ----------------------------------------

def _get_pooling(pooling: str):
    if pooling == "sum":
        return global_add_pool
    elif pooling == "mean":
        return global_mean_pool
    else:
        raise ValueError(f"pooling type: {pooling} is not supported by equimol")


class EGNNRegressor(nn.Module):
    def __init__(self,
                 node_feat_dim: int,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 residual: bool = True,
                 update_coords: bool = True,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1,
                 pooling: str = "sum",
                 eps: float = 1e-8) -> None:
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.egnn = EGNNBackbone(num_layers = num_layers,
                                 hidden_dim = hidden_dim,
                                 edge_attr_dim = edge_attr_dim,
                                 message_dim = message_dim,
                                 residual = residual,
                                 update_coords = update_coords,
                                 dropout = dropout,
                                 coord_step_size = coord_step_size,
                                 eps = eps) 
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.pool = _get_pooling(pooling = pooling)

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        """Predict one scalar per graph.

        Args:
            h: Input invariant node features with shape [N, F]
            x: Coordinates with shape [N, 3]
            edge_index: Directed edges with shape [2, E]
            batch: Optional graph ids with shape [N]
            edge_attr: Optional invariant edge features with shape [E, A]

        Returns:
            Tensor with shape [B]"""

        h = self.node_encoder(h.float())
        x = x.float()
        updated_h, _ = self.egnn(h = h,
                                 x = x,
                                 edge_index = edge_index,
                                 edge_attr = edge_attr)
        graph_state = self.pool(updated_h, batch)
        return self.readout(graph_state).squeeze(-1)


# ----------------------------------------
# A regressor built on an attentive EGNN backbone.
#   - Predicts one scalar per graph using invariant edge attention inside each
#     coordinate-aware message passing layer.
#
# Shapes:
#    h: [N, F] input atom/node scalar features
#    x: [N, D] coordinates
#    edge_index: [2, E] directed graph edges
#    edge_attr: [E, A] optional invariant edge features
#    batch: [N] graph id per node, or None for one graph
#    encoded h: [N, H] hidden node states after node_encoder
#    updated_h: [N, H] hidden node states after AttentiveEGNNBackbone
#    graph_state: [B, H] pooled graph representations
#    output: [B] scalar graph predictions
#
# The model is invariant
#   - Attention scores depend on scalar node states, invariant edge features,
#     and pairwise squared distances. Coordinate updates are weighted sums of
#     relative vectors. The graph readout pools invariant hidden states, so the
#     final scalar prediction is invariant to translation, rotation, and node
#     permutation when edge_index is transformed consistently.
#
# Complexity:
#    - O(LE) message passing and attention scoring for L layers and E edges.
# ----------------------------------------

class AttentiveEGNNRegressor(nn.Module):
    def __init__(self,
                 node_feat_dim: int,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 attention_dim: int = 128,
                 residual: bool = True,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1,
                 pooling: str = "sum",
                 eps: float = 1e-8) -> None:
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.egnn = AttentiveEGNNBackbone(num_layers = num_layers,
                                          hidden_dim = hidden_dim,
                                          edge_attr_dim = edge_attr_dim,
                                          message_dim = message_dim,
                                          attention_dim = attention_dim,
                                          residual = residual,
                                          dropout = dropout,
                                          coord_step_size = coord_step_size,
                                          eps = eps)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.pool = _get_pooling(pooling = pooling)

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        """Predict one scalar per graph.

        Args:
            h: Input invariant node features with shape [N, F]
            x: Coordinates with shape [N, 3]
            edge_index: Directed edges with shape [2, E]
            batch: Optional graph ids with shape [N]
            edge_attr: Optional invariant edge features with shape [E, A]

        Returns:
            Tensor with shape [B]"""

        h = self.node_encoder(h.float())
        x = x.float()
        updated_h, _ = self.egnn(h = h,
                                 x = x,
                                 edge_index = edge_index,
                                 edge_attr = edge_attr)
        graph_state = self.pool(updated_h, batch)
        return self.readout(graph_state).squeeze(-1)
