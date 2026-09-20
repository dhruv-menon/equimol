from __future__ import annotations

import torch
import torch.nn as nn

from equimol.layers.pooling import global_add_pool, global_mean_pool
from equimol.layers.vector_egnn import VectorEGNNBackbone


def _get_pooling(pooling: str):
    if pooling == "sum":
        return global_add_pool
    if pooling == "mean":
        return global_mean_pool
    else:
        raise ValueError(f"pooling type: {pooling} is not supported by equimol")


# ----------------------------------------
# A scalar graph regressor built on a VectorEGNN backbone.
#   - Predicts one scalar per graph from invariant atom features, coordinates,
#     and learned l = 1 vector hidden states.
#   - Useful for conservative energy-force models where forces are obtained by
#     differentiating the scalar prediction with respect to coordinates.
#
# Shapes:
#    h: [N, F] input atom/node scalar features
#    x: [N, D] coordinates
#    edge_index: [2, E] directed graph edges
#    edge_attr: [E, A] optional invariant edge features
#    batch: [N] graph id per node, or None for one graph
#    encoded h: [N, H] hidden scalar node states after node_encoder
#    v: [N, D, V] learned vector hidden states inside VectorEGNNBackbone
#    updated_h: [N, H] hidden scalar node states after VectorEGNNBackbone
#    graph_state: [B, H] pooled graph representations
#    output: [B] scalar graph predictions
#
# The model is invariant
#   - Vector channels are equivariant because they are built from scalar-weighted
#     relative directions. Vector-to-scalar feedback uses vector norms, which are
#     rotation invariant. The readout pools only scalar node states, so graph
#     predictions are invariant to translation, rotation, and node permutation
#     when edge_index is transformed consistently.
#
# Complexity:
#    - O(LE(M + DV)) message passing for L layers, E edges, message width M,
#      coordinate dimension D, and vector width V.
# ----------------------------------------


class VectorEGNNRegressor(nn.Module):
    def __init__(self, 
                 node_feat_dim: int,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 edge_attr_dim: int = 0,
                 message_dim: int = 128,
                 vector_dim: int = 64,
                 attention: bool = True,
                 attention_dim: int = 128,
                 residual: bool = True,
                 update_coords: bool = False,
                 dropout: float = 0.0,
                 coord_step_size: float = 0.1, 
                 pooling: str = "sum",
                 eps: float = 1e-8) -> None:
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.vector_dim = vector_dim
        self.egnn = VectorEGNNBackbone(
            num_layers = num_layers,
            hidden_dim = hidden_dim,
            edge_attr_dim = edge_attr_dim,
            message_dim = message_dim,
            vector_dim = vector_dim,
            attention = attention,
            attention_dim = attention_dim,
            residual = residual,
            update_coords = update_coords,
            dropout = dropout,
            coord_step_size = coord_step_size,
            eps = eps)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.pool = _get_pooling(pooling = pooling)

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor | None = None,
                edge_attr: torch.Tensor | None = None,
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
        v = torch.zeros(h.size(0), x.size(1), self.vector_dim, device=h.device, dtype=h.dtype)
        h, _, _ = self.egnn(h = h,
                            x = x,
                            v = v,
                            edge_index = edge_index,
                            edge_attr = edge_attr)
        graph_state = self.pool(h, batch)
        return self.readout(graph_state).squeeze(-1)
