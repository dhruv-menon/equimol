from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3

from equimol.layers.irrep_egnn import IrrepEGNNBackbone
from equimol.layers.pooling import global_add_pool, global_mean_pool


def _get_pooling(pooling: str):
    if pooling == "sum":
        return global_add_pool
    if pooling == "mean":
        return global_mean_pool
    else:
        raise ValueError(f"pooling type: {pooling} is not supported by equimol")


def _scalar_slice(irreps: o3.Irreps) -> slice:
    start = 0
    stop = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l == 0:
            stop += width
        start += width
    return slice(0, stop)


# ----------------------------------------
# A scalar graph regressor built on an e3nn IrrepEGNN backbone.
#   - Predicts one scalar per graph from invariant atom features, coordinates,
#     and equivariant hidden states with configurable irreps.
#   - Useful for conservative energy-force models where forces are obtained by
#     differentiating the scalar prediction with respect to coordinates.
#
# Shapes:
#    h: [N, F] input atom/node scalar features
#    x: [N, 3] coordinates
#    edge_index: [2, E] directed graph edges
#    edge_attr: [E, A] optional invariant edge features
#    batch: [N] graph id per node, or None for one graph
#    encoded h: [N, H_irrep] node states in irreps_hidden
#    updated_h: [N, H_irrep] node states after IrrepEGNNBackbone
#    scalar_h: [N, H_0e] scalar 0e channels extracted for readout
#    graph_state: [B, H_0e] pooled graph representations
#    output: [B] scalar graph predictions
#
# The model is invariant
#   - IrrepEGNNBackbone propagates equivariant features using spherical
#     harmonics and e3nn tensor products. The readout uses only 0e scalar
#     channels and permutation-invariant graph pooling, so graph predictions are
#     invariant to translation, rotation, and node permutation when edge_index
#     is transformed consistently.
#
# Complexity:
#    - O(L * E * TP_cost) message passing for L layers and E edges, where
#      TP_cost depends on irreps_hidden, irreps_edge, and tensor product paths.
# ----------------------------------------


class IrrepEGNNRegressor(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        num_layers: int = 4,
        edge_attr_dim: int = 0,
        irreps_hidden: str = "64x0e + 32x1o + 16x2e",
        irreps_edge: str = "0e + 1o + 2e",
        radial_hidden_dim: int = 128,
        attention: bool = True,
        residual: bool = True,
        dropout: float = 0.0,
        pooling: str = "sum",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.scalar_slice = _scalar_slice(self.irreps_hidden)
        scalar_dim = self.scalar_slice.stop - self.scalar_slice.start

        if scalar_dim <= 0:
            raise ValueError("irreps_hidden must include at least one scalar 0e channel")

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.backbone = IrrepEGNNBackbone(
            num_layers=num_layers,
            irreps_hidden=irreps_hidden,
            irreps_edge=irreps_edge,
            edge_attr_dim=edge_attr_dim,
            radial_hidden_dim=radial_hidden_dim,
            attention=attention,
            residual=residual,
            dropout=dropout,
            eps=eps,
        )
        self.readout = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, 1),
        )
        self.pool = _get_pooling(pooling)

    def forward(
        self,
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

        scalar_h = self.node_encoder(h.float())
        h_irrep = torch.zeros(
            h.size(0),
            self.irreps_hidden.dim,
            device=h.device,
            dtype=scalar_h.dtype,
        )
        h_irrep[:, self.scalar_slice] = scalar_h
        h_irrep = self.backbone(
            h=h_irrep,
            x=x.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        graph_state = self.pool(h_irrep[:, self.scalar_slice], batch)
        return self.readout(graph_state).squeeze(-1)
