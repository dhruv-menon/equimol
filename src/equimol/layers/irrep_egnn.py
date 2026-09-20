from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import NormActivation

# ----------------------------------------
# An e3nn-based equivariant message passing layer.
#   - Represents node states using arbitrary O(3) irreps, e.g.
#     "64x0e + 32x1o + 16x2e".
#   - Computes edge geometry with spherical harmonics and mixes node/edge
#     features through a learnable tensor product.
#
# Shapes:
#    h: [N, H_irrep] node irrep features
#    x: [N, 3] coordinates
#    edge_index: [2, E] directed graph edges
#    edge_attr: [E, A] optional invariant edge features
#    direction: [E, 3] unit edge directions
#    harmonics: [E, Y_irrep] spherical harmonics matching irreps_edge
#    weights: [E, W] tensor product weights from radial_mlp
#    messages: [E, H_irrep] edge messages after tensor product
#    aggregated: [N, H_irrep] node messages after index_add over dst
#
# Mathematics:
#    r_ij = x_j - x_i
#    d_ij = ||r_ij||_2
#    rhat_ij = r_ij / (d_ij + eps)
#    Y_ij = Y_lm(rhat_ij)
#    w_ij = phi_r([d_ij, edge_attr_ij])
#    m_ij = TP(h_j, Y_ij; w_ij)
#    h_i <- sigma_norm(h_i + Linear(sum_j m_ij))
#
# Why equivariant:
#   - Spherical harmonics transform according to their irreps under rotations.
#     The e3nn tensor product applies Clebsch-Gordan rules internally, so
#     messages transform consistently with irreps_hidden. Radial weights and
#     attention depend only on invariant scalars, so they do not break
#     equivariance.
#
# Complexity:
#    - O(E * TP_cost) message passing for E edges, where TP_cost depends on the
#      multiplicities and maximum l in irreps_hidden and irreps_edge.
# ----------------------------------------


class IrrepEGNNLayer(nn.Module):
    """Single e3nn irrep message-passing layer"""

    def __init__(
        self,
        irreps_hidden: str = "64x0e + 32x1o + 16x2e",
        irreps_edge: str = "0e + 1o + 2e",
        edge_attr_dim: int = 0,
        radial_hidden_dim: int = 128,
        attention: bool = True,
        residual: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.edge_attr_dim = edge_attr_dim
        self.attention = attention
        self.residual = residual
        self.eps = eps

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_hidden,
            self.irreps_edge,
            self.irreps_hidden,
            shared_weights=False,
        )

        radial_input_dim = 1 + edge_attr_dim
        self.radial_mlp = nn.Sequential(
            nn.Linear(radial_input_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel),
        )
        self.update = o3.Linear(self.irreps_hidden, self.irreps_hidden)
        self.activation = NormActivation(
            self.irreps_hidden,
            scalar_nonlinearity=torch.nn.functional.silu,
            normalize=True,
            epsilon=eps,
        )

        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(radial_input_dim, radial_hidden_dim),
                nn.SiLU(),
                nn.Linear(radial_hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = x[src] - x[dst]
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        direction = rel / (dist + self.eps)

        if edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.size(1),
                self.edge_attr_dim,
                device=h.device,
                dtype=h.dtype,
            )

        edge_input = torch.cat([dist, edge_attr], dim=-1)
        weights = self.radial_mlp(edge_input)
        harmonics = o3.spherical_harmonics(
            self.irreps_edge,
            direction,
            normalize=True,
            normalization="component",
        )

        messages = self.tp(h[src], harmonics, weights)
        if self.attention:
            messages = messages * self.attention_mlp(edge_input)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        update = self.update(aggregated)
        h = h + update if self.residual else update
        return self.activation(h)


class IrrepEGNNBackbone(nn.Module):
    """Stacked IrrepEGNN message-passing layers"""

    def __init__(
        self,
        num_layers: int,
        irreps_hidden: str = "64x0e + 32x1o + 16x2e",
        irreps_edge: str = "0e + 1o + 2e",
        edge_attr_dim: int = 0,
        radial_hidden_dim: int = 128,
        attention: bool = True,
        residual: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.layers = nn.ModuleList(
            [
                IrrepEGNNLayer(
                    irreps_hidden=irreps_hidden,
                    irreps_edge=irreps_edge,
                    edge_attr_dim=edge_attr_dim,
                    radial_hidden_dim=radial_hidden_dim,
                    attention=attention,
                    residual=residual,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h=h, x=x, edge_index=edge_index, edge_attr=edge_attr)
        return h
