"""VectorEGNNRegressor scaffold.

Goal:
    Add l = 1 vector channels to the current l = 0 scalar EGNN regressor.

Inputs:
    z_i or h_i      atom/node scalar features
    x_i             coordinates in R^3
    edge_index      directed edges j -> i
    e_ij            invariant edge features / radial basis
    batch_i         graph id per atom

Hidden states:
    h_i^l in R^C
    v_i^l in R^{3 x C_v}

Initialization:
    h_i^0 = phi_embed(z_i)
    v_i^0 = 0

Geometry:
    r_ij = x_j - x_i
    d_ij = ||r_ij||_2
    rhat_ij = r_ij / (d_ij + eps)

Scalar edge message:
    m_ij = phi_m(h_i^l, h_j^l, d_ij, e_ij)

Optional invariant attention:
    a_ij = sigmoid(phi_a(h_i^l, h_j^l, d_ij, e_ij))
    m_ij <- a_ij * m_ij

Scalar aggregation:
    M_i = sum_{j in N(i)} m_ij

Vector message:
    alpha_ij = phi_v(m_ij)
    u_ij = rhat_ij outer alpha_ij
    U_i = sum_{j in N(i)} u_ij

Vector update:
    v_i^{l+1} = phi_vv(v_i^l) + U_i

Optional scalar gate on vector channels:
    g_i = sigmoid(phi_g(h_i^l))
    v_i^{l+1} <- g_i * v_i^{l+1}

Vector-to-scalar invariant feedback:
    s_i = ||v_i^{l+1}||_2
    optionally: s_i = [||v_i^{l+1}||_2, <v_i^a, v_i^b>]

Scalar update:
    h_i^{l+1} = h_i^l + phi_h(h_i^l, M_i, s_i)

Readout:
    g = sum_i phi_read(h_i^L)
    E_pred = phi_out(g)

Forces:
    F_pred = -dE_pred / dx

Training objective:
    L_E = MAE(E_pred, E_true)
    L_F = MAE(F_pred, F_true)
    L = lambda_E * L_E + lambda_F * L_F

Expected architecture:
    VectorEGNNRegressor
        node encoder
        L x VectorEGNNLayer
            scalar edge message
            optional invariant attention
            vector message from rhat_ij
            vector channel mixing
            vector norm/dot feedback into scalar stream
            scalar residual update
        graph pooling
        energy head
"""
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
