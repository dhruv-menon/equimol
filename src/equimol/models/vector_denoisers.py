from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from equimol.layers.time import TimestepEmbedding
from equimol.layers.vector_egnn import VectorEGNNBackbone


# ----------------------------------------
# VectorEGNN coordinate denoiser.
#    - Predict coordinate noise for DDPM-style coordinate diffusion.
#    - Uses learned l = 1 vector hidden states in the message-passing backbone.
#
# Shapes:
#    h: [N, F] invariant node or atom features.
#    x_t: [N, 3] noisy coordinates.
#    t: [] or [B] timestep indices.
#    edge_index: [2, E] directed edges; row 0 is source, row 1 is target.
#    batch: optional [N] graph ids.
#    edge_attr: optional [E, A] invariant edge features.
#    v: [N, 3, V] learned vector hidden states.
#    eps_hat: [N, 3] predicted coordinate noise.
#
# Mathematics:
#    x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
#    tau = phi_t(t)
#    h_i^0 = phi_h(h_i, tau_batch_i)
#    v_i^0 = 0
#    h^L, v^L, x^L = VectorEGNNBackbone(h^0, x_t, v^0, edge_index, edge_attr)
#    eps_hat_i = gamma(h_i^L) * (x_i^L - x_{t,i})
#    loss = mean_i ||eps_hat_i - eps_i||^2
#
# Why equivariant:
#    - Timestep and node features are invariant scalars. Vector states transform
#      with coordinates, and the output is a scalar-gated coordinate
#      displacement, so eps_hat rotates/reflections like x_t and is invariant to
#      global translation.
#
# Data contract:
#    - Molecule/protein differences live in h, edge_index, edge_attr, batch, and
#      loss masks. The denoiser itself only sees a graph with noisy coordinates.
# ----------------------------------------


class VectorEGNNCoordinateDenoiser(nn.Module):
    """VectorEGNN epsilon-prediction denoiser for coordinate diffusion."""
    def __init__(
            self,
            node_dim: int,
            num_layers: int = 4,
            hidden_dim: int = 128,
            edge_attr_dim: int = 0,
            message_dim: int = 128,
            vector_dim: int = 64,
            attention: bool = True,
            attention_dim: int = 128,
            vector_gate: bool = True,
            time_embedding_dim: int = 128,
            residual: bool = True,
            dropout: float = 0.0,
            coord_step_size: float = 0.1, 
            eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if node_dim <= 0:
            raise ValueError(f"node_dim must be positive, got {node_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if edge_attr_dim < 0:
            raise ValueError(f"edge_attr_dim must be non-negative, got {edge_attr_dim}")
        if message_dim <= 0:
            raise ValueError(f"message_dim must be positive, got {message_dim}")
        if vector_dim <= 0:
            raise ValueError(f"vector_dim must be positive, got {vector_dim}")
        if attention_dim <= 0:
            raise ValueError(f"attention_dim must be positive, got {attention_dim}")
        if time_embedding_dim <= 0:
            raise ValueError(f"time_embedding_dim must be positive, got {time_embedding_dim}")

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.vector_dim = vector_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.time_embedding = TimestepEmbedding(
             embedding_dim = time_embedding_dim,
             hidden_dim = hidden_dim,
             output_dim = hidden_dim
        )

        self.node_time_embedding = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.backbone = VectorEGNNBackbone(
            num_layers = num_layers,
            hidden_dim = hidden_dim,
            edge_attr_dim = edge_attr_dim,
            message_dim = message_dim,
            vector_dim = vector_dim,
            attention = attention,
            attention_dim = attention_dim,
            vector_gate = vector_gate,
            update_coords = True,
            residual = residual,
            dropout = dropout,
            coord_step_size = coord_step_size,
            eps = eps 
        )

        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
            self, 
            h: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor, 
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """Return the predicted coordinate noise eps_hat with shape [N, 3]"""

        if h.ndim != 2:
            raise ValueError(f"h must have shape [N, F], got {tuple(h.shape)}")
        if h.shape[-1] != self.node_dim:
            raise ValueError(f"Expected h with feature dim {self.node_dim}, got {h.shape[-1]}")
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape [N, 3], got {tuple(x_t.shape)}")
        if x_t.shape[0] != h.shape[0]:
            raise ValueError(f"Expected x_t with {h.shape[0]} nodes, got {x_t.shape[0]}")
        if x_t.shape[-1] != 3:
            raise ValueError(f"Expected x_t coordinate dim 3, got {x_t.shape[-1]}")
        if t.ndim > 1:
            raise ValueError(f"t must have shape [] or [B], got {tuple(t.shape)}")
        if not torch.is_floating_point(h):
            raise TypeError(f"h must be a floating point tensor, got {h.dtype}")
        if not torch.is_floating_point(x_t):
            raise TypeError(f"x_t must be a floating point tensor, got {x_t.dtype}")
        if t.ndim != 1:
            t = t.reshape(1)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(f"edge_index must have dtype torch.long, got {edge_index.dtype}")
        if edge_index.device != h.device:
            raise ValueError("edge_index must be on the same device as h")
        if x_t.device != h.device:
            raise ValueError("x_t must be on the same device as h")
        if edge_index.numel() > 0:
            if edge_index.min() < 0 or edge_index.max() >= h.shape[0]:
                raise ValueError("edge_index contains node indices outside [0, N)")
        if batch is not None:
            if batch.ndim != 1:
                raise ValueError(f"batch must have shape [N], got {tuple(batch.shape)}")
            if batch.shape[0] != h.shape[0]:
                raise ValueError(f"Expected batch with length {h.shape[0]}, got {batch.shape[0]}")
            if batch.device != h.device:
                raise ValueError("batch must be on the same device as h")
            if batch.dtype != torch.long:
                raise TypeError(f"batch must have dtype torch.long, got {batch.dtype}")
            if batch.numel() > 0 and batch.min() < 0:
                raise ValueError("batch indices must be non-negative")
            max_batch = int(batch.max().item()) if batch.numel() > 0 else -1
            if t.shape[0] == 1:
                pass
            elif t.shape[0] <= max_batch:
                raise ValueError(
                    "graph-wise timesteps must cover all batch indices, "
                    f"got {t.shape[0]} timesteps for max batch index {max_batch}"
                )
        elif t.shape[0] != 1:
            raise ValueError("graph-wise timesteps require batch")
        if edge_attr is not None:
            if edge_attr.ndim != 2:
                raise ValueError(f"edge_attr must have shape [E, A], got {tuple(edge_attr.shape)}")
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"Expected edge_attr with {edge_index.shape[1]} edges, got {edge_attr.shape[0]}"
                )
            if edge_attr.shape[1] != self.edge_attr_dim:
                raise ValueError(
                    f"Expected edge_attr dim {self.edge_attr_dim}, got {edge_attr.shape[1]}"
                )
            if edge_attr.device != h.device:
                raise ValueError("edge_attr must be on the same device as h")

        node_embedding = self.node_mlp(h)
        t_embedding = self.time_embedding(t.to(device = h.device))
        if batch is None:
            t_embedding = t_embedding.expand(h.shape[0], -1)
        elif t_embedding.shape[0] == 1:
            t_embedding = t_embedding.expand(h.shape[0], -1)
        else:
            t_embedding = t_embedding[batch]

        input_embedding = torch.concat([node_embedding, t_embedding], dim = -1)
        input = self.node_time_embedding(input_embedding)
        v = torch.zeros(
            h.shape[0],
            x_t.shape[-1],
            self.vector_dim,
            device=x_t.device,
            dtype=x_t.dtype,
        )

        h_updated, _, x_t_updated = self.backbone(
            h = input,
            x = x_t,
            v = v,
            edge_index = edge_index,
            edge_attr = edge_attr
        )

        gate = self.output_gate(h_updated)
        delta_x = x_t_updated - x_t
        eps_hat = gate * delta_x
        return eps_hat


VectorEGNNDenoiser = VectorEGNNCoordinateDenoiser


