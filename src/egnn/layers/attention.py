"""Invariant edge attention for EGNN-style message passing.

The tensor contract is explicit because attention is easy to make subtly
non-equivariant if the score function sees non-invariant coordinate features.

Problem solved:
    Learn an invariant scalar attention weight for each directed edge. The
    attention weight can later modulate scalar messages and coordinate updates
    inside an EGNN block.

Expected symmetry:
    Attention scores must be invariant to translations and rotations. Therefore
    the attention MLP may use invariant scalar quantities such as node features,
    edge features, and pairwise squared distances. It must not use absolute
    coordinates directly.

Typical math:
    edge_input_ij = concat(h_i, h_j, ||x_i - x_j||^2, edge_attr_ij)
    score_ij = phi_attn(edge_input_ij)
    alpha_ij = softmax_i(score_ij) over all incoming edges i -> j

Shapes:
    h: ``[N, H]`` invariant node features.
    x: ``[N, D]`` equivariant coordinates.
    edge_index: ``[2, E]`` directed edges, with source and target rows.
    edge_attr: optional ``[E, A]`` invariant edge features.
    output alpha: ``[E, 1]`` invariant attention weights.

Complexity:
    ``O(EH + ED)`` plus the cost of the attention MLP and segmented softmax.

Common failure modes:
    Normalizing over all edges in the batch instead of per target node.
    Using absolute coordinates in the score MLP.
    Forgetting that attention weights are scalars; vector-valued attention
    needs stricter equivariance handling.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from egnn.utils import segment_sum


def segmented_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Apply softmax over incoming edges for each destination node.

    Args:
        scores: Raw edge scores with shape ``[E, 1]``.
        dst: Destination node index per edge with shape ``[E]``.
        num_nodes: Number of nodes ``N``.

    Returns:
        Edge attention weights with shape ``[E, 1]``. For each destination node
        ``j``, weights over incoming edges ``i -> j`` sum to one.
    """

    max_per_node = torch.full(
        (num_nodes, scores.size(-1)),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype,
    )
    max_per_node.scatter_reduce_(0, dst[:, None].expand_as(scores), scores, reduce="amax")

    shifted = scores - max_per_node[dst]
    exp_scores = shifted.exp() # [E, 1]
    denom_per_node = segment_sum(exp_scores, dst, num_nodes) # [N, 1]
    denom_per_edge = denom_per_node[dst].clamp_min(torch.finfo(scores.dtype).eps) # [E, 1]
    alpha = exp_scores / denom_per_edge # [E, 1]
    return alpha


class InvariantEdgeAttention(nn.Module):
    """Compute invariant attention weights for directed graph edges.

    Inputs:
        h:
            Invariant node features with shape ``[N, H]``.
        x:
            Coordinates with shape ``[N, D]``. These are used only through
            invariant pairwise distances.
        edge_index:
            Long tensor with shape ``[2, E]``.
            ``edge_index[0]`` is the source node ``i``.
            ``edge_index[1]`` is the target node ``j``.
        edge_attr:
            Optional invariant edge features with shape ``[E, A]``.

    Expected output:
        alpha:
            Attention weights with shape ``[E, 1]``.
            For each target node ``j``, incoming edge weights should sum to 1:
            ``sum_i alpha_ij = 1``.

    What the layer needs to do:
        1. Gather ``h_src`` and ``h_dst`` using ``edge_index``.
        2. Compute relative vectors ``x_src - x_dst``.
        3. Convert relative vectors to invariant squared distances.
        4. Concatenate invariant edge inputs.
        5. Map edge inputs to scalar attention scores with an MLP.
        6. Apply segmented softmax over incoming edges grouped by destination.
        7. Return ``alpha``.

    Research-level check:
        If ``x`` is translated or rotated, ``alpha`` should not change. If node
        order is permuted consistently with ``edge_index``, ``alpha`` should
        permute with the edges.
    """

    def __init__(self,
                 hidden_dim: int,
                 edge_attr_dim: int = 0,
                 attention_dim: int = 128,
                 dropout: float = 0.0,) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.attention_dim = attention_dim
        self.dropout = dropout

        # define an MLP that maps [h_src, h_dst, radial, edge_attr] to
        # one scalar score per edge: [E, 2H + 1 + A] -> [E, 1].
        edge_input_dim = 2 * self.hidden_dim + 1 + self.edge_attr_dim
        self.scorer = nn.Sequential(
            nn.Linear(edge_input_dim, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, 1),
        )
        
        # dropout should apply to attention weights after softmax.
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return invariant edge attention weights with shape [E, 1]"""

        # unpack source and destination nodes from edge_index.
        src, dst = edge_index # [E]
        h_src = h[src] # [E, H]
        h_dst = h[dst] # [E, H]
        x_src = x[src] # [E, D]
        x_dst = x[dst] # [E, D]

        # compute radial = ||x_src - x_dst||^2 with shape [E, 1].
        relative = x_src - x_dst # [E, D]
        radial = (relative * relative).sum(dim = 1, keepdim = True) # [E, 1]

        #  handle edge_attr=None as an empty [E, 0] tensor.
        if edge_attr is None:
            edge_attr = torch.zeros(h_src.size(0), 0, dtype = h_src.dtype, device = h_src.device)

        # build edge_input with shape [E, 2H + 1 + A].
        edge_input = torch.cat([h_src, h_dst, radial, edge_attr], dim = -1) # [E, 2H + 1 + A]

        # compute raw attention scores with shape [E, 1].
        score = self.scorer(edge_input) # [E, 1]

        # apply segmented softmax over destination nodes.
        alpha = segmented_softmax(score, dst, h.shape[0])

        return self.dropout_layer(alpha)
