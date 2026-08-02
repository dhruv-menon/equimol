"""PyTorch-native E(n)-equivariant molecular ML components.

The package keeps the core invariant/equivariant operations separate from
dataset-specific notebooks:

- invariant node features: h with shape [N, F]
- equivariant coordinates: x with shape [N, D]
- directed graph edges: edge_index with shape [2, E]
"""

from .graphs import fully_connected_edges, knn_graph, radius_graph
from .layers import AttentiveEGNNLayer, EGNNLayer, InvariantEdgeAttention, global_mean_pool, segment_sum
from .models import AttentiveEGNNBackbone, AttentiveEGNNRegressor, EGNNBackbone, EGNNRegressor

__all__ = [
    "AttentiveEGNNBackbone",
    "AttentiveEGNNLayer",
    "AttentiveEGNNRegressor",
    "EGNNLayer",
    "EGNNBackbone",
    "EGNNRegressor",
    "InvariantEdgeAttention",
    "fully_connected_edges",
    "global_mean_pool",
    "knn_graph",
    "radius_graph",
    "segment_sum",
]
