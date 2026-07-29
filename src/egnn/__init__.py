"""Tutorial-first EGNN components.

The package keeps the core invariant/equivariant operations separate from
dataset-specific notebooks:

- invariant node features: ``h`` with shape ``[N, F]``
- equivariant coordinates: ``x`` with shape ``[N, D]``
- directed graph edges: ``edge_index`` with shape ``[2, E]``
"""

from .graph import fully_connected_edges, knn_graph, radius_graph
from .layers import EGNNLayer, segment_sum
from .models import EGNNRegressor

__all__ = [
    "EGNNLayer",
    "EGNNRegressor",
    "fully_connected_edges",
    "knn_graph",
    "radius_graph",
    "segment_sum",
]
