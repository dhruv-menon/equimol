"""Batching adapters for dense and sparse graph representations."""

from .dense import dense_to_sparse_nodes, sparse_to_dense_nodes

__all__ = ["dense_to_sparse_nodes", "sparse_to_dense_nodes"]
