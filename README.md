> 🚧 **Under Construction:**  
> This is an initial release of the functionality. Further documentation and cleanup are still in progress.

# equimol

`equimol` is a PyTorch-native library of E(n)-equivariant neural network building blocks for molecular and protein machine learning.

The goal is to provide small, readable, testable modules for building EGNN-style architectures without requiring a full graph learning framework. The package focuses on explicit tensor contracts, symmetry behavior, and research-grade implementation details.

Current scope:

- E(n)-equivariant message passing layers
- invariant edge attention
- molecular and protein graph construction utilities
- geometric utilities for rigid transformations
- graph-level property prediction models
- tests for equivariance, invariance, and graph construction behavior

The repository is still early, but the intended direction is a practical library for researchers and engineers who want to compose their own equivariant molecular models from PyTorch modules.

## Design Contract

Most modules follow the same core EGNN tensor convention:

```python
h, x = layer(h, x, edge_index, edge_attr=None)
```

where:

- `h`: invariant scalar node features with shape [N, H]
- `x`: equivariant coordinates with shape [N, 3]
- `edge_index`: directed graph edges with shape [2, E]
- `edge_attr`: optional invariant edge features with shape [E, A]

The expected symmetry behavior is:

- scalar node features remain invariant under translations and rotations
- coordinates transform equivariantly under translations and rotations
- graph-level scalar predictions are invariant after permutation-invariant pooling
- no layer should feed absolute coordinates directly into scalar MLPs

## Package Layout

```text
src/equimol/
  graphs/
    fully_connected.py
    knn.py
    radius.py
    utils.py

  layers/
    attention.py
    attentive_egnn.py
    egnn.py
    pooling.py

  models/
    backbones.py
    regressors.py

  utils/
    geometry.py
    segment_sum.py
```

## Current API

Layers:

```python
from equimol.layers import EGNNLayer
from equimol.layers import AttentiveEGNNLayer
from equimol.layers import InvariantEdgeAttention
```

Backbones and models:

```python
from equimol.models import EGNNBackbone
from equimol.models import AttentiveEGNNBackbone
from equimol.models import EGNNRegressor
from equimol.models import AttentiveEGNNRegressor
```

Graph builders:

```python
from equimol.graphs import fully_connected_edges
from equimol.graphs import radius_graph
from equimol.graphs import knn_graph
```

Example:

```python
import torch

from equimol.graphs import fully_connected_edges
from equimol.models import EGNNRegressor

h = torch.randn(8, 16)                # [N, F]
x = torch.randn(8, 3)                 # [N, 3]
edge_index = fully_connected_edges(8) # [2, E]

model = EGNNRegressor(
    node_feat_dim=16,
    hidden_dim=128,
    num_layers=4,
)

y = model(h, x, edge_index)           # [B]
```

## Roadmap

### Completed

Stage 1: Core EGNN package contract

- `src/equimol/` package layout
- vanilla `EGNNLayer`
- `InvariantEdgeAttention`
- `AttentiveEGNNLayer`
- `EGNNBackbone`
- `AttentiveEGNNBackbone`
- `EGNNRegressor`
- `AttentiveEGNNRegressor`
- fully connected, radius, and kNN graph construction utilities
- geometry utilities for translation, rotation, and centering
- `segment_sum`
- equivariance and graph construction tests
- first notebook renamed to `notebooks/01_egnn_from_scratch.ipynb`

### In Progress

Stage 2: Graph construction, pooling, and task-model reliability

- regressor tests for output shape [B]
- translation, rotation, and permutation invariance tests for graph regressors
- batched graph pooling tests
- `global_mean_pool`
- configurable pooling in regressors
- clearer graph construction contracts for dense, radius, and kNN edges
- documentation for when to use dense O(N^2) graphs versus sparse molecular/protein graphs

### Planned

Stage 3: Geometric feature layers

- radial distance expansion
- Gaussian radial basis features
- safe norm and unit vector helpers
- bond angle utilities
- dihedral angle utilities
- tests showing radial, angle, and torsion features are E(n)-invariant

Stage 4: Molecular task models

- graph classifiers
- node-level prediction heads
- coordinate denoising models
- examples showing molecular property prediction and coordinate noise prediction
- documentation connecting denoising objectives to diffusion models

Stage 5: Protein and higher-order geometry

- sequential/local residue graph construction
- protein backbone graph helpers
- N, CA, C, O coordinate utilities
- CA-CA distance and backbone bond-length metrics
- angle-aware EGNN layers
- local-frame utilities for protein structure modeling

Longer-term scope:

- keep the library focused on E(n)-equivariant scalar/vector EGNN-style models
- avoid becoming a general PyTorch Geometric replacement
- add higher-order geometry only when the symmetry contract remains explicit and testable

## Setup

Create and sync the uv environment:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest -q
```

## Notebook

The first notebook is:

```text
notebooks/01_egnn_from_scratch.ipynb
```

It introduces the EGNN computation from first principles using molecular graph data. The reusable code lives under `src/equimol/`; notebooks should explain and demonstrate the package rather than hide important logic inside notebook cells.

## References

- Satorras, Hoogeboom, and Welling. [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
- Distill. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/).
