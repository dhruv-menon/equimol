# equimol

PyTorch-native E(n)-equivariant neural network building blocks for molecular and protein machine learning.

`equimol` is a small research-oriented library for composing EGNN-style models directly with PyTorch modules. It keeps the core symmetry contract explicit: scalar node features are invariant, coordinates are equivariant, and graph-level predictions become invariant through pooling.

The project is under active development. The current focus is a clean sparse `edge_index` API, tested equivariant layers, molecular graph utilities, and task-level EGNN models.

## Install

Development install:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest -q
```

## Quickstart

```python
import torch

from equimol.graphs import fully_connected_edges
from equimol.layers import GaussianRadialBasis
from equimol.models import EGNNRegressor

num_nodes = 8
node_feat_dim = 16

h = torch.randn(num_nodes, node_feat_dim)   # [N, F]
x = torch.randn(num_nodes, 3)               # [N, 3]
edge_index = fully_connected_edges(num_nodes)

rbf = GaussianRadialBasis(num_basis=32, cutoff=10.0)
edge_attr = rbf(x, edge_index)              # [E, 32]

model = EGNNRegressor(
    node_feat_dim=node_feat_dim,
    hidden_dim=128,
    num_layers=4,
    edge_attr_dim=32,
    pooling="mean",
)

y = model(h, x, edge_index, edge_attr=edge_attr)  # [1]
```

## Tensor Convention

Most layers and models use sparse graph batching:

```text
h:          [N, H] or [N, F] invariant scalar node features
x:          [N, D] coordinates
edge_index: [2, E] directed sparse edges
edge_attr:  [E, A] optional invariant edge features
batch:      [N] graph id per node
```

Layer contract:

```python
h, x = layer(h, x, edge_index, edge_attr=None)
```

Backbone contract:

```python
h, x = backbone(h, x, edge_index, edge_attr=None)
```

Graph regressor contract:

```python
y = model(h, x, edge_index, batch=None, edge_attr=None)
```

The library represents undirected graphs as two directed edges. For example, `0 -- 1` is represented as `0 -> 1` and `1 -> 0`.

## Symmetry Contract

`equimol` currently targets E(n)-equivariant scalar/vector EGNN-style models.

- Node features `h` are invariant scalar features.
- Coordinates `x` transform equivariantly under translations and rotations.
- Pairwise distances are invariant edge inputs.
- Coordinate updates are scalar-weighted sums of relative vectors.
- Graph-level scalar predictions are invariant after permutation-invariant pooling.

Absolute coordinates should not be passed directly into scalar MLPs. Use relative vectors, squared distances, distances, or other invariant/equivariant geometric features.

## Modules

### Adapters

```python
from equimol.adapters import MolecularGraphTensors
from equimol.adapters import MoleculeAdapter
from equimol.adapters import ProteinBackboneTensors
from equimol.adapters import ProteinBackboneAdapter
```

Adapters define the boundary between user data and `equimol` tensors. They are
normalizers, not neural network modules.

For proteins, the canonical residue-major backbone contract is:

```text
coordinates:   [R, 4, 3]
atom_mask:     [R, 4] or None
residue_types: [R] or None
residue_index: [R] or None
batch:         [R] or None
```

The backbone atom order is:

```text
0: N
1: CA
2: C
3: O
```

`ProteinBackboneAdapter.to_backbone_tensors(...)` should wrap and validate an
already-loaded protein representation into `ProteinBackboneTensors`. It is not
responsible for parsing PDB/mmCIF files, selecting chains, building graph edges,
computing geometry, or running an EGNN backbone.

The intended pipeline is:

```text
parser or user tensors
-> adapter
-> geometry/features
-> graph construction
-> layers/backbone
-> task head
```

### Layers

```python
from equimol.layers import EGNNLayer
from equimol.layers import AttentiveEGNNLayer
from equimol.layers import InvariantEdgeAttention
from equimol.layers import PairwiseDistance
from equimol.layers import GaussianRadialBasis
from equimol.layers import global_add_pool
from equimol.layers import global_mean_pool
```

Core layer outputs:

```text
EGNNLayer:          h [N, H], x [N, D] -> h' [N, H], x' [N, D]
AttentiveEGNNLayer: h [N, H], x [N, D] -> h' [N, H], x' [N, D]
PairwiseDistance:   x [N, D], edge_index [2, E] -> distance [E, 1]
GaussianRadialBasis: x [N, D], edge_index [2, E] -> edge_attr [E, K]
```

### Graphs

```python
from equimol.graphs import fully_connected_edges
from equimol.graphs import radius_graph
from equimol.graphs import knn_graph
```

Graph builders return sparse directed `edge_index` tensors with shape `[2, E]`.

Dense fully connected graphs have O(N^2) edges. Radius, kNN, and future sequential/local graph builders are intended for sparse molecular and protein settings.

### Geometry

```python
from equimol.geometry import bond_angle
from equimol.geometry import dihedral_angle
from equimol.geometry import distance
from equimol.geometry import squared_distance
```

Geometry helpers operate on tensor coordinates and return invariant scalar
quantities such as distances, bond angles, and torsions. They do not parse
structure files or build graphs.

### Models

```python
from equimol.models import EGNNBackbone
from equimol.models import AttentiveEGNNBackbone
from equimol.models import EGNNRegressor
from equimol.models import AttentiveEGNNRegressor
```

Backbones return hidden node states and updated coordinates:

```python
updated_h, updated_x = backbone(h, x, edge_index, edge_attr=None)
```

Regressors return one scalar per graph:

```python
y = regressor(h, x, edge_index, batch=batch, edge_attr=edge_attr)
```

Pooling options:

```text
pooling="sum"   extensive graph properties
pooling="mean"  intensive or size-normalized graph properties
```

## Batched Graphs

`equimol` uses sparse batching by concatenating graph nodes:

```text
graph 0: 3 nodes
graph 1: 2 nodes

h:     [5, H]
x:     [5, D]
batch: [0, 0, 0, 1, 1]
```

Example:

```python
batch = torch.tensor([0, 0, 0, 1, 1])
edge_index = fully_connected_edges(batch.numel(), batch=batch)

y = model(h, x, edge_index, batch=batch)  # [2]
```

This avoids padding and keeps message passing O(E). Dense batching adapters may be added later, but the core layer API is sparse-first.

## Tests

The test suite checks shape contracts, graph construction, pooling, and symmetry behavior:

```bash
uv run pytest -q
```

Current tested properties include:

- translation equivariance of coordinates
- rotation equivariance of coordinates
- permutation equivariance of node ordering
- invariance of graph-level scalar predictions
- segmented softmax behavior for edge attention
- dense/radius/kNN graph construction shapes
- radial distance feature invariance

## Roadmap

In progress:

- sequential/local graph construction for protein chains
- graph construction documentation and examples
- denser test coverage for molecular/protein graph utilities

Planned:

- molecular denoising models
- graph classifiers and node-level heads
- protein backbone utilities for N, CA, C, O coordinates
- bond angle and dihedral geometry features
- dense-to-sparse batching adapters

## Notebook

The first notebook is:

```text
notebooks/01_egnn_from_scratch.ipynb
```

The package code lives under `src/equimol/`; notebooks are intended to explain and demonstrate the modules rather than hide important implementation logic.

## References

- Satorras, Hoogeboom, and Welling. [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
- Distill. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/).
