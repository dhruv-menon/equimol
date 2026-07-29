# Equivariant Molecular ML

PyTorch building blocks for equivariant architectures in molecular and protein machine learning.

This repository provides small, readable implementations of geometry-aware neural network components: EGNN layers, invariant attention, graph construction utilities, pooling, and equivariance tests. The goal is to make the core operations easy to inspect, test, and extend for molecular property prediction, coordinate denoising, diffusion, and protein backbone modeling.

## Contents

- `annotated-egnn.ipynb`: annotated QM9 property-prediction example.
- `img/`: small diagrams used by the notebook.
- `src/egnn/`: reusable EGNN layer, graph construction, geometry, and model code.
- `tests/`: executable checks for equivariance, invariance, and graph shapes.

## Setup

Create and sync the uv environment:

```bash
uv sync --group dev
```

If you prefer a manual environment, install the same core dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit
pip install tqdm numpy pandas matplotlib jupyter pytest
```

## Running The Example

Open the notebook:

```bash
jupyter notebook annotated-egnn.ipynb
```

Then run the cells from top to bottom. The notebook preprocesses QM9 into PyTorch Geometric `Data` objects, trains an EGNN regressor on the HOMO-LUMO gap, and evaluates the best checkpoint on a held-out test split.

## Running The Tests

The first package tests focus on the core EGNN contract:

- invariant scalar node features stay invariant under rigid motions;
- coordinates transform equivariantly under translations and rotations;
- graph-level scalar predictions are invariant to node permutations and E(n) transforms.

```bash
uv run pytest -q
```

## Notes

- The notebook includes detailed commentary for readers who want to understand the mechanics of EGNNs rather than only run the code.
- Generated data and checkpoints are ignored by Git. Recreate them by running the notebook.
- The model is a supervised property-prediction baseline. By changing the target index in the configuration class, the task can focus on any of the 19 targets available in the dataset. 

## References

- Satorras, Hoogeboom, and Welling. [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
- Distill. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/).
