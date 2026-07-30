> 🚧 **Under Construction:**  
> This is an initial release of the functionality. Further documentation and cleanup are still in progress.

# Equivariant Molecular Machine-Learning

PyTorch building blocks for equivariant architectures in molecular and protein machine learning.

This repository starts from first principles and builds toward practical geometric deep learning systems: EGNN layers, invariant attention, graph construction, pooling, equivariance tests, coordinate denoising, diffusion, and protein backbone modeling. The goal is to make the core operations small, readable, testable, and easy to extend.

I've tried my best to make the code as accessible as possible. The documentation is generous to aid beginners. 

## Roadmap

| Module | Topic | Status |
|---|---|---|
| 00 | Geometry and equivariance primer | Planned |
| 01 | EGNN from scratch | In progress |
| 02 | Graph construction and pooling | Planned |
| 03 | QM9 property prediction | Planned |
| 04 | Coordinate denoising and diffusion | Planned |
| 05 | Protein backbone diffusion | Planned |
| 06 | Invariant attention / IPA-style blocks | Planned |

## Contents

- `notebooks/01_egnn_from_scratch.ipynb`: annotated QM9 property-prediction example.
- `assets/figures/`: diagrams used by notebooks and docs.
- `src/equimol/`: reusable EGNN layer, graph construction, geometry, and model code.
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
jupyter notebook notebooks/01_egnn_from_scratch.ipynb
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
