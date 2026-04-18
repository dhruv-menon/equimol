# Annotated E(n)-Equivariant Graph Neural Network

A practical, annotated walkthrough for building an E(n)-Equivariant Graph Neural Network (EGNN) for molecular property prediction on QM9.

The notebook focuses on clarity over compactness: it walks through molecule featurisation, graph construction, coordinate handling, the EGNN layer, training, checkpointing, and evaluation. The long-term motivation is to use this supervised QM9 baseline as a stepping stone toward more geometry-centric models such as equivariant denoisers, diffusion models, or flow-matching models.

## Contents

- `annotated-egnn.ipynb`: main tutorial notebook.
- `img/`: small diagrams used by the notebook.
- `annotated_egnn_ref.ipynb`: earlier clean reference version.
- `annotated_egnn_geom_drugs.ipynb`: experimental GEOM-Drugs extension.

## Setup

Create a fresh conda environment:

```bash
conda create -n egnn python=3.11 -y
conda activate egnn
python -m pip install --upgrade pip
```

Install the core dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit
pip install tqdm numpy pandas matplotlib jupyter
```

Depending on your platform, PyTorch and PyG may require installation commands specific to your CUDA or CPU setup. If installation fails, check the official PyTorch and PyTorch Geometric install pages.

## Running The Tutorial

Open the notebook:

```bash
jupyter notebook annotated-egnn.ipynb
```

Then run the cells from top to bottom. The notebook preprocesses QM9 into PyTorch Geometric `Data` objects, trains an EGNN regressor on the HOMO-LUMO gap, and evaluates the best checkpoint on a held-out test split.

## Notes

- The notebook intentionally includes detailed commentary for readers who want to understand the mechanics of EGNNs rather than only run the code.
- Generated data and checkpoints are ignored by Git. Recreate them by running the notebook.
- The model is a supervised property-prediction baseline, not yet a diffusion model. It is designed to be a clean foundation for later denoising or generative extensions.

## References

- Satorras, Hoogeboom, and Welling. [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
- Distill. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/).
