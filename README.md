# Annotated E(n)-Equivariant Graph Neural Network Walkthrough

A practical, annotated walkthrough for building an E(n)-Equivariant Graph Neural Network (EGNN). As an illustrative example, I perform molecular property prediction on QM9 -- specifically, the HOMO-LUMO gap.

The notebook is heavily annotated: it walks through molecule featurisation, graph construction, coordinate handling, the EGNN layer, training, checkpointing, and evaluation. In principle, the model can be used as a stepping stone toward more geometry-centric models such as equivariant denoisers, diffusion models, or flow-matching models.

## Contents

- `annotated-egnn.ipynb`: main tutorial notebook.
- `img/`: small diagrams used by the notebook.

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

## Running The Tutorial

Open the notebook:

```bash
jupyter notebook annotated-egnn.ipynb
```

Then run the cells from top to bottom. The notebook preprocesses QM9 into PyTorch Geometric `Data` objects, trains an EGNN regressor on the HOMO-LUMO gap, and evaluates the best checkpoint on a held-out test split.

## Notes

- The notebook intentionally includes detailed commentary for readers who want to understand the mechanics of EGNNs rather than only run the code.
- Generated data and checkpoints are ignored by Git. Recreate them by running the notebook.
- The model is a supervised property-prediction baseline. By changing the target index in the configuration class, the task can focus on any of the 19 targets available in the dataset. 

## References

- Satorras, Hoogeboom, and Welling. [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
- Distill. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/).
