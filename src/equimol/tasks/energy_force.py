from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from equimol.data.types import GeometricBatch
from equimol.training import to_device


def forces_from_energy(
        energy: torch.Tensor,
        pos: torch.Tensor,
        *,
        create_graph: bool
        ) -> torch.Tensor:
    """Forces as the negative gradient of the energy: F = -dE/dpos.

    Args:
        energy: Predicted energy per graph, shape [G], computed from pos.
        pos: Atom positions, shape [N, 3], with requires_grad=True.
        create_graph: Keep the graph of the gradient so a loss on the forces
            can be backpropagated. True for training, False for evaluation.

    Returns:
        Forces with the same shape as pos.
    """
    energy_scalar = energy.sum()
    return -torch.autograd.grad(outputs=energy_scalar, inputs=pos, create_graph=create_graph)[0]


class EnergyForceTask:
    """Energy-and-force regression (MD17/rMD17) as a training Task.

    The model predicts a normalised energy per graph; this task handles the
    normalisation, the forces (-dE/dx), the loss and the metrics.

    Usage:
        featurize = partial(prepare_md17_batch, radial_basis=rbf, graph="radius", radius=5.0)
        energy_mean, energy_std = compute_energy_stats(dataset, train_idx)
        task = EnergyForceTask(featurize, energy_mean, energy_std, lambda_force=10.0)
        trainer = Trainer(model, task, optimizer, TrainerConfig(..., select_metric="force_mae"))

    Model contract:
        model(h, x, edge_index, batch, edge_attr) -> normalised energy, shape [G]

    Metrics are in the dataset's units (kcal/mol and kcal/mol/Å for rMD17);
    convert with data.KCAL_MOL_TO_MEV for meV.
    """

    def __init__(
        self,
        featurize: Callable[[Any], GeometricBatch],
        energy_mean: float | torch.Tensor,
        energy_std: float | torch.Tensor,
        *,
        lambda_energy: float = 1.0,
        lambda_force: float = 1.0,
        per_atom_energy: bool = True,
    ) -> None:
        """Validate and store the task settings.

        Args:
            featurize: Turns a PyG batch into a GeometricBatch with targets in
                .y (energy) and .force.
            energy_mean: Training-set energy mean, in data units.
            energy_std: Training-set energy standard deviation, in data units.
            lambda_energy: Weight of the energy loss term.
            lambda_force: Weight of the force loss term.
            per_atom_energy: Divide each graph's energy error by its number of
                atoms before squaring, as NequIP and MACE do. Without it, the
                energy term dominates early training and the model can settle
                on predicting the mean energy with zero forces.

        Raises:
            ValueError: If energy_std <= 0, a weight is negative, or both weights
                are zero.
        """
        if energy_std <= 0:
            raise ValueError(f"energy_std must be positive, got {energy_std}")
        if lambda_energy < 0 or lambda_force < 0:
            raise ValueError(f"lambda_energy and lambda_force must be non-negative, got lambda_energy: {lambda_energy} and lambda_force: {lambda_force}")
        if lambda_energy == 0 and lambda_force == 0:
            raise ValueError("Either lambda_energy or lambda_force must be non-zero for training.")

        self.featurize = featurize
        self.lambda_energy = float(lambda_energy)
        self.lambda_force = float(lambda_force)
        self.energy_mean = float(energy_mean)
        self.energy_std = float(energy_std)
        self.per_atom_energy = per_atom_energy

    def predict(
        self,
        model: nn.Module,
        batch: Any,
        *,
        create_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor, GeometricBatch]:
        """Energies and forces, in data units, for one batch already on the device.

        batch.pos is replaced by a fresh leaf tensor requiring grad before
        featurizing, so forces include gradients through position-dependent edge
        features.

        Args:
            model: Energy model following the model contract.
            batch: PyG batch with pos, energy and force.
            create_graph: Passed to forces_from_energy.

        Returns:
            (energy [G], force [N, 3], geometric_batch)
        """
        pos = batch.pos.detach().float().requires_grad_(True)
        batch.pos = pos
        geometric_batch = self.featurize(batch)
        pred_norm = model(geometric_batch.h,
                          geometric_batch.x,
                          geometric_batch.edge_index,
                          geometric_batch.batch,
                          geometric_batch.edge_attr)

        energy = pred_norm * self.energy_std + self.energy_mean
        force = forces_from_energy(energy, pos, create_graph=create_graph)
        return energy, force, geometric_batch

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Weighted sum of the normalised energy MSE and the force L1 loss.

        Both terms are measured in units of energy_std, so the loss does not depend
        on the energy unit. With per_atom_energy, each graph's energy error is
        divided by its number of atoms.
        """
        energy, force, geometric_batch = self.predict(model=model, batch=batch, create_graph=True)
        target_energy = geometric_batch.y
        energy_error = (energy - target_energy) / self.energy_std
        if self.per_atom_energy:
            num_atoms = torch.bincount(geometric_batch.batch, minlength=energy.numel())
            energy_error = energy_error / num_atoms.to(energy_error.dtype)
        energy_loss = energy_error.pow(2).mean()

        target_force = geometric_batch.force
        target_force_norm = target_force / self.energy_std
        force_norm = force / self.energy_std

        force_loss = F.l1_loss(force_norm, target_force_norm)

        return self.lambda_energy * energy_loss + self.lambda_force * force_loss

    def evaluate(self, 
                 model: nn.Module, 
                 loader: Any,
                 device: torch.device
                 ) -> dict[str, float]:
        """Energy MAE per graph and force MAE per Cartesian component over the
        whole loader, in data units.

        Returns:
            {"energy_mae": float, "force_mae": float}

        Raises:
            ValueError: If the loader is empty.
        """
        energy_error_sum = 0.
        force_error_sum = 0.
        graph_count = 0
        cartesian_count = 0

        for batch in loader:
            batch = to_device(batch, device)
            energy, force, geometric_batch = self.predict(model, batch, create_graph=False)
            target_energy = geometric_batch.y
            target_force = geometric_batch.force

            energy_mae = F.l1_loss(energy, target_energy, reduction='sum')
            energy_error_sum += energy_mae.item()
            graph_count += geometric_batch.y.numel()

            force_mae = F.l1_loss(force, target_force, reduction='sum')
            force_error_sum += force_mae.item()
            cartesian_count += geometric_batch.force.numel()

        if graph_count == 0:
            raise ValueError("The graph_count is empty, no metrics calculated")
        
        return {
            'energy_mae': energy_error_sum / graph_count,
            'force_mae': force_error_sum / cartesian_count
            }



