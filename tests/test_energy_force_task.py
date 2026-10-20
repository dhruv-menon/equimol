"""Specification tests for equimol.tasks.EnergyForceTask.

The task turns the MD17/rMD17 energy-and-force problem into the two methods
the Trainer needs (equimol.training.Task):

    from equimol.tasks import EnergyForceTask, forces_from_energy

    forces_from_energy(energy, pos, *, create_graph) -> Tensor      # -dE/dpos, shape of pos

    task = EnergyForceTask(
        featurize,                  # PyG batch -> GeometricBatch (e.g. partial(prepare_md17_batch, ...))
        energy_mean: float,         # training-set energy statistics, data units
        energy_std: float,          # > 0
        *,
        lambda_energy: float = 1.0, # >= 0, not both zero
        lambda_force: float = 1.0,
    )
    task.predict(model, batch, *, create_graph) -> (energy [G], force [N, 3], GeometricBatch)
    task.loss(model, batch) -> scalar Tensor
    task.evaluate(model, loader, device) -> {"energy_mae": float, "force_mae": float}

Conventions:
    - model(h, x, edge_index, batch, edge_attr) predicts the NORMALISED energy
      per graph; predict() returns energies in data units:
      energy = model_output * energy_std + energy_mean.
    - Forces are -dE/dpos of the data-unit energy, differentiated with respect
      to the batch positions (so gradients through edge features count).
    - loss = lambda_energy * MSE(normalised energy)
           + lambda_force  * L1(force / energy_std)
      Both terms are in units of energy_std, so the loss is unit-invariant.
    - evaluate() reports MAEs in data units: energy per graph, force per
      Cartesian component, averaged over the whole loader (not per batch).
"""

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from equimol.data import build_radial_basis, prepare_md17_batch
from equimol.data.types import GeometricBatch
from equimol.models import EGNNRegressor
from equimol.tasks import EnergyForceTask, forces_from_energy
from equimol.training import Task, Trainer, TrainerConfig

pytestmark = pytest.mark.tasks


class QuadraticEnergy(nn.Module):
    """Normalised energy w * sum(|x|^2) per graph, so forces are analytic:
    F = -d(std * w * sum|x|^2 + mean)/dx = -2 * std * w * x."""

    def __init__(self, w: float = 0.5):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w))

    def forward(self, h, x, edge_index, batch, edge_attr=None):
        per_atom = self.w * (x**2).sum(-1)
        num_graphs = int(batch.max()) + 1
        return torch.zeros(num_graphs, dtype=x.dtype).index_add(0, batch, per_atom)


def toy_featurize(data) -> GeometricBatch:
    return GeometricBatch(
        h=torch.zeros(data.pos.shape[0], 1),
        x=data.pos.float(),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        batch=data.batch,
        y=data.energy.float().reshape(-1),
        force=data.force.float(),
    )


def _molecules(sizes=(3, 2, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        Data(
            z=torch.tensor([6] + [1] * (n - 1)),
            pos=torch.randn(n, 3, generator=g),
            energy=torch.randn(1, generator=g, dtype=torch.float64) * 3.0 - 10.0,
            force=torch.randn(n, 3, generator=g),
        )
        for n in sizes
    ]


def _loader(sizes=(3, 2, 4), batch_size=2, seed=0):
    return DataLoader(_molecules(sizes, seed), batch_size=batch_size, shuffle=False)


def _batch(sizes=(3, 2, 4), seed=0):
    return next(iter(_loader(sizes, batch_size=len(sizes), seed=seed)))


# ----- construction -----


def test_energy_force_task_satisfies_the_task_protocol():
    assert isinstance(EnergyForceTask(toy_featurize, 0.0, 1.0), Task)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_std": 0.0},
        {"energy_std": -1.0},
        {"lambda_energy": -1.0},
        {"lambda_force": -0.1},
        {"lambda_energy": 0.0, "lambda_force": 0.0},
    ],
)
def test_energy_force_task_rejects_invalid_settings(kwargs):
    settings = {"energy_mean": 0.0, "energy_std": 1.0, **kwargs}
    with pytest.raises(ValueError):
        EnergyForceTask(toy_featurize, **settings)


def test_energy_force_task_accepts_tensor_statistics():
    # compute_energy_stats returns 0-d tensors; the task should take them as-is.
    task = EnergyForceTask(toy_featurize, torch.tensor(-11.0), torch.tensor(2.0))
    energy, _, _ = task.predict(QuadraticEnergy(), _batch(), create_graph=False)
    assert energy.dtype == torch.float32


# ----- forces_from_energy -----


def test_forces_from_energy_is_the_negative_gradient():
    pos = torch.randn(5, 3, requires_grad=True)
    energy = (pos**2).sum(-1) * 1.5  # one "graph" per atom
    force = forces_from_energy(energy, pos, create_graph=False)
    torch.testing.assert_close(force, -3.0 * pos.detach())


def test_forces_from_energy_create_graph_controls_double_backward():
    pos = torch.randn(4, 3, requires_grad=True)
    w = torch.tensor(2.0, requires_grad=True)
    energy = w * (pos**2).sum()

    assert forces_from_energy(energy, pos, create_graph=True).requires_grad
    energy = w * (pos**2).sum()
    assert not forces_from_energy(energy, pos, create_graph=False).requires_grad


# ----- predict -----


def test_predict_denormalises_energy_and_differentiates_it():
    mean, std, w = -11.0, 2.0, 0.5
    batch = _batch()
    pos = batch.pos.clone()
    task = EnergyForceTask(toy_featurize, mean, std)

    energy, force, geometric_batch = task.predict(QuadraticEnergy(w), batch, create_graph=False)

    per_atom = (pos**2).sum(-1)
    expected_energy = torch.zeros(3).index_add(0, batch.batch, per_atom) * w * std + mean
    torch.testing.assert_close(energy.detach(), expected_energy)
    torch.testing.assert_close(force.detach(), -2.0 * std * w * pos)
    assert force.shape == pos.shape
    assert isinstance(geometric_batch, GeometricBatch)


def test_predict_forces_include_gradients_through_edge_features():
    # The radial basis is built from positions inside featurize; forces must
    # include that path, so they are taken w.r.t. the batch positions.
    torch.manual_seed(0)
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0)
    featurize = partial(
        prepare_md17_batch, radial_basis=radial_basis, graph="fully_connected", num_atom_types=10
    )
    model = EGNNRegressor(
        node_feat_dim=10,
        num_layers=1,
        hidden_dim=16,
        edge_attr_dim=4,
        message_dim=16,
        update_coords=False,
        pooling="mean",
    ).eval()
    batch = _batch()
    pos = batch.pos.clone().float().requires_grad_(True)

    task = EnergyForceTask(featurize, -10.0, 3.0)
    _, force, _ = task.predict(model, batch, create_graph=False)

    reference = _batch()
    reference.pos = pos
    gb = featurize(reference)
    energy = model(gb.h, gb.x, gb.edge_index, gb.batch, gb.edge_attr) * 3.0 - 10.0
    expected = -torch.autograd.grad(energy.sum(), pos)[0]
    torch.testing.assert_close(force, expected)


# ----- loss -----


def test_loss_is_weighted_normalised_energy_mse_plus_force_l1():
    mean, std, w = -10.0, 2.5, 0.3
    lambda_energy, lambda_force = 0.7, 4.0
    batch = _batch()
    pos, target_energy, target_force = batch.pos.clone(), batch.energy.float(), batch.force.clone()
    task = EnergyForceTask(toy_featurize, mean, std, lambda_energy=lambda_energy, lambda_force=lambda_force)

    loss = task.loss(QuadraticEnergy(w), batch)

    pred_norm = torch.zeros(3).index_add(0, batch.batch, (pos**2).sum(-1)) * w
    pred_force = -2.0 * std * w * pos
    expected = lambda_energy * F.mse_loss(pred_norm, (target_energy - mean) / std) + lambda_force * F.l1_loss(
        pred_force / std, target_force / std
    )
    assert loss.dim() == 0
    torch.testing.assert_close(loss, expected)


def test_force_only_loss_trains_the_model_through_the_forces():
    # With lambda_energy=0 every gradient flows through F = -dE/dx, which only
    # exists if the force graph was built with create_graph=True.
    model = QuadraticEnergy(0.5)
    task = EnergyForceTask(toy_featurize, 0.0, 1.0, lambda_energy=0.0, lambda_force=1.0)

    task.loss(model, _batch()).backward()

    assert model.w.grad is not None
    assert model.w.grad.abs() > 0


def test_loss_is_invariant_to_the_energy_unit():
    # kcal/mol vs meV: scaling every energy and force by c, and the statistics
    # with them, leaves the loss unchanged.
    c = 43.364
    batch, scaled = _batch(), _batch()
    scaled.energy = scaled.energy * c
    scaled.force = scaled.force * c

    loss = EnergyForceTask(toy_featurize, -10.0, 2.0).loss(QuadraticEnergy(0.4), batch)
    scaled_loss = EnergyForceTask(toy_featurize, -10.0 * c, 2.0 * c).loss(QuadraticEnergy(0.4), scaled)

    torch.testing.assert_close(loss, scaled_loss)


# ----- evaluate -----


def test_evaluate_reports_data_unit_maes_over_the_whole_loader():
    # Batches of 2 molecules then 1, with different atom counts: the MAE must
    # be over all graphs / force components, not a mean of batch means.
    mean, std, w = -10.0, 2.0, 0.25
    molecules = _molecules((3, 2, 4))
    task = EnergyForceTask(toy_featurize, mean, std)

    metrics = task.evaluate(QuadraticEnergy(w), DataLoader(molecules, batch_size=2), torch.device("cpu"))

    energy_errors, force_errors = [], []
    for m in molecules:
        energy = w * (m.pos**2).sum() * std + mean
        energy_errors.append((energy - m.energy.float()).abs())
        force_errors.append((-2.0 * std * w * m.pos - m.force).abs().flatten())
    assert set(metrics) == {"energy_mae", "force_mae"}
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["energy_mae"] == pytest.approx(torch.cat(energy_errors).mean().item(), rel=1e-5)
    assert metrics["force_mae"] == pytest.approx(torch.cat(force_errors).mean().item(), rel=1e-5)


def test_evaluate_does_not_touch_parameter_gradients():
    model = QuadraticEnergy(0.5)
    EnergyForceTask(toy_featurize, 0.0, 1.0).evaluate(model, _loader(), torch.device("cpu"))
    assert model.w.grad is None


def test_evaluate_rejects_an_empty_loader():
    with pytest.raises(ValueError, match="empty"):
        EnergyForceTask(toy_featurize, 0.0, 1.0).evaluate(QuadraticEnergy(), [], torch.device("cpu"))


# ----- with the Trainer -----


def test_energy_force_task_trains_an_egnn_with_the_trainer(tmp_path):
    torch.manual_seed(0)
    radial_basis = build_radial_basis(num_basis=4, cutoff=5.0)
    featurize = partial(
        prepare_md17_batch, radial_basis=radial_basis, graph="fully_connected", num_atom_types=10
    )
    model = EGNNRegressor(
        node_feat_dim=10,
        num_layers=1,
        hidden_dim=16,
        edge_attr_dim=4,
        message_dim=16,
        update_coords=False,
        pooling="mean",
    )
    task = EnergyForceTask(featurize, -10.0, 3.0, lambda_force=10.0)
    trainer = Trainer(
        model,
        task,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        TrainerConfig(epochs=3, checkpoint_path=tmp_path / "ckpt.pt", select_metric="force_mae", ema_decay=0.9),
    )

    result = trainer.fit(_loader(), _loader(seed=1))
    test_metrics = trainer.test(_loader(seed=2))

    assert [h["epoch"] for h in result.history] == [1, 2, 3]
    assert {"energy_mae", "force_mae", "train_loss"} <= set(result.history[0])
    assert result.best_metric == min(h["force_mae"] for h in result.history)
    assert all(torch.isfinite(torch.tensor(v)) for v in test_metrics.values())
