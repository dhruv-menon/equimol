"""Tests for the smooth radial cutoff in irrep-egnn.

With `cutoff=r_c`, every message is multiplied by PolynomialCutoff(r_c)(r_ij), so
on a radius graph with radius r_c the energy and forces stay continuous when an
atom crosses the cutoff and its edge appears or disappears.
"""

import math

import pytest
import torch

from equimol.graphs import radius_graph
from equimol.layers.irrep_egnn import IrrepEGNNLayer
from equimol.models.irrep_egnn import IrrepEGNNRegressor

pytestmark = pytest.mark.radial

R_CUT = 3.0
IRREPS = "4x0e + 4x1o + 4x2e"


def _model(cutoff: float | None, seed: int = 0) -> IrrepEGNNRegressor:
    torch.manual_seed(seed)
    return IrrepEGNNRegressor(
        node_feat_dim=3,
        num_layers=2,
        irreps_hidden=IRREPS,
        irreps_edge="0e + 1o + 2e",
        radial_hidden_dim=8,
        cutoff=cutoff,
    ).double()


def _energy_and_forces(model: IrrepEGNNRegressor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy and forces on a radius graph with radius R_CUT."""
    x = x.clone().requires_grad_(True)
    h = torch.eye(3, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    edge_index = radius_graph(x, radius=R_CUT, batch=batch)
    energy = model(h, x, edge_index, batch)
    (grad,) = torch.autograd.grad(energy.sum(), x)
    return energy.detach(), -grad


def _three_atoms(r_12: float) -> torch.Tensor:
    # Atoms 0 and 1 are always bonded; only the pair (1, 2) crosses the cutoff,
    # since atom 2 stays further than R_CUT from atom 0.
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0 + r_12, 0.3, 0.0]],
        dtype=torch.float64,
    )


def _jump(model: IrrepEGNNRegressor, delta: float = 1e-4) -> tuple[float, float]:
    """Change in energy and forces as atom 2 crosses the cutoff of atom 1."""
    r_inside = math.sqrt((R_CUT - delta) ** 2 - 0.3**2)
    r_outside = math.sqrt((R_CUT + delta) ** 2 - 0.3**2)
    e_in, f_in = _energy_and_forces(model, _three_atoms(r_inside))
    e_out, f_out = _energy_and_forces(model, _three_atoms(r_outside))
    return (e_in - e_out).abs().item(), (f_in - f_out).abs().max().item()


# ----- continuity -----

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_energy_and_forces_are_continuous_across_the_cutoff(seed):
    energy_jump, force_jump = _jump(_model(cutoff=R_CUT, seed=seed))

    assert energy_jump < 1e-8
    assert force_jump < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_without_the_envelope_the_energy_jumps(seed):
    # Negative control: shows the continuity test can detect a discontinuity.
    energy_jump, force_jump = _jump(_model(cutoff=None, seed=seed))

    assert energy_jump > 1e-4 or force_jump > 1e-4


def test_message_from_a_neighbour_at_the_cutoff_is_zero():
    # Adding a neighbour exactly at r_c changes nothing, even though the radius
    # graph (dist <= radius) includes that edge.
    model = _model(cutoff=R_CUT)
    h = torch.eye(3, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    x = _three_atoms(math.sqrt(R_CUT**2 - 0.3**2))
    with_edge = radius_graph(x, radius=R_CUT + 1e-9, batch=batch)
    without_edge = with_edge[:, ~(((with_edge[0] == 1) & (with_edge[1] == 2)) | ((with_edge[0] == 2) & (with_edge[1] == 1)))]

    assert with_edge.size(1) == without_edge.size(1) + 2
    assert torch.allclose(model(h, x, with_edge, batch), model(h, x, without_edge, batch), atol=1e-12)


# ----- symmetry and defaults -----

def test_energy_is_invariant_and_forces_equivariant_with_the_envelope():
    model = _model(cutoff=R_CUT)
    x = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, -0.3], [0.4, 1.5, 0.6]], dtype=torch.float64)
    rotation = torch.linalg.matrix_exp(
        torch.tensor([[0.0, -0.4, 0.3], [0.4, 0.0, -0.8], [-0.3, 0.8, 0.0]], dtype=torch.float64)
    )

    e, f = _energy_and_forces(model, x)
    e_rot, f_rot = _energy_and_forces(model, x @ rotation.T)

    assert torch.allclose(e, e_rot, atol=1e-10)
    assert torch.allclose(f @ rotation.T, f_rot, atol=1e-10)


def test_without_a_cutoff_the_layer_has_no_envelope():
    assert IrrepEGNNLayer(irreps_hidden=IRREPS, irreps_edge="0e + 1o").envelope is None


def test_cutoff_reaches_every_layer():
    model = _model(cutoff=R_CUT)

    assert [layer.envelope.r_cut for layer in model.backbone.layers] == [R_CUT, R_CUT]
