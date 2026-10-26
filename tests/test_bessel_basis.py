"""Specification tests for BesselBasis, the radial basis used with PolynomialCutoff.

    from equimol.layers import BesselBasis

    basis = BesselBasis(r_cut=5.0, num_basis=8)
    e = basis(distance)           # [E] or [E, 1] -> [E, K]

For n = 1..K:
    e_n(r) = sqrt(2 / r_cut) * sin(n pi r / r_cut) / r
with the finite limit e_n(0) = sqrt(2 / r_cut) * n pi / r_cut.
"""

import math

import pytest
import torch

from equimol.layers import BesselBasis

pytestmark = pytest.mark.radial

R_CUT = 5.0
K = 8


def _manual(r: float, n: int, r_cut: float = R_CUT) -> float:
    return math.sqrt(2 / r_cut) * math.sin(n * math.pi * r / r_cut) / r


def _basis() -> BesselBasis:
    # A default (float32) module: float64 inputs must still get float64 precision.
    return BesselBasis(r_cut=R_CUT, num_basis=K)


def _inside(steps: int = 4001) -> torch.Tensor:
    """Distances on (0, r_cut], float64."""
    return torch.linspace(0.0, R_CUT, steps, dtype=torch.float64)[1:]


# ----- shape and values -----

@pytest.mark.parametrize("shape", [(6,), (6, 1)])
def test_output_has_one_column_per_basis_function(shape):
    r = torch.rand(shape) * R_CUT + 0.1

    assert BesselBasis(r_cut=R_CUT, num_basis=K)(r).shape == torch.Size([6, K])


def test_matches_the_formula():
    r = torch.tensor([0.3, 1.7, 4.2], dtype=torch.float64)

    e = _basis()(r)

    expected = torch.tensor([[_manual(ri, n) for n in range(1, K + 1)] for ri in r.tolist()], dtype=torch.float64)
    assert torch.allclose(e, expected, atol=1e-12)


def test_every_function_vanishes_at_the_cutoff():
    e = _basis()(torch.tensor([R_CUT], dtype=torch.float64))

    assert torch.allclose(e, torch.zeros_like(e), atol=1e-12)


def test_function_n_has_n_minus_one_interior_roots():
    # Catches frequencies that start at n = 0 or are not multiples of pi / r_cut.
    e = _basis()(_inside()[:-1])  # stop short of the root at r_cut

    sign_changes = (torch.sign(e[1:]) != torch.sign(e[:-1])).sum(dim=0)
    assert sign_changes.tolist() == list(range(K))


def test_is_orthonormal_with_weight_r_squared():
    # Integral over [0, r_cut] of e_m e_n r^2 dr = delta_mn; this pins down sqrt(2 / r_cut).
    r = torch.linspace(0.0, R_CUT, 20001, dtype=torch.float64)
    e = _basis()(r)

    weighted = e * r[:, None] ** 2
    gram = torch.trapezoid(weighted[:, :, None] * e[:, None, :], r, dim=0)
    assert torch.allclose(gram, torch.eye(K, dtype=torch.float64), atol=1e-6)


# ----- r -> 0 -----

def test_is_finite_at_zero_with_the_analytic_limit():
    e = _basis()(torch.zeros(1, dtype=torch.float64))

    limit = [math.sqrt(2 / R_CUT) * n * math.pi / R_CUT for n in range(1, K + 1)]
    assert torch.allclose(e[0], torch.tensor(limit, dtype=torch.float64), atol=1e-12)


def test_gradients_are_finite_everywhere_including_zero():
    r = torch.linspace(0.0, 2 * R_CUT, 201, dtype=torch.float64, requires_grad=True)

    (grad,) = torch.autograd.grad(_basis()(r).sum(), r)

    assert torch.isfinite(grad).all()
    assert abs(grad[0].item()) < 1e-8  # e_n is even in r, so its slope at 0 is zero


# ----- dtype, device, validation -----

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_dtype(dtype):
    r = torch.rand(5, dtype=dtype) * R_CUT

    assert BesselBasis(r_cut=R_CUT, num_basis=K)(r).dtype == dtype


def test_stores_the_integers_n_as_its_only_buffer():
    basis = BesselBasis(r_cut=R_CUT, num_basis=K)

    buffers = list(basis.buffers())
    assert len(buffers) == 1
    assert torch.equal(buffers[0], torch.arange(1, K + 1, dtype=buffers[0].dtype))
    assert buffers[0].is_floating_point()
    assert not list(basis.parameters())


@pytest.mark.parametrize(("r_cut", "num_basis"), [(0.0, 8), (-1.0, 8), (5.0, 0), (5.0, -3)])
def test_validates_inputs(r_cut, num_basis):
    with pytest.raises(ValueError):
        BesselBasis(r_cut=r_cut, num_basis=num_basis)
