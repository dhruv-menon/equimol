"""Specification tests for PolynomialCutoff, the smooth envelope used to make
radius-graph messages vanish continuously at the cutoff.

    from equimol.layers import PolynomialCutoff

    cutoff = PolynomialCutoff(r_cut=5.0, p=6)
    u = cutoff(distance)          # same shape and dtype as distance

With d = r / r_cut:
    u(d) = 1 - (p+1)(p+2)/2 d^p + p(p+2) d^(p+1) - p(p+1)/2 d^(p+2)   for d < 1
    u(d) = 0                                                          for d >= 1

Its derivative factorises, which the tests use as an independent check:
    du/dr = -p(p+1)(p+2) / (2 r_cut) * d^(p-1) (1 - d)^2               for d < 1
"""

import pytest
import torch

from equimol.layers import PolynomialCutoff

pytestmark = pytest.mark.radial

R_CUT = 5.0


def _grid(r_cut: float = R_CUT, steps: int = 401) -> torch.Tensor:
    """Distances from 0 to 2 r_cut, in float64 so derivative checks are tight."""
    return torch.linspace(0.0, 2.0 * r_cut, steps, dtype=torch.float64)


def _derivatives(cutoff: PolynomialCutoff, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First and second derivatives of u with respect to r, elementwise."""
    r = r.clone().requires_grad_(True)
    first = torch.autograd.grad(cutoff(r).sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), r)[0]
    return first.detach(), second


# ----- values -----

def test_matches_hand_computed_values_for_p6():
    # d = 0.5, p = 6: 1 - 28/64 + 48/128 - 21/256 = 0.85546875 exactly.
    cutoff = PolynomialCutoff(r_cut=R_CUT, p=6)
    r = torch.tensor([0.0, 0.5 * R_CUT, R_CUT], dtype=torch.float64)

    assert torch.allclose(cutoff(r), torch.tensor([1.0, 0.85546875, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("r", [R_CUT, 1.0001 * R_CUT, 1.5 * R_CUT, 3.0 * R_CUT, 100.0 * R_CUT])
def test_is_zero_at_and_beyond_the_cutoff(r):
    # The polynomial grows again past d = 1, so it must be masked, not just evaluated.
    cutoff = PolynomialCutoff(r_cut=R_CUT, p=6)

    assert cutoff(torch.tensor([r], dtype=torch.float64)).item() == 0.0


@pytest.mark.parametrize("p", [1, 3, 6, 9])
def test_stays_in_unit_interval_and_never_increases(p):
    u = PolynomialCutoff(r_cut=R_CUT, p=p)(_grid())

    assert (u >= 0).all() and (u <= 1).all()
    assert (u[1:] <= u[:-1] + 1e-12).all()


# ----- derivatives -----

@pytest.mark.parametrize("p", [1, 3, 6, 9])
def test_first_derivative_matches_the_factorised_form(p):
    # Catches wrong coefficients even when u(0) = 1 and u(r_cut) = 0 still hold.
    cutoff = PolynomialCutoff(r_cut=R_CUT, p=p)
    r = _grid()[1:-1]
    r = r[r < R_CUT]
    d = r / R_CUT

    first, _ = _derivatives(cutoff, r)

    expected = -p * (p + 1) * (p + 2) / (2 * R_CUT) * d.pow(p - 1) * (1 - d).pow(2)
    assert torch.allclose(first, expected, atol=1e-10)


@pytest.mark.parametrize("p", [3, 6, 9])
def test_first_and_second_derivatives_vanish_at_the_cutoff(p):
    cutoff = PolynomialCutoff(r_cut=R_CUT, p=p)
    just_inside = torch.tensor([R_CUT * (1 - 1e-6)], dtype=torch.float64)

    first, second = _derivatives(cutoff, just_inside)

    assert first.abs().item() < 1e-8
    assert second.abs().item() < 1e-3


def test_gradient_is_zero_and_finite_beyond_the_cutoff():
    # Masking must not leak the polynomial's gradient (or NaN) past r_cut.
    cutoff = PolynomialCutoff(r_cut=R_CUT, p=6)
    r = torch.tensor([1.0001 * R_CUT, 2.0 * R_CUT, 1e3 * R_CUT], dtype=torch.float64)

    first, second = _derivatives(cutoff, r)

    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(second, torch.zeros_like(second))


def test_gradients_are_finite_everywhere_including_zero():
    first, second = _derivatives(PolynomialCutoff(r_cut=R_CUT, p=6), _grid())

    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_far_distances_give_zero_not_nan_in_float32():
    # d^(p+2) overflows to inf in float32 here; 0 * inf would be NaN.
    r = torch.tensor([1e6 * R_CUT], requires_grad=True)

    u = PolynomialCutoff(r_cut=R_CUT, p=6)(r)
    (grad,) = torch.autograd.grad(u.sum(), r)

    assert u.item() == 0.0 and grad.item() == 0.0


# ----- shapes, dtypes, validation -----

@pytest.mark.parametrize("shape", [(7,), (7, 1), (2, 3, 4)])
def test_preserves_shape(shape):
    r = torch.rand(shape) * 2 * R_CUT

    assert PolynomialCutoff(r_cut=R_CUT)(r).shape == torch.Size(shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_dtype(dtype):
    r = torch.rand(5, dtype=dtype) * 2 * R_CUT

    assert PolynomialCutoff(r_cut=R_CUT)(r).dtype == dtype


def test_scales_with_the_cutoff():
    # u depends on r only through r / r_cut.
    r = torch.rand(20, dtype=torch.float64) * 4.0

    assert torch.allclose(PolynomialCutoff(r_cut=4.0)(r), PolynomialCutoff(r_cut=8.0)(2 * r))


@pytest.mark.parametrize(("r_cut", "p"), [(0.0, 6), (-1.0, 6), (5.0, 0), (5.0, -2)])
def test_validates_inputs(r_cut, p):
    with pytest.raises(ValueError):
        PolynomialCutoff(r_cut=r_cut, p=p)
