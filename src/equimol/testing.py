"""Property tests for Euclidean symmetry.

Check that any callable of coordinates respects E(3) (or a subgroup) by
applying random rigid motions to the input and comparing outputs.

Usage:
    from equimol.testing import assert_equivariant

    model = model.double().eval()
    assert_equivariant(
        lambda x: model(h, x, t, edge_index),
        x,
        output="equivariant",
    )

Output types (one per returned tensor; the last dim must be the coordinate dim
for "equivariant" and "coordinates"):
    - "invariant":   f(Rx + s) = f(x)              e.g. energies, scalar node states
    - "equivariant": f(Rx + s) = f(x) R^T          e.g. predicted noise, forces, displacements
    - "coordinates": f(Rx + s) = f(x) R^T + s      e.g. updated positions

Groups:
    - "E3":  rotations, reflections, translations (default; EGNN-style models)
    - "SE3": proper rotations and translations (chiral models)
    - "O3":  rotations and reflections only
    - "SO3": proper rotations only

Run checks in float64. Float32 round-off through deep message passing reaches
~1e-4 relative error, which hides real symmetry bugs of similar size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence, Union

import torch

OutputType = Literal["invariant", "equivariant", "coordinates"]
Group = Literal["E3", "SE3", "O3", "SO3"]

_DEFAULT_RTOL = {torch.float64: 1e-8, torch.float32: 1e-3}


@dataclass(frozen=True)
class EquivarianceReport:
    """Result of an equivariance check.

    Fields:
        - max_error: worst relative error over trials and outputs
        - errors: relative error per trial, per output, shape [trials][outputs]
        - determinants: det(R) per trial (+1 rotation, -1 reflection)
        - rtol: tolerance used for passed
    """

    max_error: float
    errors: list[list[float]] = field(repr=False)
    determinants: list[float]
    rtol: float

    @property
    def passed(self) -> bool:
        return self.max_error <= self.rtol


def random_orthogonal(
    dim: int = 3,
    *,
    proper: bool,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a Haar-random rotation (proper=True) or reflection (proper=False)."""
    a = torch.randn(dim, dim, dtype=dtype, generator=generator).to(device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if (torch.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return q


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.abs().max().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).abs().max() / scale).item())


def _as_tuple(out) -> tuple[torch.Tensor, ...]:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def check_equivariance(
    fn: Callable[[torch.Tensor], Union[torch.Tensor, Sequence[torch.Tensor]]],
    x: torch.Tensor,
    *,
    output: Union[OutputType, Sequence[OutputType]] = "equivariant",
    group: Group = "E3",
    trials: int = 4,
    translation_scale: float = 10.0,
    rtol: float | None = None,
    seed: int = 0,
) -> EquivarianceReport:
    """Measure how far fn deviates from the requested symmetry.

    Args:
        fn: Callable mapping coordinates [N, D] to a tensor or tuple of tensors.
            Close over every other input (h, t, edge_index, ...).
        x: Coordinates with shape [N, D].
        output: Output type, or one type per returned tensor.
        group: Symmetry group to test.
        trials: Number of random transforms. For E3/O3 the trials alternate
            between proper rotations and reflections.
        translation_scale: Std of random translations (ignored for O3/SO3).
        rtol: Allowed relative error. Defaults to 1e-8 for float64 and 1e-3
            for float32.
        seed: Seed for the transforms (does not touch the global RNG).

    Returns:
        EquivarianceReport
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [N, D], got {tuple(x.shape)}")
    if group not in ("E3", "SE3", "O3", "SO3"):
        raise ValueError(f"Unsupported group {group!r}")
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    rtol = rtol if rtol is not None else _DEFAULT_RTOL.get(x.dtype, 1e-3)
    dim = x.shape[-1]
    translate = group in ("E3", "SE3")
    reflections = group in ("E3", "O3")
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        reference = _as_tuple(fn(x))
    output_types = [output] * len(reference) if isinstance(output, str) else list(output)
    if len(output_types) != len(reference):
        raise ValueError(
            f"fn returned {len(reference)} tensors but {len(output_types)} output types were given"
        )

    errors: list[list[float]] = []
    determinants: list[float] = []
    for trial in range(trials):
        proper = not reflections or trial % 2 == 0
        rotation = random_orthogonal(
            dim, proper=proper, dtype=x.dtype, device=x.device, generator=generator
        )
        shift = torch.zeros(dim, dtype=x.dtype, device=x.device)
        if translate:
            shift = translation_scale * torch.randn(dim, dtype=x.dtype, generator=generator).to(x.device)

        with torch.no_grad():
            transformed = _as_tuple(fn(x @ rotation.T + shift))

        trial_errors = []
        for out_type, ref, out in zip(output_types, reference, transformed):
            if out_type == "invariant":
                expected = ref
            elif out_type == "equivariant":
                expected = ref @ rotation.T
            elif out_type == "coordinates":
                expected = ref @ rotation.T + shift
            else:
                raise ValueError(f"Unsupported output type {out_type!r}")
            trial_errors.append(_relative_error(out, expected))

        errors.append(trial_errors)
        determinants.append(float(torch.det(rotation).item()))

    max_error = max(max(row) for row in errors)
    return EquivarianceReport(max_error=max_error, errors=errors, determinants=determinants, rtol=rtol)


def assert_equivariant(
    fn: Callable[[torch.Tensor], Union[torch.Tensor, Sequence[torch.Tensor]]],
    x: torch.Tensor,
    **kwargs,
) -> EquivarianceReport:
    """Raise AssertionError if fn breaks the requested symmetry.

    Accepts the same keyword arguments as check_equivariance.
    """
    report = check_equivariance(fn, x, **kwargs)
    if not report.passed:
        worst_trial = max(range(len(report.errors)), key=lambda i: max(report.errors[i]))
        kind = "rotation" if report.determinants[worst_trial] > 0 else "reflection"
        raise AssertionError(
            f"Symmetry violated: relative error {report.max_error:.3e} > rtol {report.rtol:.1e} "
            f"(worst case: {kind}, per-output errors {report.errors[worst_trial]})"
        )
    return report


__all__ = [
    "EquivarianceReport",
    "assert_equivariant",
    "check_equivariance",
    "random_orthogonal",
]
