"""The edges of the loss functions, held to the reference (CHA-82).

A sweep of ``nn/functional/loss.py`` found the losses right on ordinary
inputs and wrong at the places training actually reaches: a probability
that is exactly 0 or 1, a logit that is exactly 0, a one-hot target, an
ignored index, a batch that is all padding, a half-precision batch longer
than 65504.  One class per issue; each states the case that was wrong,
checks the value written out on CPU and Metal, and — under ``-m parity`` —
the same case against the reference.
"""

import math

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


def _close(got: object, want: object, tol: float = 1e-5) -> bool:
    """Element-wise closeness of two (nested) lists or scalars, NaN equal to NaN."""
    if isinstance(got, (list, tuple)):
        assert isinstance(want, (list, tuple)) and len(got) == len(want)
        return all(_close(g, w, tol) for g, w in zip(got, want))
    g, w = float(got), float(want)  # type: ignore[arg-type]
    if math.isnan(w):
        return math.isnan(g)
    if math.isinf(w):
        return g == w
    return abs(g - w) <= tol * max(1.0, abs(w))


def _vals(t: lucid.Tensor) -> object:
    return t.detach().tolist()


# ── CHA-83 ─────────────────────────────────────────────────────────────────


class TestBinaryCrossEntropyAtTheBoundary:
    """``binary_cross_entropy`` was NaN at a probability of exactly 0 or 1.

    The input was clamped to ``[1e-12, 1 - 1e-12]``, and ``1 - 1e-12`` is
    1.0 in float32: a sigmoid of a logit above ~17 is exactly 1, so
    ``log(1 - p)`` was -inf and ``0 * -inf`` was NaN.  The log terms are
    now clamped at -100 and the gradient's denominator floored at 1e-12.
    """

    def test_a_saturated_sigmoid_is_finite(self, device: str) -> None:
        z = lucid.tensor([20.0, -20.0, 0.0], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0, 1.0], device=device)
        loss = nn.BCELoss()(lucid.sigmoid(z), y)
        loss.backward()
        assert _close(loss.item(), 0.2310490608215332)
        assert z.grad is not None
        assert _close(_vals(z.grad), [0.0, 6.870512e-10, -0.16666667])

    def test_the_log_terms_are_clamped_at_minus_100(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        y = lucid.tensor([1.0, 0.0, 0.0, 1.0], device=device)
        out = F.binary_cross_entropy(p, y, reduction="none")
        assert _close(_vals(out), [100.0, 100.0, 0.0, 0.0])

    def test_the_gradient_denominator_is_floored(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.0, 1.0, 0.5], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0, 0.0, 1.0, 0.3], device=device)
        F.binary_cross_entropy(p, y, reduction="sum").backward()
        assert p.grad is not None
        assert _close(_vals(p.grad), [-1e12, 1e12, 0.0, 0.0, 0.8])

    def test_the_interior_keeps_its_second_derivative(self) -> None:
        p = lucid.tensor([0.3, 0.7], requires_grad=True)
        y = lucid.tensor([1.0, 0.0])
        (g,) = lucid.autograd.grad(F.binary_cross_entropy(p, y), [p], create_graph=True)
        (gg,) = lucid.autograd.grad(g.sum(), [p])
        assert _close(_vals(gg), [1.0 / (2 * 0.09), 1.0 / (2 * 0.09)])

    def test_float16_is_finite_at_the_boundary(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.5], dtype=lucid.float16, device=device)
        y = lucid.tensor([1.0, 0.0, 1.0], dtype=lucid.float16, device=device)
        out = F.binary_cross_entropy(p, y, reduction="none")
        assert out.dtype == lucid.float16
        assert _close(_vals(out), [100.0, 100.0, 0.6931], tol=1e-3)

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        vals_p = [0.0, 1.0, 0.0, 1.0, 1e-13, 0.5, 0.999]
        vals_y = [1.0, 0.0, 0.0, 1.0, 1.0, 0.3, 0.2]
        w = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0]
        for reduction in ("none", "mean", "sum"):
            lp = lucid.tensor(vals_p, requires_grad=True, device=device)
            lo = F.binary_cross_entropy(
                lp,
                lucid.tensor(vals_y, device=device),
                weight=lucid.tensor(w, device=device),
                reduction=reduction,
            )
            lo.sum().backward()
            rp = R.tensor(vals_p, requires_grad=True)  # type: ignore[attr-defined]
            ro = R.nn.functional.binary_cross_entropy(  # type: ignore[attr-defined]
                rp, R.tensor(vals_y), weight=R.tensor(w), reduction=reduction  # type: ignore[attr-defined]
            )
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lp.grad is not None
            assert _close(_vals(lp.grad), rp.grad.tolist(), tol=1e-4)
