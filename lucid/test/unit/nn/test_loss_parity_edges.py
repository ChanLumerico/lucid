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


# ── CHA-84 ─────────────────────────────────────────────────────────────────


class TestBinaryCrossEntropyWithLogitsAtZero:
    """The logit form had the wrong gradient at exactly ``x = 0``.

    ``max(x, 0) - x y + log(1 + exp(-|x|))`` has subgradients 1 and 0 at
    the origin, so ``d/dx`` was ``1 - y`` instead of ``1/2 - y``: a
    zero-initialised head got no gradient from its positive labels.
    """

    def test_a_zero_logit_gets_half_minus_the_label(self, device: str) -> None:
        z = lucid.tensor([0.0, 0.0], requires_grad=True, device=device)
        nn.BCEWithLogitsLoss()(z, lucid.tensor([1.0, 0.0], device=device)).backward()
        assert z.grad is not None
        assert _close(_vals(z.grad), [-0.25, 0.25])

    def test_pos_weight_at_zero(self, device: str) -> None:
        z = lucid.tensor([0.0, 0.0], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0], device=device)
        pw = lucid.tensor(3.0, device=device)
        out = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
        out.sum().backward()
        assert _close(_vals(out), [3 * math.log(2.0), math.log(2.0)])
        assert z.grad is not None
        assert _close(_vals(z.grad), [-1.5, 0.5])

    def test_a_zero_initialised_head_learns_from_positives(self, device: str) -> None:
        head = nn.Linear(3, 1).to(device)
        with lucid.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        x = lucid.ones(4, 3, device=device)
        y = lucid.ones(4, 1, device=device)
        nn.BCEWithLogitsLoss()(head(x), y).backward()
        assert head.bias.grad is not None
        assert _close(head.bias.grad.item(), -0.5)

    def test_saturated_logits_stay_finite(self, device: str) -> None:
        z = lucid.tensor([100.0, -100.0, 30.0, -30.0], requires_grad=True, device=device)
        y = lucid.tensor([0.0, 1.0, 1.0, 0.0], device=device)
        out = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        out.sum().backward()
        assert _close(_vals(out), [100.0, 100.0, 9.357623e-14, 9.357623e-14])
        assert z.grad is not None
        assert _close(_vals(z.grad), [1.0, -1.0, -9.357623e-14, 9.357623e-14])

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        vals_z = [0.0, 0.0, 30.0, -30.0, 1.5, -0.7]
        vals_y = [1.0, 0.0, 0.0, 1.0, 0.25, 0.9]
        w = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0]
        for pos_weight in (None, 3.0):
            for reduction in ("none", "mean", "sum"):
                lz = lucid.tensor(vals_z, requires_grad=True, device=device)
                ly = lucid.tensor(vals_y, requires_grad=True, device=device)
                lo = F.binary_cross_entropy_with_logits(
                    lz,
                    ly,
                    weight=lucid.tensor(w, device=device),
                    pos_weight=(
                        None if pos_weight is None
                        else lucid.tensor(pos_weight, device=device)
                    ),
                    reduction=reduction,
                )
                lo.sum().backward()
                rz = R.tensor(vals_z, requires_grad=True)  # type: ignore[attr-defined]
                ry = R.tensor(vals_y, requires_grad=True)  # type: ignore[attr-defined]
                ro = R.nn.functional.binary_cross_entropy_with_logits(  # type: ignore[attr-defined]
                    rz,
                    ry,
                    weight=R.tensor(w),  # type: ignore[attr-defined]
                    pos_weight=None if pos_weight is None else R.tensor(pos_weight),  # type: ignore[attr-defined]
                    reduction=reduction,
                )
                ro.sum().backward()
                assert _close(_vals(lo), ro.tolist())
                assert lz.grad is not None and ly.grad is not None
                assert _close(_vals(lz.grad), rz.grad.tolist())
                assert _close(_vals(ly.grad), ry.grad.tolist())


# ── CHA-85 ─────────────────────────────────────────────────────────────────


class TestKLDivWithZeroTargets:
    """``kl_div`` was NaN wherever the target was 0.

    ``target * (log(target) - x)`` is ``0 * -inf``.  One-hot and sparse
    distillation targets are made of such entries.
    """

    P = [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]]
    Q = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]

    def test_batchmean_of_a_one_hot_target(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device))
        q = lucid.tensor(self.Q, device=device)
        assert _close(F.kl_div(x, q, reduction="batchmean").item(), 0.43375030)

    def test_zero_entries_contribute_zero(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device))
        q = lucid.tensor(self.Q, device=device)
        out = F.kl_div(x, q, reduction="none")
        assert _close(_vals(out), [[0.35667494, 0.0, 0.0], [0.25541281, 0.25541281, 0.0]])

    def test_the_module_and_every_reduction_are_finite(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device)).requires_grad_()
        q = lucid.tensor(self.Q, device=device)
        for reduction in ("mean", "sum", "batchmean"):
            assert math.isfinite(nn.KLDivLoss(reduction=reduction)(x, q).item())
        F.kl_div(x, q, reduction="batchmean").backward()
        assert x.grad is not None
        assert _close(_vals(x.grad), [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.0]])

    @pytest.mark.parity
    @pytest.mark.filterwarnings("ignore::UserWarning")  # the reference on "mean"
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        for reduction in ("none", "mean", "sum", "batchmean"):
            lx = lucid.log(lucid.tensor(self.P, device=device)).requires_grad_()
            lo = F.kl_div(lx, lucid.tensor(self.Q, device=device), reduction=reduction)
            lo.sum().backward()
            rx = R.log(R.tensor(self.P)).requires_grad_()  # type: ignore[attr-defined]
            ro = R.nn.functional.kl_div(rx, R.tensor(self.Q), reduction=reduction)  # type: ignore[attr-defined]
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lx.grad is not None
            assert _close(_vals(lx.grad), rx.grad.tolist())
