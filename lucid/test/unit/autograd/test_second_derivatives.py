"""Eleven ops gained a graph-mode derivative, and one unblocked the rest.

An op needs ``grad_formula_impl`` for ``grad(create_graph=True)`` to work
— an eager ``grad_formula`` computes a gradient but not a differentiable
one.  Eight activations and three comparisons had only the eager form and
refused the second derivative outright.

``where`` was the load-bearing one.  It had ``apply`` and no
``apply_for_graph``, and the base class raises before an op name is
available, so the message read "not yet supported for op 'unknown'" — the
largest single unexplained bucket in the audit.  ``where`` is how every
piecewise function is written, so ``softplus``, ``celu``, ``prelu`` and
the rest inherited a refusal from a composite they had no say in.

The reference cannot differentiate some of its own backward kernels
(``derivative for aten::hardsigmoid_backward is not implemented``), so it
is the arbiter for the *first* derivative only; the second is checked
against a central difference of the reference's gradient.
"""

import numpy as np
import pytest

import lucid
import lucid.autograd
import lucid.nn.functional as F
from lucid.test._fixtures.ref_framework import require_ref

X = np.array([-4.0, -3.5, -1.0, -0.25, 0.25, 1.0, 3.5, 4.0])

ACTIVATIONS = [
    ("leaky_relu", F.leaky_relu, "leaky_relu"),
    ("softplus", F.softplus, "softplus"),
    ("elu", F.elu, "elu"),
    ("selu", F.selu, "selu"),
    ("mish", F.mish, "mish"),
    ("hardsigmoid", F.hardsigmoid, "hardsigmoid"),
    ("hardswish", F.hardswish, "hardswish"),
    ("relu6", F.relu6, "relu6"),
]


def _first(fn, values=X):
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(fn(x).sum(), [x], create_graph=True)
    return x, g, np.asarray(g.numpy())


def _second(x, g):
    x.grad = None
    g.sum().backward()
    return np.zeros(x.shape) if x.grad is None else np.asarray(x.grad.numpy())


# ── the activations ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,fn,ref_name", ACTIVATIONS)
def test_first_derivative_matches_the_reference(name, fn, ref_name) -> None:
    t = require_ref()
    _, _, got = _first(fn)
    r = t.from_numpy(X.copy()).requires_grad_(True)
    (rg,) = t.autograd.grad(
        getattr(t.nn.functional, ref_name)(r).sum(), [r], create_graph=True
    )
    assert np.allclose(got, np.asarray(rg.tolist()), atol=1e-6)


@pytest.mark.parametrize("name,fn,ref_name", ACTIVATIONS)
def test_second_derivative_matches_a_finite_difference(name, fn, ref_name) -> None:
    """Of the *reference's* first derivative, so the check does not lean
    on the implementation it is checking."""
    t = require_ref()
    x, g, _ = _first(fn)
    got = _second(x, g)

    def ref_first_at(values, index):
        r = t.from_numpy(values).requires_grad_(True)
        (rg,) = t.autograd.grad(getattr(t.nn.functional, ref_name)(r).sum(), [r])
        return np.asarray(rg.tolist())[index]

    step = 1e-4
    numeric = np.empty_like(X)
    for i in range(X.size):
        up, down = X.copy(), X.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (ref_first_at(up, i) - ref_first_at(down, i)) / (2 * step)
    assert np.allclose(got, numeric, atol=2e-3), (got, numeric)


def test_a_piecewise_linear_second_derivative_is_zero() -> None:
    """``relu6`` is linear on each piece, so ``d2`` is genuinely zero away
    from the kinks — the truthful answer, not a missing gradient."""
    x, g, _ = _first(F.relu6, np.array([-1.0, 1.0, 3.0, 7.0]))
    assert np.allclose(_second(x, g), 0.0)


def test_mish_has_a_curved_second_derivative() -> None:
    """Guard the guard: if every case were flat, the tests above would
    pass on a formula that returned zero."""
    x, g, _ = _first(F.mish)
    assert np.abs(_second(x, g)).max() > 0.1


# ── the comparisons ───────────────────────────────────────────────────────────

A = np.array([-2.0, -0.5, 0.5, 1.5, 3.0])
B = np.array([-1.0, 0.5, -0.5, 2.5, 1.0])


@pytest.mark.parametrize(
    "name,lf,rf",
    [("maximum", lucid.maximum, "maximum"), ("minimum", lucid.minimum, "minimum")],
)
@pytest.mark.parametrize("wrt", ["a", "b"])
def test_comparison_first_derivative(name, lf, rf, wrt) -> None:
    t = require_ref()
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(
        lf(a, b).sum(), [a if wrt == "a" else b], create_graph=True
    )
    ra = t.from_numpy(A.copy()).requires_grad_(True)
    rb = t.from_numpy(B.copy()).requires_grad_(True)
    (rg,) = t.autograd.grad(
        getattr(t, rf)(ra, rb).sum(), [ra if wrt == "a" else rb], create_graph=True
    )
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()), atol=1e-8)


def test_the_two_branches_sum_to_the_incoming_gradient() -> None:
    """Nothing created, nothing lost: a tie must not send the gradient to
    both operands, and a win must not drop it."""
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    out = lucid.maximum(a, b)
    (ga,) = lucid.autograd.grad(out.sum(), [a], create_graph=True, retain_graph=True)
    (gb,) = lucid.autograd.grad(out.sum(), [b], create_graph=True)
    total = np.asarray(ga.numpy()) + np.asarray(gb.numpy())
    assert np.allclose(total, 1.0)


def test_clip_first_derivative() -> None:
    t = require_ref()
    x = lucid.tensor(A.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(lucid.clip(x, -1.0, 2.0).sum(), [x], create_graph=True)
    r = t.from_numpy(A.copy()).requires_grad_(True)
    (rg,) = t.autograd.grad(t.clip(r, -1.0, 2.0).sum(), [r], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()), atol=1e-8)


# ── where, which unblocked the rest ───────────────────────────────────────────


def test_where_supports_create_graph() -> None:
    condition = lucid.tensor(np.array([True, False, True, False, True]))
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    out = lucid.where(condition, a, b)
    (ga,) = lucid.autograd.grad(out.sum(), [a], create_graph=True, retain_graph=True)
    (gb,) = lucid.autograd.grad(out.sum(), [b], create_graph=True)
    assert np.allclose(np.asarray(ga.numpy()), [1.0, 0.0, 1.0, 0.0, 1.0])
    assert np.allclose(np.asarray(gb.numpy()), [0.0, 1.0, 0.0, 1.0, 0.0])


def test_a_composite_over_where_is_differentiable_twice() -> None:
    """``softplus`` is ``where(bx > threshold, x, softplus(bx)/beta)``, so
    it inherited ``where``'s refusal despite having its own formula."""
    x, g, first = _first(F.softplus)
    assert np.allclose(first, 1.0 / (1.0 + np.exp(-X)), atol=1e-6)
    assert np.abs(_second(x, g)).max() > 0.0
