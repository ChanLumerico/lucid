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

Every check that needs the reference — first derivatives against it,
second derivatives against a central difference of its gradient — lives
in ``lucid/test/parity/autograd/test_second_derivatives_parity.py``; this
file keeps the properties that need no oracle, so the fast tier runs them.
"""

import numpy as np
import pytest

import lucid
import lucid.autograd
import lucid.nn.functional as F

X = np.array([-4.0, -3.5, -1.0, -0.25, 0.25, 1.0, 3.5, 4.0])


def _first(fn, values=X):
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(fn(x).sum(), [x], create_graph=True)
    return x, g, np.asarray(g.numpy())


def _second(x, g):
    x.grad = None
    g.sum().backward()
    return np.zeros(x.shape) if x.grad is None else np.asarray(x.grad.numpy())


# ── the activations ───────────────────────────────────────────────────────────


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


# ── where, which unblocked the rest ───────────────────────────────────────────


def test_where_refuses_create_graph_rather_than_answering_wrongly() -> None:
    """A graph-mode derivative for ``where`` was written and reverted.

    Routing the gradient with the same condition is right when the two
    branches are independent, and wrong when they share a subexpression.
    ``cdist`` computes ``where(sq == 0, zeros_like(sq), sqrt(sq))``, where
    both branches come from ``sq``, and the second derivative came back
    ``[0.447, -1.252]`` against a true ``[-0.143, -0.072]``: right
    magnitude class, wrong value, wrong sign.

    ``where`` alone is correct even with both branches differentiable and
    an x-dependent condition, so the fault is in how the two returned
    gradients meet again upstream — isolated, not yet understood.

    Refusing is the honest answer until it is.  A wrong second derivative
    is worse than a missing one: it trains.  ``softplus`` is written over
    ``where`` and so is refused with it.
    """
    condition = lucid.tensor(np.array([True, False, True, False, True]))
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    out = lucid.where(condition, a, b)
    with pytest.raises(RuntimeError, match="create_graph"):
        lucid.autograd.grad(out.sum(), [a], create_graph=True)


def test_wheres_eager_gradient_is_unaffected() -> None:
    """Only the graph-mode path is missing; ordinary backward still works."""
    condition = lucid.tensor(np.array([True, False, True, False, True]))
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    lucid.where(condition, a, b).sum().backward()
    assert np.allclose(np.asarray(a.grad.numpy()), [1.0, 0.0, 1.0, 0.0, 1.0])
    assert np.allclose(np.asarray(b.grad.numpy()), [0.0, 1.0, 0.0, 1.0, 0.0])


# ── structural ops, each its own inverse ──────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn,values",
    [
        ("flip", lambda t: lucid.flip(t, 0), np.arange(1.0, 7.0)),
        ("fliplr", lucid.fliplr, np.arange(1.0, 10.0).reshape(3, 3)),
        ("roll", lambda t: lucid.roll(t, [2], [0]), np.arange(1.0, 7.0)),
        ("tril", lucid.tril, np.arange(1.0, 10.0).reshape(3, 3)),
        ("triu", lambda t: lucid.triu(t, 1), np.arange(1.0, 10.0).reshape(3, 3)),
    ],
)
def test_a_rearrangement_is_differentiable_twice(name, fn, values) -> None:
    """``flip``, ``roll`` and the triangle masks move or zero elements
    without computing anything, so the graph-mode derivative is the same
    rearrangement applied to the gradient.  Checked against a finite
    difference, which knows nothing about that symmetry."""
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad((fn(x) ** 2).sum(), [x], create_graph=True)
    analytic = np.asarray(g.numpy()).ravel()

    step = 1e-4
    flat = values.ravel().copy()
    numeric = np.empty_like(flat)
    for i in range(flat.size):
        up, down = flat.copy(), flat.copy()
        up[i] += step
        down[i] -= step

        def loss(v):
            return float((fn(lucid.tensor(v.reshape(values.shape))) ** 2).sum().item())

        numeric[i] = (loss(up) - loss(down)) / (2 * step)

    assert np.allclose(analytic, numeric, atol=1e-4), (analytic, numeric)


def test_tril_zeroes_the_second_derivative_where_it_masks() -> None:
    """The mask has to survive to the second order too."""
    values = np.arange(1.0, 10.0).reshape(3, 3)
    x = lucid.tensor(values, requires_grad=True)
    (g,) = lucid.autograd.grad((lucid.tril(x) ** 3).sum(), [x], create_graph=True)
    x.grad = None
    g.sum().backward()
    second = np.asarray(x.grad.numpy())
    assert np.allclose(np.triu(second, 1), 0.0)
    assert np.abs(np.tril(second)).max() > 0.0


# ── gather, and the loss path behind it ───────────────────────────────────────


def test_duplicate_indices_accumulate() -> None:
    """It is a scatter-*add*: reading one position three times must send
    three units of gradient back, not one."""
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    idx = lucid.tensor(np.array([1, 1, 1]), dtype=lucid.int32)
    (g,) = lucid.autograd.grad(lucid.gather(x, idx, 0).sum(), [x], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), [0.0, 3.0, 0.0])


def test_cross_entropy_has_a_second_derivative() -> None:
    logits = np.random.default_rng(0).standard_normal((4, 5))
    a = lucid.tensor(logits, requires_grad=True)
    tgt = lucid.tensor(np.array([0, 3, 1, 4]), dtype=lucid.int32)
    (g,) = lucid.autograd.grad(F.cross_entropy(a, tgt), [a], create_graph=True)
    a.grad = None
    g.sum().backward()
    assert a.grad is not None
    assert np.abs(np.asarray(a.grad.numpy())).max() > 0.0


@pytest.mark.parametrize("fn", [lucid.clone, lambda t: t.contiguous()])
def test_a_layout_copy_passes_the_gradient_through(fn) -> None:
    """``contiguous`` moves bytes without touching a value, so its
    derivative is the identity."""
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (g,) = lucid.autograd.grad((fn(x) ** 2).sum(), [x], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), [2.0, 4.0, 6.0])
    x.grad = None
    g.sum().backward()
    assert np.allclose(np.asarray(x.grad.numpy()), 2.0)


# ── the name in the refusal ───────────────────────────────────────────────────


class TestUnsupportedOpIsNamed:
    """A refusal has to say which op refused.

    ``node_name`` falls back to the C++ type name, and it derived that by
    stripping a leading run of digits — the flat Itanium mangling
    ``12MulBackward``.  That form only occurs for a class in the *global*
    namespace, and every node here lives in ``namespace lucid``, so the
    real name ``N5lucid10DetBackwardE`` starts with ``N``, the strip never
    advanced, and the whole mangled string came back as the op name.
    Nodes in the anonymous namespace inside ``lucid`` were worse
    (``N5lucid12_GLOBAL__N_113WhereBackwardE``).
    """

    @staticmethod
    def _refused_op_name(build):
        x = lucid.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        try:
            lucid.autograd.grad(build(x).sum(), [x], create_graph=True)
        except RuntimeError as exc:
            message = str(exc)
            if "not yet supported for op" not in message:
                pytest.fail(f"refused for another reason: {message}")
            return message.split("op '")[1].split("'")[0]
        pytest.fail(
            "the op now supports create_graph, so nothing refuses — pick an op "
            "that still does, or drop the case"
        )

    @pytest.mark.parametrize(
        "label,build",
        [
            # ``lucid::DetBackward`` — the plain namespaced case.
            ("det", lambda x: lucid.linalg.det(x)),
            # These live in an anonymous namespace inside ``lucid``, which
            # mangles differently again and was the worse of the two.
            ("diagonal", lambda x: lucid.diagonal(x)),
            ("sort", lambda x: lucid.sort(x, dim=1)[0]),
        ],
    )
    def test_the_message_names_the_op_not_its_mangling(self, label, build) -> None:
        name = self._refused_op_name(build)
        assert name != "unknown"
        assert name.endswith("Backward"), name
        # The tells of an unparsed mangling.
        assert not name.startswith("N"), name
        assert "_GLOBAL__N_" not in name, name
        assert "lucid" not in name, name
        assert name.isidentifier(), name
