"""A loss that cannot train anything says so, and a parameter cannot be bypassed.

Three ways a training loop used to run without error and leave weights
untouched, each refused now as the reference framework refuses it:

* ``backward()`` on a tensor with no graph returned quietly.  A loss built
  under ``no_grad()``, or through an op that did not track gradients,
  trained nothing and said nothing.
* An in-place write into a leaf that requires grad was refused only until
  the leaf had taken part in a forward: the check read "no ``grad_fn``",
  and a used leaf carries its gradient accumulator there.  After the first
  step, ``p.mul_(2)`` or ``p[0] = 0`` went through and turned the
  parameter into a non-leaf that never received ``.grad`` again.
* ``x[...] = v`` (and the in-place activations built on it) never checked
  at all.

And two places where refusing would have gone too far, because the
reference answers zeros: a gradient whose own derivative is zero
(``relu6``'s mask times a constant seed) and a tensor overwritten by
``fill_`` / ``zero_`` stay in the graph with that zero derivative.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.autograd import backward, grad


def test_backward_on_an_untracked_tensor_raises() -> None:
    with pytest.raises(RuntimeError, match="does not require grad"):
        lucid.tensor([1.0, 2.0]).sum().backward()
    x = lucid.randn(3, requires_grad=True)
    with lucid.no_grad():
        loss = (x * x).sum()
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()
    tracked = (x * 2).sum()
    with pytest.raises(RuntimeError, match="element 1"):
        backward([tracked, lucid.tensor(1.0)])


IN_PLACE = [
    ("abs_", lambda p: p.abs_()),
    ("mul_", lambda p: p.mul_(2.0)),
    ("fill_", lambda p: p.fill_(0.0)),
    ("zero_", lambda p: p.zero_()),
    ("copy_", lambda p: p.copy_(lucid.zeros(3))),
    ("setitem", lambda p: p.__setitem__(0, 0.0)),
    ("setitem whole", lambda p: p.__setitem__(slice(None), 1.0)),
    ("F.relu_", lambda p: F.relu_(p)),
]


@pytest.mark.parametrize("used", [False, True], ids=["fresh", "after a forward"])
@pytest.mark.parametrize("name,write", IN_PLACE, ids=[c[0] for c in IN_PLACE])
def test_a_leaf_that_requires_grad_refuses_in_place_writes(name, write, used) -> None:  # type: ignore[no-untyped-def]
    p = lucid.randn(3, requires_grad=True)
    if used:
        (p * 2).sum()
    with pytest.raises(Exception, match="leaf tensor that requires grad"):
        write(p)
    assert p.is_leaf


def test_no_grad_still_writes_parameters() -> None:
    layer = nn.Linear(3, 2)
    layer(lucid.randn(4, 3)).sum().backward()
    with lucid.no_grad():
        layer.weight.mul_(0.5)
        layer.weight[0] = 1.0
        layer.bias.zero_()
    assert layer.weight.is_leaf and layer.weight.requires_grad
    layer.weight.grad = None
    layer(lucid.randn(4, 3)).sum().backward()
    assert layer.weight.grad is not None


@pytest.mark.parametrize(
    "fn",
    [F.relu6, F.leaky_relu, F.hardtanh, lambda x: x.clamp(-1, 1)],
    ids=["relu6", "leaky_relu", "hardtanh", "clamp"],
)
def test_a_flat_second_derivative_is_zero_rather_than_unreachable(fn) -> None:  # type: ignore[no-untyped-def]
    x = lucid.tensor([-1.5, 0.5, 2.0, 7.0], requires_grad=True)
    (g,) = grad(fn(x).sum(), [x], create_graph=True)
    assert g.requires_grad
    (h,) = grad(g.sum(), [x], retain_graph=True)
    assert h.tolist() == [0.0, 0.0, 0.0, 0.0]
    x.grad = None
    g.sum().backward()
    assert x.grad.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_an_overwritten_tensor_keeps_a_zero_derivative() -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True)
    y = x * 3.0
    y.fill_(5.0)
    assert y.requires_grad and y.tolist() == [5.0, 5.0]
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 0.0]
    x.grad = None
    z = x * 3.0
    z.zero_()
    (z * 2 + x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0])


def test_a_second_pass_over_a_freed_graph_raises_instead_of_crashing() -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    first, second = y.sum(), (y * 3).sum()
    first.backward()
    with pytest.raises(Exception, match="second time"):
        second.backward()
    loss = (x * x).sum()
    loss.backward()
    with pytest.raises(Exception, match="second time"):
        loss.backward()
    kept = (x * x).sum()
    kept.backward(retain_graph=True)
    kept.backward()  # retained, so the second pass is fine


def test_a_graph_with_nothing_saved_may_run_twice() -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True)
    x.sum().backward()
    x.sum().backward()
    assert x.grad.tolist() == [2.0, 2.0]
