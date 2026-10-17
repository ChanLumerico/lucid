"""A leaf reports no ``grad_fn``, however it has been used.

The engine keeps a leaf's gradient accumulator in the leaf's ``grad_fn``
slot once the leaf enters a graph, and ``Tensor.grad_fn`` handed that
node out: a leaf that had been through an op or a backward answered
``AccumulateGrad`` where the reference answers ``None``, so ``is_leaf``
and ``grad_fn is None`` disagreed (LCD-499).  Every case below holds the
two to each other and to the reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid


def _fresh(lib, device):
    return lib.ones(3, requires_grad=True, device=device)


def _used_in_an_op(lib, device):
    x = lib.ones(3, requires_grad=True, device=device)
    x * 2
    return x


def _after_backward(lib, device):
    x = lib.ones(3, requires_grad=True, device=device)
    (x * 2).sum().backward()
    return x


def _backward_root(lib, device):
    x = lib.ones(3, requires_grad=True, device=device)
    x.backward(lib.ones(3, device=device))
    return x


def _parameter_after_backward(lib, device):
    layer = lib.nn.Linear(2, 2).to(device)
    layer(lib.ones(1, 2, device=device)).sum().backward()
    return layer.weight


def _without_grad(lib, device):
    return lib.ones(3, device=device)


def _detached(lib, device):
    x = lib.ones(3, requires_grad=True, device=device)
    return (x * 2).detach()


def _made_under_no_grad(lib, device):
    x = lib.ones(3, requires_grad=True, device=device)
    with lib.no_grad():
        return x * 2


def _op_result(lib, device):
    return lib.ones(3, requires_grad=True, device=device) * 2


def _view_of_a_leaf(lib, device):
    return lib.ones(3, requires_grad=True, device=device)[0:2]


def _clone_of_a_leaf(lib, device):
    return lib.ones(3, requires_grad=True, device=device).clone()


def _in_place_on_an_op_result(lib, device):
    y = lib.ones(3, requires_grad=True, device=device) * 2
    y.add_(1)
    return y


_CASES: dict[str, Callable[[ModuleType, str], object]] = {
    "fresh leaf": _fresh,
    "leaf used in an op": _used_in_an_op,
    "leaf after backward": _after_backward,
    "leaf as the backward root": _backward_root,
    "parameter after backward": _parameter_after_backward,
    "tensor without grad": _without_grad,
    "detached result": _detached,
    "result made under no_grad": _made_under_no_grad,
    "op result": _op_result,
    "view of a leaf": _view_of_a_leaf,
    "clone of a leaf": _clone_of_a_leaf,
    "in-place on an op result": _in_place_on_an_op_result,
}


def _flags(t):
    return t.is_leaf, t.grad_fn is None


@pytest.mark.parametrize("case", sorted(_CASES))
def test_is_leaf_is_having_no_grad_fn(case: str, device: str) -> None:
    is_leaf, no_grad_fn = _flags(_CASES[case](lucid, device))
    assert is_leaf is no_grad_fn


@pytest.mark.parity
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_flags_match_the_reference(ref: ModuleType, case: str, device: str) -> None:
    # The flags do not depend on where a tensor lives, so the reference
    # runs on its CPU against each Lucid device.
    assert _flags(_CASES[case](lucid, device)) == _flags(_CASES[case](ref, "cpu"))


def test_a_leaf_still_receives_its_gradient(device: str) -> None:
    # The accumulator the property no longer reports is still the graph's
    # sink: a backward through an op result lands in the leaf's ``.grad``.
    x = lucid.ones(3, requires_grad=True, device=device)
    y = x * 2
    assert x.grad_fn is None and y.grad_fn is not None
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0, 2.0]
    assert x.grad_fn is None
