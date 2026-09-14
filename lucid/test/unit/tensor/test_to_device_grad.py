"""``.to()`` across devices is differentiable.

It returned a new leaf, so backward stopped at the move: the source of a
tensor moved to Metal and back never received a gradient, with no error.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal not available")


def test_gradient_reaches_a_cpu_leaf_through_metal() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.to("metal")
    assert y.is_metal and not y.is_leaf and y.requires_grad
    (y * y).sum().backward()
    assert x.grad is not None and not x.grad.is_metal
    assert x.grad.tolist() == [2.0, 4.0, 6.0]


def test_gradient_reaches_a_metal_leaf_through_cpu() -> None:
    x = lucid.tensor([1.0, 2.0], device="metal", requires_grad=True)
    (x.to("cpu") * 3.0).sum().backward()
    assert x.grad is not None and x.grad.is_metal
    assert x.grad.tolist() == [3.0, 3.0]


def test_a_move_and_a_cast_differentiate_together() -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True)
    x.to("metal", dtype=lucid.float16).float().sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == lucid.float32
    assert x.grad.tolist() == [1.0, 1.0]


def test_no_grad_still_returns_a_leaf() -> None:
    x = lucid.tensor([1.0], requires_grad=True)
    with lucid.no_grad():
        y = x.to("metal")
    assert y.is_leaf


def test_module_to_keeps_parameters_as_trainable_leaves() -> None:
    m = nn.Linear(2, 2).to("metal")
    assert all(p.is_leaf and p.requires_grad for p in m.parameters())
    m(lucid.ones(1, 2, device="metal")).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_second_derivative_through_a_move() -> None:
    x = lucid.tensor([2.0], requires_grad=True)
    y = (x.to("metal") ** 3).sum()
    (g,) = lucid.autograd.grad(y, x, create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert g.tolist() == [12.0]
    assert h.tolist() == [12.0]
