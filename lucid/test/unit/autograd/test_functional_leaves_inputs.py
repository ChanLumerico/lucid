"""``autograd.functional`` leaves the caller's tensors as they were (LCD-68).

``jacobian``, ``hessian``, ``vjp`` and ``jvp`` called ``x.requires_grad_(True)``
on the caller's tensor: afterwards a plain data tensor built a graph on every
op and refused in-place writes as a leaf that requires grad.  ``hessian``
also zeroed and refilled the caller's ``.grad``.  They now differentiate at
copies (``_differentiable_inputs``), as the reference does — a fresh leaf
sharing the input's storage, or, under ``create_graph`` for an input that
already requires grad, a view, so the result stays differentiable with
respect to it.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid.autograd import hessian, jacobian, jvp, vjp


def _square(x: lucid.Tensor) -> lucid.Tensor:
    return x * x


def _square_sum(x: lucid.Tensor) -> lucid.Tensor:
    return (x * x).sum()


_CALLS: dict[str, Callable[[lucid.Tensor, bool], object]] = {
    "jacobian": lambda x, cg: jacobian(_square, x, create_graph=cg),
    "hessian": lambda x, cg: hessian(_square_sum, x, create_graph=cg),
    "vjp": lambda x, cg: vjp(_square, x, lucid.ones_like(x), create_graph=cg),
    "jvp": lambda x, cg: jvp(_square, x, lucid.ones_like(x), create_graph=cg),
}


@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create-graph"])
@pytest.mark.parametrize("requires_grad", [False, True], ids=["data", "parameter"])
@pytest.mark.parametrize("api", list(_CALLS))
def test_the_inputs_flags_and_grad_are_unchanged(
    api: str, requires_grad: bool, create_graph: bool, device: str
) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=requires_grad)
    if requires_grad:
        x.grad = lucid.full((3,), 7.0, device=device)
    _CALLS[api](x, create_graph)
    assert x.requires_grad is requires_grad
    assert x.is_leaf
    if requires_grad:
        assert x.grad is not None and x.grad.tolist() == [7.0, 7.0, 7.0]
    else:
        assert x.grad is None
        x.mul_(2.0)  # still a plain tensor: in-place writes are allowed


def test_the_jacobian_is_still_right(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    J = jacobian(_square, x)
    assert J.tolist() == [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]


def test_a_create_graph_jacobian_differentiates_back_to_the_input(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    J = jacobian(_square, x, create_graph=True)
    (g,) = lucid.autograd.grad(J.sum(), x)
    assert g is not None and g.tolist() == [2.0, 2.0, 2.0]
