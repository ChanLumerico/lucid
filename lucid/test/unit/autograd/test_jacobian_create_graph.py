"""Regression test: ``jacobian`` was not differentiable through.

Its *values* were right — they match the reference exactly — so the
defect only showed when something differentiated the result.  Each row
was produced by zeroing ``x``'s gradient slot, running a backward pass
into it, and taking what landed there, so every row referred to the same
slot; under ``create_graph=True`` only the last one still had a graph
attached by the time the rows were stacked.

``d/dx Σ jacobian(x², x)`` answered ``[0, 0, 6]``.  The Jacobian is
``diag(2x)``, its sum is ``2Σx``, and the derivative is ``[2, 2, 2]``.
"""

import numpy as np

import lucid
import lucid.autograd


def test_jacobian_values() -> None:
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    jac = np.asarray(lucid.autograd.jacobian(lambda a: a * a, x).numpy())
    assert np.allclose(jac, np.diag([2.0, 4.0, 6.0]))


def test_jacobian_is_differentiable_under_create_graph() -> None:
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    lucid.autograd.jacobian(lambda a: a * a, x, create_graph=True).sum().backward()
    assert np.allclose(np.asarray(x.grad.numpy()).ravel(), [2.0, 2.0, 2.0])


def test_jacobian_of_a_coupled_function() -> None:
    """Off-diagonal rows too, so the fix is not a diagonal coincidence."""

    def f(a: lucid.Tensor) -> lucid.Tensor:
        return lucid.stack([a[0] * a[1], a[1] * a[1]])

    x = lucid.tensor(np.array([2.0, 3.0]), requires_grad=True)
    jac = np.asarray(lucid.autograd.jacobian(f, x).numpy())
    assert np.allclose(jac, [[3.0, 2.0], [0.0, 6.0]])

    y = lucid.tensor(np.array([2.0, 3.0]), requires_grad=True)
    lucid.autograd.jacobian(f, y, create_graph=True).sum().backward()
    assert np.allclose(np.asarray(y.grad.numpy()).ravel(), [1.0, 3.0])
