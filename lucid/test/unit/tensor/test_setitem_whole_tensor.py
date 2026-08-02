"""Regression tests: whole-tensor assignment took a general scatter.

Found 2026-08-02 while looking for a correct way to write the in-place
activations.  ``_setitem`` builds an ``arange`` of positions per
dimension, forms the flat index of the entire cross-product, and scatters
through it — even when the index selects every element and the operation
is a straight copy.  Measured on ``64x3x128x128``:

    x[:] = y        51.7 ms
    x.copy_(y)       0.63 ms

an **82x** penalty on one of the most common idioms in the language.  The
fast path added for the all-full-slices case brings it to 0.52 ms.

The tests here are about semantics, not speed.  A first draft of the fast
path bound the value's own storage into ``x`` and was 17,000x faster
because it copied nothing — which made ``x[:] = y`` alias, so a later
``y.copy_(...)`` silently rewrote ``x``.  That is what
``test_does_not_alias_the_source`` exists to stop, and it is why the fast
path materialises through ``contiguous``.
"""

import numpy as np
import pytest

import lucid


def test_does_not_alias_the_source() -> None:
    """The trap: the general path scatters into a fresh buffer, so must this.

    Both mutators are needed.  ``copy_`` and ``add_`` write through to
    storage, while ``y[0] = ...`` rebinds and would hide the aliasing.
    """
    x = lucid.zeros((3,))
    y = lucid.ones((3,))
    x[:] = y
    y.copy_(lucid.tensor([9.0, 9.0, 9.0]))
    assert np.allclose(x.numpy(), [1.0, 1.0, 1.0])

    a = lucid.zeros((3,))
    b = lucid.ones((3,))
    a[:] = b
    b.add_(lucid.tensor([5.0, 5.0, 5.0]))
    assert np.allclose(a.numpy(), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("index", [slice(None), Ellipsis])
def test_whole_tensor_assignment_values(index) -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    x[index] = lucid.tensor([[9.0, 8.0], [7.0, 6.0]])
    assert np.allclose(x.numpy(), [[9.0, 8.0], [7.0, 6.0]])


@pytest.mark.parametrize("index", [slice(None), Ellipsis])
def test_whole_tensor_assignment_scalar(index) -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    x[index] = 5.0
    assert np.allclose(x.numpy(), 5.0)


def test_whole_tensor_assignment_broadcasts() -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    x[:] = lucid.tensor([9.0, 8.0])
    assert np.allclose(x.numpy(), [[9.0, 8.0], [9.0, 8.0]])


def test_whole_tensor_assignment_keeps_the_destination_dtype() -> None:
    x = lucid.tensor([[1, 2], [3, 4]])
    x[:] = lucid.tensor([[9.7, 8.2], [7.1, 6.9]])
    assert str(x.dtype).endswith("int64")
    assert np.allclose(x.numpy(), [[9, 8], [7, 6]])


def test_whole_tensor_assignment_extends_the_autograd_graph() -> None:
    """The property the in-place activations were rebuilt on.

    ``Tensor.copy_`` severs the graph by documented design; indexed
    assignment must not, or the fix that depends on it is worthless.
    """
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), dtype=lucid.float64)
    x.requires_grad_(True)
    destination = lucid.tensor(np.zeros(3), dtype=lucid.float64) * 1.0
    destination[:] = x * x
    destination.sum().backward()
    assert np.allclose(x.grad.numpy(), [2.0, 4.0, 6.0])


def test_partial_assignment_still_works() -> None:
    """The general path must survive: the fast path is a special case, not a rewrite."""
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    x[0] = lucid.tensor([9.0, 8.0])
    assert np.allclose(x.numpy(), [[9.0, 8.0], [3.0, 4.0]])

    y = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    y[:, 0] = 0.0
    assert np.allclose(y.numpy(), [[0.0, 2.0], [0.0, 4.0]])

    z = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    z[0:1, :] = lucid.tensor([[7.0, 7.0]])
    assert np.allclose(z.numpy(), [[7.0, 7.0], [3.0, 4.0]])


def test_partial_full_slice_is_not_mistaken_for_the_whole_tensor() -> None:
    """``x[:, 0]`` contains a full slice but does not select everything."""
    x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x[:, 1] = 0.0
    assert np.allclose(x.numpy(), [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]])
