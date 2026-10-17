"""``t.data = value`` swaps the storage under a tensor and keeps the tensor.

There was no setter, so ``param.data = t`` — the reference idiom for
re-initialising a weight or a hand-written update — raised "property 'data'
of 'Parameter' object has no setter" (LCD-478).  The reference keeps the
object, its ``requires_grad``, a module's registration of it and its
gradient, takes ``value``'s storage without a copy, and records no graph.
``Module._apply`` (``.to()`` / ``.double()``) swaps through the same setter.
"""

import pytest

import lucid
import lucid.nn as nn


def _holders(device):
    """A plain tensor that requires grad, and a module's Parameter."""
    module = nn.Linear(2, 3).to(device)
    return [
        ("tensor", lucid.ones(3, 2, device=device, requires_grad=True), None),
        ("parameter", module.weight, module),
    ]


@pytest.mark.parametrize("which", ["tensor", "parameter"])
def test_assignment_swaps_the_data_and_keeps_the_tensor(device, which):
    _, t, module = next(h for h in _holders(device) if h[0] == which)
    kind, before = type(t), id(t)
    value = lucid.arange(8.0, device=device).reshape(4, 2)
    t.data = value
    assert id(t) == before and type(t) is kind
    assert t.shape == (4, 2) and t.tolist() == value.tolist()
    assert t.requires_grad and t.is_leaf and t.grad_fn is None
    assert t.device == value.device
    if module is not None:
        assert any(p is t for p in module.parameters())
        assert module.weight is t


def test_the_tensor_still_trains_after_a_swap(device):
    module = nn.Linear(2, 1).to(device)
    module.weight.data = lucid.zeros(1, 2, device=device)
    module(lucid.ones(1, 2, device=device)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.tolist() == [[1.0, 1.0]]


def test_an_optimizer_steps_the_new_data(device):
    module = nn.Linear(2, 1).to(device)
    opt = lucid.optim.SGD(module.parameters(), lr=0.5)
    module.weight.data = lucid.zeros(1, 2, device=device)
    module(lucid.ones(1, 2, device=device)).sum().backward()
    opt.step()
    assert module.weight.tolist() == [[-0.5, -0.5]]


def test_a_gradient_that_still_fits_is_kept(device):
    t = lucid.ones(3, device=device, requires_grad=True)
    (t * 2.0).sum().backward()
    t.data = lucid.zeros(3, device=device)
    assert t.grad is not None and t.grad.tolist() == [2.0, 2.0, 2.0]
    t.data = t.data - 0.5 * t.grad  # the hand-written update
    assert t.tolist() == [-1.0, -1.0, -1.0] and t.grad is not None


def test_a_gradient_of_the_old_shape_is_dropped(device):
    t = lucid.ones(3, device=device, requires_grad=True)
    (t * 2.0).sum().backward()
    t.data = lucid.zeros(2, 2, device=device)
    assert t.grad is None


def test_the_value_keeps_its_own_flag_and_graph(device):
    src = lucid.ones(2, device=device, requires_grad=True)
    value = src * 5.0
    t = lucid.zeros(2, device=device)
    t.data = value
    assert not t.requires_grad and t.grad_fn is None
    assert value.requires_grad and value.grad_fn is not None


def test_the_storage_is_shared_with_the_value_and_not_with_old_views():
    value, t = lucid.zeros(2), lucid.ones(3)
    view = t[:2]
    t.data = value
    value.add_(5.0)
    assert t.tolist() == [5.0, 5.0]
    assert view.tolist() == [1.0, 1.0]


def test_an_integer_value_on_a_tensor_without_grad_is_taken():
    t = lucid.ones(2)
    t.data = lucid.zeros(2, dtype=lucid.int64)
    assert t.dtype == lucid.int64


@pytest.mark.parametrize(
    "case,error",
    [
        ("not a tensor", TypeError),
        ("integer into a tensor that requires grad", RuntimeError),
        ("not a leaf", RuntimeError),
    ],
)
def test_assignment_refusals_leave_the_tensor_alone(case, error):
    leaf = lucid.ones(2, requires_grad=True)
    t = leaf * 2.0 if case == "not a leaf" else leaf
    value = {
        "not a tensor": 3,
        "integer into a tensor that requires grad": lucid.zeros(2, dtype=lucid.int64),
        "not a leaf": lucid.zeros(2),
    }[case]
    impl = t._impl
    with pytest.raises(error):
        t.data = value
    assert t._impl is impl


@pytest.mark.parity
def test_assignment_matches_the_reference(ref):
    import numpy as np

    def run(lib, tensor_of):
        module = lib.nn.Linear(2, 2)
        w = module.weight
        module(tensor_of(np.ones((1, 2)))).sum().backward()
        w.data = tensor_of(np.arange(6.0).reshape(3, 2))
        grad = None if w.grad is None else w.grad.shape
        return (
            module.weight is w,
            type(w).__name__,
            w.requires_grad,
            tuple(w.shape),
            w.grad_fn is None,
            any(p is w for p in module.parameters()),
            np.asarray(w.detach().numpy()).tolist(),
            grad,
        )

    mine = run(lucid, lambda a: lucid.tensor(a.astype(np.float32)))
    theirs = run(ref, lambda a: ref.tensor(a.astype(np.float32)))
    # The reference keeps a gradient of the old shape; Lucid's gradient
    # always matches its tensor, so that one is dropped.
    assert mine[:-1] == theirs[:-1]
    assert theirs[-1] is not None and mine[-1] is None
