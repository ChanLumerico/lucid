"""A custom ``Function`` that returns ``None`` for an input gives it no gradient.

The engine carries that ``None`` as an empty storage.  ``backward`` used to
route it on like any other gradient, so a leaf reached only through it got
a ``.grad`` of its own shape over no memory — read back as whatever the
allocator had last left there.  In a compiled training step it showed up
as ``1e-7``-sized values and a NaN in a gradient eager computed as zero,
and only after another test had run first.

A native op's empty result is different: it is the real gradient of a
zero-size tensor, which a CPU allocation of zero bytes also leaves empty.
"""

import pytest

import lucid
from lucid.autograd import Function

DEVICES = ["cpu", "metal"]


class _FirstOnly(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a * 1.0 + b * 0.0

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _dirty_the_allocator(device: str) -> None:
    # Leave values in the pool an unwritten gradient would read back.
    (lucid.randn(4, 6, device=device) * 3.0).sum().item()


@pytest.mark.parametrize("device", DEVICES)
def test_backward_leaves_the_none_input_without_a_gradient(device: str) -> None:
    _dirty_the_allocator(device)
    a = lucid.ones(4, 6, device=device, requires_grad=True)
    b = lucid.ones(4, 6, device=device, requires_grad=True)
    _FirstOnly.apply(a, b).sum().backward()
    assert float(a.grad.sum().item()) == 24.0
    assert b.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_grad_reports_the_none_input_as_unused(device: str) -> None:
    _dirty_the_allocator(device)
    a = lucid.ones(4, 6, device=device, requires_grad=True)
    b = lucid.ones(4, 6, device=device, requires_grad=True)
    (gb,) = lucid.autograd.grad(_FirstOnly.apply(a, b).sum(), [b], allow_unused=True)
    assert gb is None


@pytest.mark.parametrize("device", DEVICES)
def test_a_second_path_still_reaches_the_none_input(device: str) -> None:
    b = lucid.ones(4, 6, device=device, requires_grad=True)
    (_FirstOnly.apply(b * 2.0, b) + b).sum().backward()
    assert float(b.grad.sum().item()) == 3.0 * 24


@pytest.mark.parametrize("device", DEVICES)
def test_a_zero_size_leaf_keeps_its_empty_gradient(device: str) -> None:
    p = lucid.zeros(0, 3, device=device, requires_grad=True)
    (p * 2.0).sum().backward()
    assert p.grad is not None
    assert tuple(p.grad.shape) == (0, 3)
