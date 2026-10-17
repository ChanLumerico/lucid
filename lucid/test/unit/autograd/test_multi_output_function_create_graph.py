"""``create_graph`` through a custom Function with several outputs (LCD-152).

A Function returning several tensors is a barrier: its outputs' gradients
arrive separately and its ``backward`` runs once on them all.  The eager
pass collected them per output; a ``create_graph`` pass had no way to, and
refused with NotImplementedError where the reference differentiates the
Function's ``backward`` like any other code.  It now parks each output's
gradient as the tensor it arrived as.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str], object]


def _values(t: object) -> object:
    if t is None:
        return None
    # Four places: Metal's float32 sin and cos differ from the CPU's in the
    # fifth.
    return [round(float(v), 4) for v in t.detach().cpu().reshape(-1).tolist()]


def _sin_cos(m: ModuleType) -> type:
    class SinCos(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            ctx.save_for_backward(a)
            return a.sin(), a.cos()

        @staticmethod
        def backward(ctx: object, gs: object, gc: object) -> object:
            (a,) = ctx.saved_tensors
            return gs * a.cos() - gc * a.sin()

    return SinCos


def _pair(m: ModuleType) -> type:
    class Pair(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object, b: object) -> object:
            ctx.save_for_backward(a, b)
            return a * b, a + b

        @staticmethod
        def backward(ctx: object, g0: object, g1: object) -> object:
            a, b = ctx.saved_tensors
            return g0 * b + g1, g0 * a + g1

    return Pair


def _second_derivative(m: ModuleType, device: str) -> object:
    x = m.tensor([0.5, 1.0, 2.0], requires_grad=True, device=device)
    s, c = _sin_cos(m).apply(x)
    y = (s * s + 3.0 * c).sum()
    (g,) = m.autograd.grad(y, x, create_graph=True)
    (gg,) = m.autograd.grad(g.sum(), x)
    return [_values(g), _values(gg), bool(g.requires_grad)]


def _two_inputs(m: ModuleType, device: str) -> object:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    p, q = _pair(m).apply(a, b)
    y = (p * q).sum()
    ga, gb = m.autograd.grad(y, (a, b), create_graph=True)
    (gga,) = m.autograd.grad((ga * gb).sum(), a)
    return [_values(ga), _values(gb), _values(gga)]


def _one_output_unused(m: ModuleType, device: str) -> object:
    x = m.tensor([0.5, 1.0], requires_grad=True, device=device)
    s, _ = _sin_cos(m).apply(x)
    (g,) = m.autograd.grad((s * s).sum(), x, create_graph=True)
    (gg,) = m.autograd.grad(g.sum(), x)
    return [_values(g), _values(gg)]


def _backward_create_graph(m: ModuleType, device: str) -> object:
    x = m.tensor([0.5, 1.0], requires_grad=True, device=device)
    s, c = _sin_cos(m).apply(x)
    (s * c).sum().backward(create_graph=True)
    first = _values(x.grad)
    (gg,) = m.autograd.grad(x.grad.sum(), x)
    return [first, _values(gg)]


def _output_hook(m: ModuleType, device: str) -> object:
    x = m.tensor([0.5, 1.0], requires_grad=True, device=device)
    s, c = _sin_cos(m).apply(x)
    seen: list[object] = []
    c.register_hook(lambda g: seen.append(_values(g)))
    (g,) = m.autograd.grad((s + c * 2.0).sum(), x, create_graph=True)
    return [seen, _values(g)]


_SCENARIOS: dict[str, Scenario] = {
    "second-derivative": _second_derivative,
    "two-inputs": _two_inputs,
    "one-output-unused": _one_output_unused,
    "backward-create-graph": _backward_create_graph,
    "output-hook": _output_hook,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    assert _SCENARIOS[name](lucid, device) == _SCENARIOS[name](ref, "cpu")


def test_a_once_differentiable_backward_is_still_refused(device: str) -> None:
    class Split(lucid.autograd.Function):
        _once_differentiable = True

        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2.0, a * 3.0

        @staticmethod
        def backward(ctx: object, g0: object, g1: object) -> object:
            return g0 * 2.0 + g1 * 3.0

    x = lucid.ones(2, requires_grad=True, device=device)
    y0, y1 = Split.apply(x)
    with pytest.raises(NotImplementedError, match="once_differentiable"):
        lucid.autograd.grad((y0 + y1).sum(), x, create_graph=True)
