"""``autograd.grad`` by a tensor a barrier node produced (CHA-112).

A barrier node — the output side of a module with a backward hook, a
custom Function with several outputs — takes its gradients slot by slot,
and the engine's pending map holds only an empty placeholder for it.
``autograd.grad`` captured that placeholder as the requested gradient: a
tensor of the right shape over no memory, which crashed the process the
moment it was read on the CPU, and was garbage on Metal.  The engine now
gathers what reaches the requested tensor's own slot on the way in.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._helpers.compare import assert_close


def _compare(got: dict[str, object], want: dict[str, object]) -> None:
    assert got.keys() == want.keys()
    for key, w in want.items():
        g = got[key]
        if w is None or g is None:
            assert g is None and w is None, key
        else:
            assert_close(g, w, msg=key)


def _two_outputs(lib: ModuleType) -> type:
    class Two(lib.autograd.Function):  # type: ignore[name-defined, misc]
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2, a * 3

        @staticmethod
        def backward(ctx: object, g1: object, g2: object) -> object:
            return g1 * 2 + g2 * 3  # type: ignore[operator]

    return Two


def _hooked_linear(lib: ModuleType, device: str) -> object:
    lin = lib.nn.Linear(2, 2, bias=False)
    with lib.no_grad():
        lin.weight.copy_(lib.tensor([[1.0, 0.5], [-1.0, 2.0]]))
    lin = lin.to(device)
    lin.register_full_backward_hook(lambda mod, gi, go: None)
    return lin


def _x(lib: ModuleType, device: str) -> object:
    return lib.tensor([[1.0, -2.0]], device=device, requires_grad=True)


# ── scenarios: run on Lucid and on the reference ──────────────────────────


def _hooked_output(lib: ModuleType, device: str) -> dict[str, object]:
    lin = _hooked_linear(lib, device)
    y = lin(_x(lib, device))
    (g,) = lib.autograd.grad((3 * y).sum() + (y * y).sum(), y)
    return {"g": g}


def _hooked_output_create_graph(lib: ModuleType, device: str) -> dict[str, object]:
    lin = _hooked_linear(lib, device)
    x = _x(lib, device)
    y = lin(x)
    (g,) = lib.autograd.grad((3 * y).sum() + (y * y).sum(), y, create_graph=True)
    (h,) = lib.autograd.grad(g.sum(), x)
    return {"g": g, "h": h}


def _function_output(lib: ModuleType, device: str) -> dict[str, object]:
    a = lib.tensor([1.0, -0.5], device=device, requires_grad=True)
    o1, o2 = _two_outputs(lib).apply(a)
    (g,) = lib.autograd.grad((5 * o2).sum() + (o1 * o1).sum(), o2)
    return {"g": g}


def _function_output_create_graph(lib: ModuleType, device: str) -> dict[str, object]:
    a = lib.tensor([1.0, -0.5], device=device, requires_grad=True)
    o1, o2 = _two_outputs(lib).apply(a)
    (g,) = lib.autograd.grad((o2 * o2).sum() + o1.sum(), o2, create_graph=True)
    return {"g": g}


def _unused_output(lib: ModuleType, device: str) -> dict[str, object]:
    a = lib.tensor([1.0, -0.5], device=device, requires_grad=True)
    o1, o2 = _two_outputs(lib).apply(a)
    (g,) = lib.autograd.grad(o1.sum(), o2, allow_unused=True)
    return {"g": g}


def _root_is_the_barrier_output(lib: ModuleType, device: str) -> dict[str, object]:
    class Total(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return (x * x).sum()  # type: ignore[operator]

    m = Total()
    m.register_full_backward_hook(lambda mod, gi, go: None)
    s = m(_x(lib, device))
    (g,) = lib.autograd.grad(s, s)
    return {"g": g}


_SCENARIOS: dict[str, Callable[[ModuleType, str], dict[str, object]]] = {
    "hooked-output": _hooked_output,
    "hooked-output-create-graph": _hooked_output_create_graph,
    "function-output": _function_output,
    "function-output-create-graph": _function_output_create_graph,
    "unused-output": _unused_output,
    "root-is-the-barrier-output": _root_is_the_barrier_output,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    _compare(_SCENARIOS[name](lucid, device), _SCENARIOS[name](ref, "cpu"))


# ── without the reference ──────────────────────────────────────────────────


def test_the_reported_case_reads_back(device: str) -> None:
    lin = lucid.nn.Linear(2, 2).to(device)
    lin.register_full_backward_hook(lambda *a: None)
    y = lin(lucid.ones(1, 2, requires_grad=True, device=device))
    (g,) = lucid.autograd.grad((3 * y).sum(), y)
    assert g.tolist() == [[3.0, 3.0]]


def test_a_multi_output_function_reads_back(device: str) -> None:
    a = lucid.ones(2, requires_grad=True, device=device)
    o1, o2 = _two_outputs(lucid).apply(a)
    (g,) = lucid.autograd.grad((5 * o2).sum() + o1.sum(), o2)
    assert g.tolist() == [5.0, 5.0]


def test_an_output_nothing_reached_is_refused(device: str) -> None:
    a = lucid.ones(2, requires_grad=True, device=device)
    o1, o2 = _two_outputs(lucid).apply(a)
    with pytest.raises(RuntimeError, match="allow_unused"):
        lucid.autograd.grad(o1.sum(), o2)
