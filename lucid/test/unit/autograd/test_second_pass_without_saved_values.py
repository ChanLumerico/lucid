"""A second pass over a graph is refused only when the first freed what it reads (LCD-350).

A pass without ``retain_graph`` frees what every node it ran saved for
backward, and a later pass must not read that memory.  Whether a node's
release freed anything was read off the capacity of its version list, which
every kernel filled — including ``add``, ``sum`` and the views, which save
nothing — so ``(a + 1).sum()`` refused a second ``backward()`` the reference
runs.  Each node now records what its release freed: values its eager
backward reads (refused by every pass), or only the tensors graph-mode
backward reads (refused by a ``create_graph`` pass).  The root is cut from
its graph only when something an eager pass reads was freed.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Build = Callable[[ModuleType, object], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 5) for v in t.detach().cpu().reshape(-1).tolist()]


def _outcome(m: ModuleType, device: str, scenario: Callable[[ModuleType, str], object]) -> object:
    try:
        return {"value": scenario(m, device)}
    except RuntimeError as e:
        return {"second_pass": "second time" in str(e).lower()}


# Graphs that save no value their backward reads.
_SAVE_NOTHING: dict[str, Build] = {
    "add-scalar": lambda m, x: (x + 1.0).sum(),
    "sub-scalar": lambda m, x: (x - 2.0).mean(),
    "neg": lambda m, x: (-x).sum(),
    "add-tensors": lambda m, x: (x + x).sum(),
    "reshape-transpose": lambda m, x: x.reshape(3, 2).transpose(0, 1).sum(),
    "permute": lambda m, x: x.reshape(3, 2).permute(1, 0).sum(),
    "expand": lambda m, x: x.reshape(1, 6).expand(2, 6).sum(),
    "slice": lambda m, x: x[1:4].sum(),
    "cat": lambda m, x: m.cat([x, x - 1.0], 0).mean(),
    "stack": lambda m, x: m.stack([x, x + 1.0], 0).sum(),
    "flip": lambda m, x: m.flip(x, [0]).sum(),
    "pad": lambda m, x: m.nn.functional.pad(x, (1, 2)).sum(),
}

# Graphs with a node whose backward reads a value it saved.
_SAVE_VALUES: dict[str, Build] = {
    "mul": lambda m, x: (x * x).sum(),
    "exp": lambda m, x: x.exp().sum(),
    "sin-after-add": lambda m, x: (x + 1.0).sin().sum(),
    "matmul": lambda m, x: (x.reshape(2, 3) @ x.reshape(3, 2)).sum(),
}


def _leaf(m: ModuleType, device: str) -> object:
    return m.arange(6, dtype=m.float32, device=device).requires_grad_(True)


def _backward_twice(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        y.backward()
        y.backward()
        return _values(x.grad)

    return scenario


def _grad_twice(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        m.autograd.grad(y, x)
        return [_values(g) for g in m.autograd.grad(y, x)]

    return scenario


def _grad_then_backward(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        m.autograd.grad(y, x)
        y.backward()
        return _values(x.grad)

    return scenario


_PASSES = {
    "backward-twice": _backward_twice,
    "grad-twice": _grad_twice,
    "grad-then-backward": _grad_then_backward,
}


@pytest.mark.parametrize("graph", list(_SAVE_NOTHING))
@pytest.mark.parametrize("passes", list(_PASSES))
def test_a_graph_that_saves_no_values_runs_again(
    graph: str, passes: str, ref: ModuleType, device: str
) -> None:
    scenario = _PASSES[passes](_SAVE_NOTHING[graph])
    want = _outcome(ref, "cpu", scenario)
    assert "value" in want
    assert _outcome(lucid, device, scenario) == want


@pytest.mark.parametrize("graph", list(_SAVE_VALUES))
@pytest.mark.parametrize("passes", list(_PASSES))
def test_a_graph_that_saves_values_is_still_refused(
    graph: str, passes: str, ref: ModuleType, device: str
) -> None:
    scenario = _PASSES[passes](_SAVE_VALUES[graph])
    assert _outcome(ref, "cpu", scenario) == {"second_pass": True}
    assert _outcome(lucid, device, scenario) == {"second_pass": True}


@pytest.mark.parametrize("graph", list(_SAVE_NOTHING))
def test_the_same_graph_without_a_second_pass_is_unchanged(graph: str, device: str) -> None:
    x = _leaf(lucid, device)
    _SAVE_NOTHING[graph](lucid, x).backward()
    once = _values(x.grad)
    x.grad = None
    y = _SAVE_NOTHING[graph](lucid, x)
    y.backward(retain_graph=True)
    assert _values(x.grad) == once


@pytest.mark.parametrize(
    "build",
    [
        _SAVE_NOTHING["add-scalar"],
        _SAVE_NOTHING["cat"],
        lambda m, x: m.nn.functional.linear(x.reshape(2, 3), m.ones(4, 3, device=x.device)).sum(),
        lambda m, x: m.nn.functional.mse_loss(x, m.zeros(6, device=x.device)),
    ],
    ids=["add-scalar", "cat-of-add", "linear", "mse-loss"],
)
def test_create_graph_after_a_release_refuses_rather_than_reads_freed_inputs(
    build: Build, device: str
) -> None:
    # Graph-mode backward reads the tensors each node was handed, and a
    # release frees them all, so a create_graph pass refuses any released
    # node — where it read null, or the reference answers from the values
    # it never freed (add) — rather than crash.
    x = _leaf(lucid, device)
    y = build(lucid, x)
    y.backward()
    with pytest.raises(RuntimeError, match="second time"):
        lucid.autograd.grad(y, x, create_graph=True)


def test_a_save_free_pass_keeps_the_root_attached(device: str) -> None:
    x = _leaf(lucid, device)
    y = (x + 1.0).sum()
    fn = y.grad_fn
    y.backward()
    assert y.grad_fn is not None and y.grad_fn.name() == fn.name()


def test_a_pass_that_freed_values_cuts_the_root_from_its_graph(device: str) -> None:
    x = _leaf(lucid, device)
    y = (x * x).sum()
    y.backward()
    assert y.grad_fn is not None and y.grad_fn.name() == "SpentGraph"


@pytest.mark.xfail(
    strict=True,
    reason="LCD-350 follow-up: the spent root of (a*a).sum() hides that nothing "
    "between it and the interior tensor h was freed, so grad(y, h) is refused "
    "where the reference answers [1, 1]",
)
def test_grad_to_an_interior_tensor_behind_save_free_nodes(device: str) -> None:
    a = lucid.ones(2, requires_grad=True, device=device)
    h = a * a
    y = h.sum()
    y.backward()
    assert [_values(g) for g in lucid.autograd.grad(y, h)] == [[1.0, 1.0]]
