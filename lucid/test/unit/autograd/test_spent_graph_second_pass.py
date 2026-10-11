"""A second pass over a spent graph is refused on every path (LCD-333).

A pass without ``retain_graph`` frees what the graph saved and leaves the
root a node with no edges.  ``backward()`` refused a second pass there, but
``grad()`` read that node as a dead end: it reported the inputs unreachable,
and under ``allow_unused`` — or from ``backward(inputs=)``, which is built on
it — answered None (or left ``.grad`` as it was) without a word.

A spent graph that does not lead to the requested inputs is no reason to
refuse: a gradient taken past a spent loss used as a constant is answered.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference, so the reference states the expected outcome.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 6) for v in t.detach().cpu().reshape(-1).tolist()]


def _outcome(m: ModuleType, device: str, scenario: Scenario) -> dict[str, object]:
    try:
        return {"value": scenario(m, device)}
    except RuntimeError as e:
        return {"second_pass": "a second time" in str(e).lower()}


def _spent(m: ModuleType, device: str) -> tuple[object, object]:
    a = m.ones(2, requires_grad=True, device=device)
    y = (a * a).sum()
    y.backward()
    return a, y


def _grad(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return [_values(g) for g in m.autograd.grad(y, a)]


def _grad_allow_unused(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return [_values(g) for g in m.autograd.grad(y, a, allow_unused=True)]


def _grad_after_grad(m: ModuleType, device: str) -> object:
    a = m.ones(2, requires_grad=True, device=device)
    y = (a * a).sum()
    m.autograd.grad(y, a)
    return [_values(g) for g in m.autograd.grad(y, a, allow_unused=True)]


def _grad_create_graph(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return [_values(g) for g in m.autograd.grad(y, a, create_graph=True)]


def _backward(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    y.backward()
    return _values(a.grad)


def _backward_inputs(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    y.backward(inputs=[a])
    return _values(a.grad)


def _backward_inputs_one_unused(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    c = m.ones(2, requires_grad=True, device=device)
    y.backward(inputs=[a, c])
    return [_values(a.grad), _values(c.grad)]


def _downstream_grad(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return [_values(g) for g in m.autograd.grad(y * 2, a, allow_unused=True)]


def _shared_leaf(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return [_values(g) for g in m.autograd.grad((a * 3).sum() + y, a)]


def _spent_twice(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    w = m.ones(2, requires_grad=True, device=device)
    z = (w * y).sum()
    m.autograd.grad(z, w)
    return [_values(g) for g in m.autograd.grad(z, a, allow_unused=True)]


def _retained(m: ModuleType, device: str) -> object:
    a = m.ones(2, requires_grad=True, device=device)
    y = (a * a).sum()
    m.autograd.grad(y, a, retain_graph=True)
    y.backward(inputs=[a], retain_graph=True)
    return [_values(g) for g in m.autograd.grad(y, a)] + [_values(a.grad)]


# A spent loss taken as a constant: no requested input lies behind it.
def _unrelated_grad(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    w = m.ones(2, requires_grad=True, device=device)
    return [_values(g) for g in m.autograd.grad((w * y).sum(), w)]


def _functional(m: ModuleType) -> ModuleType:
    return m.autograd.functional if hasattr(m.autograd, "functional") else m.autograd


def _vjp_past_spent(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    out, (g,) = _functional(m).vjp(lambda x: (x * y).sum(), (a,))
    return [_values(out), _values(g)]


def _jacobian_past_spent(m: ModuleType, device: str) -> object:
    a, y = _spent(m, device)
    return _values(_functional(m).jacobian(lambda x: x * y, a))


_SCENARIOS: dict[str, Scenario] = {
    "grad": _grad,
    "grad-allow-unused": _grad_allow_unused,
    "grad-after-grad": _grad_after_grad,
    "grad-create-graph": _grad_create_graph,
    "backward": _backward,
    "backward-inputs": _backward_inputs,
    "backward-inputs-one-unused": _backward_inputs_one_unused,
    "downstream-grad": _downstream_grad,
    "shared-leaf": _shared_leaf,
    "spent-twice": _spent_twice,
    "retained": _retained,
    "unrelated-grad": _unrelated_grad,
    "vjp-past-spent": _vjp_past_spent,
    "jacobian-past-spent": _jacobian_past_spent,
}

_REFUSED = {"second_pass": True}

# Without the reference installed, what the reference answers.
_EXPECTED: dict[str, dict[str, object]] = {
    "grad": _REFUSED,
    "grad-allow-unused": _REFUSED,
    "grad-after-grad": _REFUSED,
    "grad-create-graph": _REFUSED,
    "backward": _REFUSED,
    "backward-inputs": _REFUSED,
    "backward-inputs-one-unused": _REFUSED,
    "downstream-grad": _REFUSED,
    "shared-leaf": _REFUSED,
    "spent-twice": _REFUSED,
    "retained": {"value": [[2.0, 2.0], [2.0, 2.0]]},
    "unrelated-grad": {"value": [[2.0, 2.0]]},
    "vjp-past-spent": {"value": [[4.0], [2.0, 2.0]]},
    "jacobian-past-spent": {"value": [2.0, 0.0, 0.0, 2.0]},
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    got = _outcome(lucid, device, _SCENARIOS[name])
    assert got == _outcome(ref, "cpu", _SCENARIOS[name])


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_the_outcome(name: str, device: str) -> None:
    assert _outcome(lucid, device, _SCENARIOS[name]) == _EXPECTED[name]


def test_the_refusal_is_a_runtime_error_naming_retain_graph(device: str) -> None:
    a, y = _spent(lucid, device)
    with pytest.raises(RuntimeError, match="retain_graph=True"):
        lucid.autograd.grad(y, a, allow_unused=True)


def test_a_refused_pass_leaves_grad_alone(device: str) -> None:
    a, y = _spent(lucid, device)
    before = _values(a.grad)
    with pytest.raises(RuntimeError, match="second time"):
        y.backward(inputs=[a])
    assert _values(a.grad) == before


def _nothing_saved_grad(m: ModuleType, device: str) -> object:
    a = m.ones(2, requires_grad=True, device=device)
    y = (a + 1).sum()
    m.autograd.grad(y, a)
    return [_values(g) for g in m.autograd.grad(y, a)]


def _nothing_saved_backward(m: ModuleType, device: str) -> object:
    a = m.ones(2, requires_grad=True, device=device)
    y = (a + 1).sum()
    y.backward()
    y.backward()
    return _values(a.grad)


@pytest.mark.xfail(
    strict=True,
    reason="LCD-350: a spent root refuses whatever its graph saved; a graph "
    "that saves nothing should run again, as the reference allows",
)
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [(_nothing_saved_grad, [[1.0, 1.0]]), (_nothing_saved_backward, [2.0, 2.0])],
    ids=["grad", "backward"],
)
def test_a_graph_that_saves_nothing_runs_again(
    scenario: Scenario, expected: object, device: str
) -> None:
    assert scenario(lucid, device) == expected
