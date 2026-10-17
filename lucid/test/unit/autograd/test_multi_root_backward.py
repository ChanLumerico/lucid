"""``backward`` and ``grad`` from several roots run one pass (LCD-170).

``autograd.backward([a, b])`` ran one engine pass per root.  Roots sharing a
graph needed it kept for every root but the last, and a tensor they share had
its hooks run once per root, each time on that root's part of its gradient —
and ``autograd.grad`` with several outputs did the same.  The reference seeds
every root in one traversal: a shared tensor's hooks see its whole gradient,
once.

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
    return [round(float(v), 5) for v in t.detach().cpu().reshape(-1).tolist()]


def _outcome(m: ModuleType, device: str, scenario: Scenario) -> object:
    try:
        return {"value": scenario(m, device)}
    except RuntimeError as e:
        return {"second_pass": "second time" in str(e).lower()}


def _shared_graph(m: ModuleType, device: str) -> object:
    x = m.ones(2, requires_grad=True, device=device)
    y = x * 2
    m.autograd.backward([(y * 3).sum(), (y * 4).sum()])
    return _values(x.grad)


def _shared_tensor_hook(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    h = x * 2.0
    seen: list[object] = []
    h.register_hook(lambda g: seen.append(_values(g)))
    m.autograd.backward([(h * 3.0).sum(), (h * 5.0).sum()])
    return [seen, _values(x.grad)]


def _leaf_hook(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    seen: list[object] = []
    x.register_hook(lambda g: seen.append(_values(g)))
    m.autograd.backward([(x * 3.0).sum(), (x * x).sum()])
    return [seen, _values(x.grad)]


def _same_root_twice(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = x * 3.0
    seen: list[object] = []
    y.register_hook(lambda g: seen.append(_values(g)))
    g1 = m.tensor([2.0, 4.0], device=device)
    g2 = m.tensor([5.0, 1.0], device=device)
    m.autograd.backward([y, y], [g1, g2])
    return [seen, _values(x.grad), _values(g1), _values(g2)]


def _seed_is_not_written(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    g = m.tensor([2.0, 4.0], device=device)
    m.autograd.backward([x, x], [g, g])
    x.grad.mul_(10)
    return [_values(x.grad), _values(g)]


def _leaf_root_and_graph_root(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    m.autograd.backward([x, (x * 2).sum()], [m.ones(2, device=device), None])
    return _values(x.grad)


def _bare_leaf_roots(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    seen: list[object] = []
    x.register_hook(lambda g: seen.append(_values(g)))
    m.autograd.backward([x, x], [m.ones(2, device=device), m.ones(2, device=device) * 2])
    return [seen, _values(x.grad)]


def _module_hook(m: ModuleType, device: str) -> object:
    class Scale(m.nn.Module):
        def forward(self, a: object) -> object:
            return a * 2.0

    mod = Scale()
    seen: list[object] = []
    mod.register_full_backward_hook(lambda _m, gi, go: seen.append(_values(go[0])))
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    out = mod(x)
    m.autograd.backward([(out * 3.0).sum(), (out * 4.0).sum()])
    return [seen, _values(x.grad)]


def _second_pass_refused(m: ModuleType, device: str) -> object:
    x = m.ones(2, requires_grad=True, device=device)
    y = x * x
    a, b = (y * 3).sum(), (y * 4).sum()
    m.autograd.backward([a, b])
    a.backward()
    return _values(x.grad)


def _retained(m: ModuleType, device: str) -> object:
    x = m.ones(2, requires_grad=True, device=device)
    y = x * x
    a, b = (y * 3).sum(), (y * 4).sum()
    m.autograd.backward([a, b], retain_graph=True)
    m.autograd.backward([a, b])
    return _values(x.grad)


def _create_graph(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = x * x
    m.autograd.backward([(y * 3).sum(), (y * x).sum()], create_graph=True)
    (gg,) = m.autograd.grad(x.grad.sum(), x)
    return [_values(x.grad), _values(gg)]


def _complex_seed(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    z = m.complex(x, x * 3.0)
    s = m.complex(m.tensor([1.0, 0.5], device=device), m.tensor([2.0, -1.0], device=device))
    m.autograd.backward([z * 2.0, z], [s, s])
    return _values(x.grad)


def _grad_shared_tensor_hook(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    h = x * 2.0
    seen: list[object] = []
    h.register_hook(lambda g: seen.append(_values(g)))
    gs = m.autograd.grad([(h * 3.0).sum(), (h * 5.0).sum()], x)
    return [seen, [_values(g) for g in gs]]


def _grad_seeds(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = x * x
    gs = m.autograd.grad(
        [y, y * 2],
        [x],
        grad_outputs=[m.tensor([1.0, 3.0], device=device), m.tensor([2.0, 0.5], device=device)],
    )
    return [_values(g) for g in gs]


def _grad_frees_once(m: ModuleType, device: str) -> object:
    x = m.ones(2, requires_grad=True, device=device)
    y = x * x
    gs = m.autograd.grad([(y * 3).sum(), (y * 4).sum()], x)
    return [_values(g) for g in gs] + [_outcome(m, device, lambda mm, d: y.sum().backward())]


def _grad_create_graph_seed_graph(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    w = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    y = x * x
    (g,) = m.autograd.grad([y, y], [x], grad_outputs=[w, w * 2], create_graph=True)
    (gw,) = m.autograd.grad(g.sum(), w)
    return [_values(g), _values(gw)]


def _grad_orphan_root(m: ModuleType, device: str) -> object:
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    gs = m.autograd.grad([x, (x * 4).sum()], x, grad_outputs=[m.ones(2, device=device) * 3, None])
    return [_values(g) for g in gs]


_SCENARIOS: dict[str, Scenario] = {
    "shared-graph": _shared_graph,
    "shared-tensor-hook": _shared_tensor_hook,
    "leaf-hook": _leaf_hook,
    "same-root-twice": _same_root_twice,
    "seed-is-not-written": _seed_is_not_written,
    "leaf-root-and-graph-root": _leaf_root_and_graph_root,
    "bare-leaf-roots": _bare_leaf_roots,
    "module-hook": _module_hook,
    "second-pass-refused": _second_pass_refused,
    "retained": _retained,
    "create-graph": _create_graph,
    "complex-seed": _complex_seed,
    "grad-shared-tensor-hook": _grad_shared_tensor_hook,
    "grad-seeds": _grad_seeds,
    "grad-frees-once": _grad_frees_once,
    "grad-create-graph-seed-graph": _grad_create_graph_seed_graph,
    "grad-orphan-root": _grad_orphan_root,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    want = _outcome(ref, "cpu", _SCENARIOS[name])
    assert _outcome(lucid, device, _SCENARIOS[name]) == want


def test_a_shared_tensor_hook_runs_once_on_the_whole_gradient(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    h = x * 2.0
    seen: list[object] = []
    h.register_hook(lambda g: seen.append(_values(g)))
    lucid.autograd.backward([(h * 3.0).sum(), (h * 5.0).sum()])
    assert seen == [[8.0, 8.0, 8.0]]
    assert _values(x.grad) == [16.0, 16.0, 16.0]


def test_the_root_and_seed_counts_must_match() -> None:
    from lucid._C import engine as _C_engine

    x = lucid.ones(2, requires_grad=True)
    with pytest.raises(RuntimeError, match="2 roots and 1 seeds"):
        _C_engine.engine_backward([x.impl, x.impl], [None])
