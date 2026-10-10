"""A module's gradient hook must not keep the module alive (LCD-302).

A tensor hook is held by the engine — the tensor's autograd slot — where
the garbage collector cannot see it.  ``Embedding(scale_grad_by_freq=True)``
registered its own bound method on its own weight: module -> weight ->
engine slot -> bound method -> module, a cycle nothing could free, so every
such module leaked.  The hook also stayed on the old weight when ``.to()``,
``.double()`` or ``copy.deepcopy`` replaced it, and the scaling was lost.

The scaling now lives on each lookup's graph node
(``F.embedding(..., scale_grad_by_freq=True)``), which owns nothing a module
holds.  Pinned here: every site in :data:`_SITES` dies once dropped, the
scaled gradient matches the reference across the cases that used to break
it, and a static scan keeps library code from handing the engine a
callable that closes over ``self`` — a tensor hook, or a module call's
backward-hook runners, whose holder once also held the engine state that
kept them (LCD-167, ``test_module_backward_hook_lifetime.py``).
"""

import ast
import copy
import gc
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn

# ── leak: every site dies once dropped ─────────────────────────────────────


def _embedding_scale_grad_by_freq(device: str) -> nn.Module:
    return nn.Embedding(5, 3, scale_grad_by_freq=True, device=device)


def _embedding_step(module: nn.Module, device: str) -> None:
    idx = lucid.tensor([1, 1, 1, 2, 4, 4], dtype=lucid.int64, device=device)
    module(idx).sum().backward()


#: name -> (build, step that runs the scaled backward)
_SITES: dict[
    str, tuple[Callable[[str], nn.Module], Callable[[nn.Module, str], None]]
] = {
    "embedding-scale-grad-by-freq": (_embedding_scale_grad_by_freq, _embedding_step),
}


def _assert_freed(ref: weakref.ref[nn.Module]) -> None:
    gc.collect()
    assert ref() is None, "the module outlived every reference to it"


@pytest.mark.parametrize("name", list(_SITES))
def test_module_is_freed_without_a_backward(name: str, device: str) -> None:
    build, _ = _SITES[name]
    module = build(device)
    ref = weakref.ref(module)
    del module
    _assert_freed(ref)


@pytest.mark.parametrize("name", list(_SITES))
def test_module_is_freed_after_a_backward(name: str, device: str) -> None:
    build, step = _SITES[name]
    module = build(device)
    step(module, device)
    ref = weakref.ref(module)
    del module
    _assert_freed(ref)


# ── Embedding(scale_grad_by_freq) against the reference ────────────────────

_TABLE = [[0.1 * (r + 1) + 0.01 * c for c in range(3)] for r in range(5)]
_IDX = [[1, 1, 1, 2], [4, 4, 0, 1]]
_OTHER = [3, 3, 1]


def _embedding(lib: ModuleType, device: str, **kwargs: object) -> object:
    """Built on the CPU and then moved, so Metal also covers ``.to()``."""
    emb = lib.nn.Embedding(5, 3, scale_grad_by_freq=True, **kwargs)
    with lib.no_grad():
        emb.weight.copy_(lib.tensor(_TABLE))
    return emb.to(device)


def _lookup(lib: ModuleType, emb: object, idx: list[object], device: str) -> object:
    x = lib.tensor(idx, dtype=lib.int64, device=device)
    out = emb(x)  # type: ignore[operator]
    upstream = lib.arange(out.numel(), dtype=out.dtype, device=device)
    return (out * upstream.reshape(out.shape)).sum()


def _grad(emb: object) -> list[list[float]]:
    return emb.weight.grad.detach().cpu().tolist()  # type: ignore[attr-defined, no-any-return]


def _one_lookup(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device)
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _two_lookups_one_backward(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device)
    total = _lookup(lib, emb, _IDX, device) + _lookup(lib, emb, _OTHER, device)  # type: ignore[operator]
    total.backward()
    return _grad(emb)


def _accumulated_micro_batches(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device)
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    _lookup(lib, emb, _OTHER, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _deepcopied(lib: ModuleType, device: str) -> list[list[float]]:
    emb = copy.deepcopy(_embedding(lib, device))
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _doubled(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device).double()  # type: ignore[attr-defined]
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _retained_graph_twice(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device)
    loss = _lookup(lib, emb, _IDX, device)
    loss.backward(retain_graph=True)  # type: ignore[attr-defined]
    loss.backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _with_padding_idx(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device, padding_idx=4)
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


def _with_max_norm(lib: ModuleType, device: str) -> list[list[float]]:
    emb = _embedding(lib, device, max_norm=0.2)
    _lookup(lib, emb, _IDX, device).backward()  # type: ignore[attr-defined]
    return _grad(emb)


_SCENARIOS: dict[str, Callable[[ModuleType, str], list[list[float]]]] = {
    "one-lookup": _one_lookup,
    "two-lookups-one-backward": _two_lookups_one_backward,
    "accumulated-micro-batches": _accumulated_micro_batches,
    "deepcopy": _deepcopied,
    "double": _doubled,
    "retain-graph-twice": _retained_graph_twice,
    "padding-idx": _with_padding_idx,
    "max-norm": _with_max_norm,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_scaled_gradient_matches_the_reference(
    name: str, ref: ModuleType, device: str
) -> None:
    if name == "double" and device == "metal":
        pytest.skip("Metal holds no float64")
    got = _SCENARIOS[name](lucid, device)
    want = _SCENARIOS[name](ref, "cpu")
    for got_row, want_row in zip(got, want, strict=True):
        for g, w in zip(got_row, want_row, strict=True):
            assert abs(g - w) <= 1e-5 * max(1.0, abs(w)), (got, want)


def test_the_result_reads_the_unscaled_gradient(device: str) -> None:
    """The scaling sits behind the result: a caller's ``retain_grad`` on it
    sees the gradient that arrived, as the reference's does."""
    weight = lucid.ones(5, 3, device=device, requires_grad=True)
    idx = lucid.tensor([1, 1, 2], dtype=lucid.int64, device=device)
    out = lucid.nn.functional.embedding(idx, weight, scale_grad_by_freq=True)
    out.retain_grad()
    out.sum().backward()
    assert out.grad is not None and weight.grad is not None
    assert out.grad.tolist() == [[1.0] * 3] * 3
    assert weight.grad.tolist()[1] == [1.0] * 3


# ── class guard: no engine-held library callable closes over ``self`` ─────


_LIBRARY = Path(lucid.__file__).parent

#: engine entry points that keep a Python callable -> index of the first
#: positional argument that is one (every later one, and every keyword, is)
_REGISTRARS: dict[str, int] = {
    "register_hook": -1,
    "_register_tensor_hook": -1,  # ``_register_tensor_hook(tensor, hook)``
    # ``(n_inputs, pre_runner, full_runner)`` — a module call's backward
    # hooks; held by the call's graph (LCD-167)
    "_create_module_backward_hook_state": 1,
}


def _hooks_holding_self(source: str) -> list[int]:
    """Lines handing the engine a callable whose expression mentions ``self``.

    A bound method (``self.m``), a lambda or a partial over ``self`` all
    hand the engine a strong reference to the object that registers it.
    The engine's edges are invisible to the garbage collector, so if that
    object also holds what the engine keeps — a weight's slot, a module
    call's hook state — the pair can never be freed.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _REGISTRARS or not node.args:
            continue
        hooks = [*node.args[_REGISTRARS[name] :], *(kw.value for kw in node.keywords)]
        if any(
            isinstance(n, ast.Name) and n.id == "self"
            for hook in hooks
            for n in ast.walk(hook)
        ):
            lines.append(node.lineno)
    return lines


def test_the_scan_catches_the_pattern_it_guards() -> None:
    source = (
        "class M:\n"
        "    def __init__(self, w):\n"
        "        w.register_hook(self.scale)\n"
        "        w.register_hook(lambda g: self.scale(g))\n"
        "        _register_tensor_hook(w, self.scale)\n"
        "        w.register_hook(module_level_fn)\n"
        "        _register_tensor_hook(self.w, module_level_fn)\n"
        "        self.state = E._create_module_backward_hook_state(\n"
        "            n, self.run_pre_hooks, self.run_full_hooks)\n"
        "        E._create_module_backward_hook_state(\n"
        "            len(self.args), runners.pre, full_runner=self.full)\n"
        "        E._create_module_backward_hook_state(len(self.args), a, b)\n"
    )
    assert _hooks_holding_self(source) == [3, 4, 5, 8, 10]


def test_no_library_tensor_hook_holds_its_owner() -> None:
    offenders = [
        f"{path.relative_to(_LIBRARY)}:{line}"
        for path in sorted(_LIBRARY.rglob("*.py"))
        if path.relative_to(_LIBRARY).parts[0] != "test"
        for line in _hooks_holding_self(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
