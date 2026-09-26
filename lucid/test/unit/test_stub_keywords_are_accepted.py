"""Every keyword a stub offers, the runtime accepts.

``Tensor`` methods built from the op registry take ``*args, **kwargs`` and
forward them to an engine function or adapter, so ``inspect`` cannot
compare them with ``tensor.pyi``.  The stub and the callee drifted apart
unnoticed: ``x.flatten(start=1)``, ``x.softmax(axis=-1)``,
``x.unbind(axis=0)``, ``x.swapaxes(axis0=0, axis1=1)``,
``x.topk(k, largest=False)``, ``x.sort(descending=True)`` and
``x.sum(correction=1)`` were all in the stub and all raised
``TypeError``.  The registry's own ``extra_kwargs`` listed ``axis`` /
``axes`` / ``keepdims`` aliases that no callee took.

Both lists are held to the callee's real parameter names here.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Callable

import pytest

import lucid
from lucid._ops._registry import _REGISTRY
from lucid._tensor.tensor import Tensor

STUB = Path(lucid.__file__).parent / "_tensor" / "tensor.pyi"


def _parameters(fn: Callable[..., object]) -> list[str] | None:
    """The callee's parameter names, or None when it takes ``**kwargs``."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except TypeError, ValueError:
        # A pybind11 function: the first docstring line is its signature.
        first = (fn.__doc__ or "").split("\n")[0]
        m = re.match(r"\w+\((.*)\)\s*->", first)
        return re.findall(r"(\w+):", m.group(1)) if m else None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return [p.name for p in params]


@pytest.mark.parametrize(
    "entry", [e for e in _REGISTRY if e.extra_kwargs], ids=lambda e: e.name
)
def test_registry_keywords_name_real_parameters(entry) -> None:  # type: ignore[no-untyped-def]
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    assert [k for k in entry.extra_kwargs if k not in accepted] == []


def _stub_methods() -> list[tuple[str, list[str]]]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tensor"
    )
    out = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("__"):
            continue
        if fn.args.vararg is not None:
            continue
        names = [a.arg for a in fn.args.args[1:]] + [a.arg for a in fn.args.kwonlyargs]
        if names:
            out.append((fn.name, names))
    return out


_BY_METHOD = {e.method_name: e for e in _REGISTRY if e.method_name}


@pytest.mark.parametrize("name,keywords", _stub_methods(), ids=lambda v: str(v))
def test_tensor_stub_keywords_reach_the_callee(name: str, keywords: list[str]) -> None:
    method = getattr(Tensor, name, None)
    assert (
        method is not None
    ), f"tensor.pyi declares Tensor.{name}, which does not exist"
    direct = _parameters(method)
    if direct is not None and "args" not in direct:
        assert [k for k in keywords if k not in direct] == []
        return
    entry = _BY_METHOD.get(name)
    if entry is None:
        pytest.skip("forwards **kwargs to something other than a registry callee")
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    # ``self`` fills the first tensor argument; the rest are passed on.
    offered = keywords[max(entry.n_tensor_args - 1, 0) :]
    assert [k for k in offered if k not in accepted] == []
