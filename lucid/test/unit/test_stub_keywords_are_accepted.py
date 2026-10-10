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

Both lists are held to the callee's real parameter names here — the
callee's own, not the engine function an adapter wraps: following
``__wrapped__`` to the engine's ``(a, b)`` hid that ``x.add(other=2)`` and
``lucid.add(input=x, other=y)`` raised ``TypeError`` (LCD-341).  The
leading tensor operands are the wrapper's to bind: their advertised names
(``input`` / ``other``) are put back in position before the callee runs.
"""

import ast
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

import lucid
import lucid._ops as ops
from lucid._ops._adapters import _make_arith_adapter
from lucid._ops._registry import _REGISTRY, OpEntry
from lucid._tensor.tensor import Tensor

STUB = Path(lucid.__file__).parent / "_tensor" / "tensor.pyi"


def _parameters(fn: Callable[..., object]) -> list[str] | None:
    """The callee's parameter names, or None when it takes ``**kwargs``."""
    try:
        params = list(inspect.signature(fn, follow_wrapped=False).parameters.values())
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


#: Stub keywords no callee takes yet — strict, so a fix flips them to XPASS.
_KNOWN_GAPS = {
    "bmm": "LCD-357: tensor.pyi names bmm's operand mat2, the registry other",
}


def _stub_cases() -> list[object]:
    return [
        pytest.param(
            name,
            keywords,
            id=str((name, keywords)),
            marks=(
                [pytest.mark.xfail(strict=True, reason=_KNOWN_GAPS[name])]
                if name in _KNOWN_GAPS
                else []
            ),
        )
        for name, keywords in _stub_methods()
    ]


@pytest.mark.parametrize("name,keywords", _stub_cases())
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
    # ``self`` fills the first parameter; the method binds the other
    # advertised tensor operands itself, and passes the rest on.
    bound = ops._leading_tensor_names(entry)[1:]
    assert [k for k in keywords if k not in accepted[1:] and k not in bound] == []


_FREE = [e for e in _REGISTRY if e.free_fn_name and e.n_tensor_args != -1]


@pytest.mark.parametrize("entry", _FREE, ids=lambda e: e.free_fn_name)
def test_free_function_keywords_reach_the_callee(entry: OpEntry) -> None:
    # The registry's own wrapper: ``lucid.softmax`` and a few others are
    # other functions of the same name, checked by their own signatures.
    advertised = list(inspect.signature(getattr(ops, entry.free_fn_name)).parameters)
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    bound = ops._leading_tensor_names(entry)
    assert [k for k in advertised if k not in accepted and k not in bound] == []


def _pair(device: str) -> tuple[Tensor, Tensor]:
    return (
        lucid.tensor([0.5, 2.0, -3.0], device=device),
        lucid.tensor([1.5, -1.0, 2.0], device=device),
    )


_ARITH = [
    e
    for e in _REGISTRY
    if e.engine_fn.__qualname__.startswith(_make_arith_adapter.__qualname__)
]


@pytest.mark.parametrize("entry", _ARITH, ids=lambda e: e.name)
def test_arithmetic_ops_take_input_and_other_by_keyword(
    entry: OpEntry, device: str
) -> None:
    def fresh() -> tuple[Tensor, Tensor]:
        a, b = _pair(device)
        if entry.name.startswith(("bitwise", "logical")):
            return a.to(lucid.int32), b.to(lucid.int32)
        return a, b

    a, b = fresh()
    want = (
        getattr(a.clone(), entry.method_name)(b)
        if entry.inplace
        else getattr(lucid, entry.free_fn_name or entry.name)(a, b)
    )
    if entry.free_fn_name:
        got = getattr(lucid, entry.free_fn_name)(input=a, other=b)
        assert got.tolist() == want.tolist()
    if entry.method_name:
        got = getattr(a.clone(), entry.method_name)(other=b)
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize(
    "call",
    [
        lambda x, y: (lucid.exp(input=x), lucid.exp(x)),
        lambda x, y: (lucid.matmul(input=x, other=y), lucid.matmul(x, y)),
        lambda x, y: (lucid.cumsum(input=x, dim=0), lucid.cumsum(x, 0)),
        lambda x, y: (
            lucid.take(input=x, indices=lucid.tensor([2, 0], device=x.device)),
            lucid.take(x, lucid.tensor([2, 0], device=x.device)),
        ),
        lambda x, y: (lucid.histc(input=x, bins=4), lucid.histc(x, 4)),
        lambda x, y: (lucid.complex(real=x, imag=y), lucid.complex(x, y)),
        lambda x, y: (x.matmul(other=y), x.matmul(y)),
        lambda x, y: (x.isclose(other=y), x.isclose(y)),
        lambda x, y: (x.add(other=2), x.add(2)),
    ],
    ids=[
        "exp",
        "matmul",
        "cumsum",
        "take",
        "histc",
        "complex",
        "Tensor.matmul",
        "Tensor.isclose",
        "Tensor.add",
    ],
)
def test_advertised_tensor_keywords_bind(
    call: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]], device: str
) -> None:
    x, y = _pair(device)
    got, want = call(x, y)
    assert got.tolist() == want.tolist()


def test_a_tensor_operand_given_twice_or_skipped_is_refused() -> None:
    x, y = _pair("cpu")
    with pytest.raises(TypeError, match="multiple values for argument 'input'"):
        lucid.add(x, input=y)
    with pytest.raises(TypeError, match="missing required argument 'input'"):
        lucid.add(other=y)


def test_atan2_reads_input_over_other(ref: ModuleType, device: str) -> None:
    # The engine names its operands (y, x), which the signature used to
    # advertise as (other, input): the order was the reference's, the names
    # were swapped, and a keyword call could not bind at all.
    x, y = _pair(device)
    assert list(inspect.signature(lucid.atan2).parameters) == ["input", "other"]
    want = ref.atan2(ref.tensor(x.tolist()), ref.tensor(y.tolist())).tolist()
    for got in (lucid.atan2(x, y), lucid.atan2(input=x, other=y), x.atan2(other=y)):
        assert got.tolist() == pytest.approx(want, rel=1e-6)
