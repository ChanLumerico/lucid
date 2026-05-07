"""
Free-function ops exposed as ``lucid.xxx``.

All functions are generated from the registry.  Each generated wrapper carries
an explicit ``__signature__`` derived from the underlying adapter (when one
exists) or parsed from the pybind11 ``__doc__`` of the engine function.  The
result is that ``inspect.signature(lucid.cumsum)`` returns
``(input: Tensor, dim: int = -1)`` rather than the generic
``(*args, **kwargs)`` — so IDE autocomplete, ``help(lucid.foo)``, IPython
introspection, and ``inspect.Signature.bind()`` all see the real parameter
names.

The wrapper *body* still uses ``*args / **kwargs`` so that the unwrap/wrap
boundary remains a single forwarding point, but the visible surface is fully
typed.

For static type-checking, the same per-op signatures are emitted into
``lucid/__init__.pyi`` by ``tools/gen_pyi.py``; both paths are driven by
``_signature_for_entry`` below to keep them in sync.
"""

import annotationlib
import builtins
import inspect
import re
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import OpEntry, _REGISTRY

# PEP 749 / Python 3.14: ``Format.FORWARDREF`` keeps unresolved names as
# ``ForwardRef`` objects instead of raising ``NameError``.  Used when
# introspecting adapter signatures whose annotations reference symbols
# that live behind ``TYPE_CHECKING`` (e.g. ``Tensor`` in ``_adapters``).
_FORWARDREF = annotationlib.Format.FORWARDREF

# Builtins we need are saved up front because we register ops named ``min`` /
# ``max`` / ``sum`` / ``any`` / ``all`` into module ``globals()`` before this
# file finishes executing — bare references to those names below would resolve
# to the wrappers (with TensorImpl-only signatures) instead of the builtins.
_b_min = builtins.min
_b_max = builtins.max

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Signature extraction ─────────────────────────────────────────────────────


# Matches the first line of a pybind11 docstring:
#   "name(param: Type [= default], ...) -> ReturnType"
# We don't need to parse types — only param names + defaults — because the
# wrapper forwards everything verbatim.
_PYBIND_SIG_RE = re.compile(r"^[a-zA-Z_][\w]*\((?P<params>.*)\)\s*->", re.DOTALL)


def _parse_pybind_signature(fn: object) -> inspect.Signature | None:
    """Extract an ``inspect.Signature`` from a pybind11 builtin's docstring.

    Returns ``None`` if the docstring is missing or unparseable.  pybind11
    doesn't expose ``__signature__`` on its builtins, but it does emit a
    well-formed signature on the first line of ``__doc__``.
    """
    doc = getattr(fn, "__doc__", None)
    if not doc:
        return None
    first_line = doc.splitlines()[0]
    m = _PYBIND_SIG_RE.match(first_line)
    if not m:
        return None
    raw = m.group("params").strip()
    if not raw:
        return inspect.Signature(parameters=[])

    # Split top-level commas only — types may contain nested brackets.
    params: list[inspect.Parameter] = []
    depth = 0
    cur = ""
    for ch in raw + ",":
        if ch == "," and depth == 0:
            piece = cur.strip()
            cur = ""
            if not piece:
                continue
            # piece looks like "name: Type" or "name: Type = default"
            if "=" in piece:
                head, default_str = piece.split("=", 1)
                default_str = default_str.strip()
                # Try to evaluate simple defaults (numbers, [], (), etc.).
                try:
                    default = eval(default_str, {"__builtins__": {}}, {})
                except Exception:
                    default = inspect.Parameter.empty
            else:
                head = piece
                default = inspect.Parameter.empty
            pname = head.split(":", 1)[0].strip()
            if not pname.isidentifier():
                return None
            params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                )
            )
        else:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            cur += ch
    return inspect.Signature(parameters=params)


# Internal identifiers (in adapters) and single-letter pybind names that we
# rewrite to user-facing names.  Domain-specific names (``index``, ``mask``,
# ``cond``, etc.) are kept as-is.
_GENERIC_PARAM_REWRITE = {
    "a": "input",
    "x": "input",
    "x_impl": "input",
    "a_impl": "input",
    "self": "input",
    "b": "other",
    "y": "other",
    "y_impl": "other",
    "b_impl": "other",
}


def _rename_leading_tensor_params(
    sig: inspect.Signature, n_tensor_args: int
) -> inspect.Signature:
    """Rewrite the first ``n_tensor_args`` parameters from their internal /
    single-letter names (``a``, ``b``, ``x_impl``) to their user-facing
    counterparts (``input``, ``other``).  Domain-specific names like
    ``indices`` or ``mask`` are left alone — they already convey role."""
    if n_tensor_args <= 0:
        return sig
    params = list(sig.parameters.values())
    upto = _b_min(n_tensor_args, len(params))
    for i in range(upto):
        new_name = _GENERIC_PARAM_REWRITE.get(params[i].name)
        if new_name is not None:
            params[i] = params[i].replace(name=new_name)
    return sig.replace(parameters=params)


def _signature_for_entry(entry: OpEntry) -> inspect.Signature:
    """Best-effort signature for a registry entry.

    Strategy:
      1. ``n_tensor_args == -1`` (list-style ops): synthesise a signature
         starting with ``tensors: list[Tensor]`` and parse the rest from the
         engine_fn (whether adapter or pybind builtin).
      2. Otherwise: introspect the engine_fn directly — adapters have real
         Python signatures, pybind builtins are parsed from ``__doc__``.

    Falls back to ``(*args, **kwargs)`` when extraction fails so the wrapper
    still works (``help()`` will be uninformative for that op).
    """
    fn = entry.engine_fn
    # Python adapters are introspectable directly.  We pass
    # ``annotation_format=Format.FORWARDREF`` (PEP 749, Python 3.14+) so any
    # name held under ``TYPE_CHECKING`` (e.g. ``Tensor`` in ``_adapters``) is
    # surfaced as a ``ForwardRef`` instead of raising ``NameError`` during
    # registry population.
    try:
        sig = inspect.signature(fn, annotation_format=_FORWARDREF)
    except (TypeError, ValueError, NameError):
        sig = _parse_pybind_signature(fn) or inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "args",
                    inspect.Parameter.VAR_POSITIONAL,
                ),
                inspect.Parameter(
                    "kwargs",
                    inspect.Parameter.VAR_KEYWORD,
                ),
            ]
        )

    if entry.n_tensor_args == -1:
        # Replace the first parameter (which is the impl-list at the engine
        # level) with the user-facing ``tensors: list[Tensor]``.
        params = list(sig.parameters.values())
        if params:
            params[0] = inspect.Parameter(
                "tensors",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        else:
            params = [
                inspect.Parameter(
                    "tensors",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        return sig.replace(parameters=params)

    return _rename_leading_tensor_params(sig, entry.n_tensor_args)


# ── Wrapper factories ────────────────────────────────────────────────────────


def _make_free_fn(name: str) -> object:
    """Build a free function that unwraps Tensors → TensorImpls, calls the
    engine op, and rewraps the result.  The returned wrapper carries an
    explicit ``__signature__`` derived from the registry entry."""
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name or entry.name
        if fn_name == name:
            e = entry

            if e.n_tensor_args == -1:

                def _fn_list(
                    tensors: "list[Tensor]", *args: object, **kwargs: object
                ) -> object:
                    impls = [_unwrap(t) for t in tensors]
                    result = e.engine_fn(impls, *args, **kwargs)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result

                _fn_list.__name__ = fn_name
                _fn_list.__qualname__ = fn_name
                _fn_list.__signature__ = _signature_for_entry(e)  # type: ignore[attr-defined]
                return _fn_list

            def _fn(*args: object, **kwargs: object) -> object:
                proc: list[object] = []
                for i, a in enumerate(args):
                    if i < e.n_tensor_args and hasattr(a, "_impl"):
                        proc.append(_unwrap(a))
                    else:
                        proc.append(a)
                result = e.engine_fn(*proc, **kwargs)
                if e.returns_tensor:
                    if isinstance(result, (list, tuple)):
                        return type(result)(_wrap(r) for r in result)
                    return _wrap(result)
                return result

            _fn.__name__ = fn_name
            _fn.__qualname__ = fn_name
            _fn.__signature__ = _signature_for_entry(e)  # type: ignore[attr-defined]
            return _fn
    raise AttributeError(f"No op found for free function: {name}")


# ── Module population ────────────────────────────────────────────────────────


_FREE_FN_NAMES: set[str] = set()


def _populate_free_fns() -> None:
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name
        if fn_name is None:
            continue
        if fn_name in _FREE_FN_NAMES:
            continue
        _FREE_FN_NAMES.add(fn_name)
        globals()[fn_name] = _make_free_fn(fn_name)


_populate_free_fns()
