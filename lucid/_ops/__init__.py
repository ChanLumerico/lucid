"""
Free-function ops exposed as lucid.xxx.

All functions are generated from the registry. Pure-Python implementations
for missing engine ops (std, log_softmax, any, all) are also defined here.
"""

from typing import Any, TYPE_CHECKING
import numpy as np
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import _REGISTRY

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _make_free_fn(name: str) -> Any:
    """Create a free function that wraps an engine function."""
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name or entry.name
        if fn_name == name:
            e = entry

            if e.n_tensor_args == -1:
                def _fn_list(tensors: list[Tensor], *args: Any) -> Any:
                    impls = [_unwrap(t) for t in tensors]
                    result = e.engine_fn(impls, *args)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result
                _fn_list.__name__ = fn_name
                return _fn_list
            else:
                def _fn(*args: Any) -> Any:
                    proc: list[Any] = []
                    for i, a in enumerate(args):
                        if i < e.n_tensor_args and hasattr(a, "_impl"):
                            proc.append(_unwrap(a))
                        else:
                            proc.append(a)
                    result = e.engine_fn(*proc)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result
                _fn.__name__ = fn_name
                return _fn
    raise AttributeError(f"No op found for free function: {name}")


# ── generate all free functions from registry ─────────────────────────────

_FREE_FN_NAMES = set()

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


# ── Python-level implementations ──────────────────────────────────────────

def std(x: Tensor, axes: list[int] | None = None, keepdims: bool = False) -> Tensor:
    """Standard deviation (sqrt of variance)."""
    axes_ = axes if axes is not None else []
    return _wrap(_C_engine.sqrt(_C_engine.var(_unwrap(x), axes_, keepdims)))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax."""
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def any(x: Tensor) -> Tensor:
    """Return True if any element is non-zero."""
    val = bool(np.asarray(x._impl.data_as_python()).any())
    arr = np.array(val)
    return _wrap(_C_engine.TensorImpl(arr, _unwrap(x).device, False))


def all(x: Tensor) -> Tensor:
    """Return True if all elements are non-zero."""
    val = bool(np.asarray(x._impl.data_as_python()).all())
    arr = np.array(val)
    return _wrap(_C_engine.TensorImpl(arr, _unwrap(x).device, False))


def rsqrt(x: Tensor) -> Tensor:
    """Reciprocal square root: 1 / sqrt(x)."""
    return _wrap(_C_engine.reciprocal(_C_engine.sqrt(_unwrap(x))))


def detach(x: Tensor) -> Tensor:
    """Return a new tensor detached from the autograd graph."""
    arr = np.ascontiguousarray(np.asarray(x._impl.data_as_python()))
    impl = _C_engine.TensorImpl(arr, x._impl.device, False)
    return _wrap(impl)


def clone(x: Tensor) -> Tensor:
    """Return a deep copy of x, preserving autograd history."""
    return _wrap(_C_engine.contiguous(_unwrap(x)))


__all__ = list(_FREE_FN_NAMES) + ["std", "log_softmax", "any", "all", "detach", "clone"]
