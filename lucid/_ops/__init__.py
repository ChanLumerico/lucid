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

                def _fn(*args: Any, **kwargs: Any) -> Any:
                    proc: list[Any] = []
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


def detach(x: Tensor) -> Tensor:
    """Return a new tensor detached from the autograd graph."""
    arr = np.ascontiguousarray(np.asarray(x._impl.data_as_python()))
    impl = _C_engine.TensorImpl(arr, x._impl.device, False)
    return _wrap(impl)


def clone(x: Tensor) -> Tensor:
    """Return a deep copy of x, preserving autograd history."""
    return _wrap(_C_engine.contiguous(_unwrap(x)))


def clamp(x: Tensor, min: float | None = None, max: float | None = None) -> Tensor:
    """Clamp all elements to [min, max]. Alias for clip."""
    lo = min if min is not None else float("-inf")
    hi = max if max is not None else float("inf")
    return _wrap(_C_engine.clip(_unwrap(x), lo, hi))


__all__ = list(_FREE_FN_NAMES) + ["detach", "clone", "clamp"]
