"""
Auto-inject Tensor methods from the ops registry.

Called once at module import time by tensor.py.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""
    from lucid._ops._registry import _REGISTRY, OpEntry
    from lucid._dispatch import _unwrap, _wrap
    from lucid._C import engine as _C_engine

    def _make_method(e: OpEntry) -> Any:
        if e.n_tensor_args == -1:
            def method_list(self: "Tensor", tensors: list[Any], *args: Any) -> Any:
                all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
                result = e.engine_fn(all_tensors, *args)
                if e.returns_tensor:
                    if isinstance(result, (list, tuple)):
                        return type(result)(_wrap(r) for r in result)
                    return _wrap(result)
                return result
            method_list.__name__ = e.method_name or e.name
            return method_list
        else:
            def method(self: "Tensor", *args: Any) -> Any:
                # Unwrap any Tensor in extra tensor arg positions
                proc_args: list[Any] = []
                from lucid._tensor.tensor import Tensor as _T
                for i, a in enumerate(args):
                    if i < (e.n_tensor_args - 1) and isinstance(a, _T):
                        proc_args.append(_unwrap(a))
                    else:
                        proc_args.append(a)

                if e.inplace:
                    result = e.engine_fn(self._impl, *proc_args)
                    self._impl = result
                    return self
                else:
                    result = e.engine_fn(self._impl, *proc_args)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result
            method.__name__ = e.method_name or e.name
            return method

    for entry in _REGISTRY:
        if entry.method_name is None:
            continue
        # skip dunders — handled in _dunders.py
        if entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry))

    # ── methods not in registry (Python-level implementations) ───────────

    def view(self: "Tensor", *shape: Any) -> "Tensor":
        """Return a tensor with the same data but a different shape."""
        s = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape)
        return _wrap(_C_engine.reshape(self._impl, s))

    def t(self: "Tensor") -> "Tensor":
        """Return the 2D transpose of this tensor."""
        return _wrap(_C_engine.T(self._impl))

    def std(self: "Tensor", axes: list[int] | None = None, keepdims: bool = False) -> "Tensor":
        """Return standard deviation (implemented as sqrt(var(x)))."""
        axes_: list[int] = axes if axes is not None else []
        return _wrap(_C_engine.sqrt(_C_engine.var(self._impl, axes_, keepdims)))

    def log_softmax(self: "Tensor", axis: int = -1) -> "Tensor":
        """Return log-softmax along the given axis."""
        sm = _C_engine.softmax(self._impl, axis)
        return _wrap(_C_engine.log(sm))

    def any(self: "Tensor") -> "Tensor":
        """Return True if any element is non-zero."""
        import numpy as np
        val = bool(np.asarray(self._impl.data_as_python()).any())
        arr = np.array(val)
        return _wrap(_C_engine.TensorImpl(arr, self._impl.device, False))

    def all(self: "Tensor") -> "Tensor":
        """Return True if all elements are non-zero."""
        import numpy as np
        val = bool(np.asarray(self._impl.data_as_python()).all())
        arr = np.array(val)
        return _wrap(_C_engine.TensorImpl(arr, self._impl.device, False))

    for _name, _fn in [
        ("view", view), ("t", t), ("std", std),
        ("log_softmax", log_softmax), ("any", any), ("all", all),
    ]:
        setattr(tensor_cls, _name, _fn)
