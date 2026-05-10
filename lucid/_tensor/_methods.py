"""
Auto-inject Tensor methods from the ops registry.

Every method is generated from an ``OpEntry`` in ``_registry.py``; signature
translation (``dim=None|int|list``, ``keepdim``, ``correction`` ...) lives
in ``_adapters.py``.  Only ``eval`` — which calls into the impl handle
directly and isn't an engine op — is special-cased below.

Called once at module import time by ``tensor.py``.
"""

from typing import TYPE_CHECKING, cast

from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import _REGISTRY, OpEntry

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""

    def _make_method(e: OpEntry) -> object:
        if e.n_tensor_args == -1:

            def method_list(
                self: Tensor,
                tensors: list[Tensor],
                *args: object,
                **kwargs: object,
            ) -> Tensor:
                all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
                result = e.engine_fn(all_tensors, *args, **kwargs)
                if e.returns_tensor:
                    if isinstance(result, (list, tuple)):
                        return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
                    return _wrap(result)
                return cast(Tensor, result)

            method_list.__name__ = e.method_name or e.name
            return method_list
        else:

            def method(self: Tensor, *args: object, **kwargs: object) -> Tensor:
                # Unwrap any Tensor in extra tensor arg positions
                proc_args: list[object] = []
                for i, a in enumerate(args):
                    if i < (e.n_tensor_args - 1) and isinstance(a, tensor_cls):
                        proc_args.append(_unwrap(a))  # type: ignore[arg-type]
                    else:
                        proc_args.append(a)

                if e.inplace:
                    result = e.engine_fn(self._impl, *proc_args, **kwargs)
                    self._impl = result
                    return self
                else:
                    result = e.engine_fn(self._impl, *proc_args, **kwargs)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
                        return _wrap(result)
                    return cast(Tensor, result)

            method.__name__ = e.method_name or e.name
            return method

    for entry in _REGISTRY:
        if entry.method_name is None:
            continue
        # skip dunders — handled in _dunders.py
        if entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry))

    # ── methods that bypass the registry — only those with no clean
    # registry mapping live here.  Everything else is auto-injected from
    # ``_REGISTRY`` above (now that ``_make_method`` forwards keyword args
    # the registry adapters handle the API-compat translation directly).

    def eval(self: Tensor) -> Tensor:
        """Force immediate evaluation of this tensor.

        Special-cased because it operates on the impl handle rather than going
        through any engine op — so it has no place in the ops registry.
        On Metal this flushes the lazy compute graph; on CPU it's a no-op.
        Returns ``self`` for chaining.
        """
        self._impl.eval()
        return self

    setattr(tensor_cls, "eval", eval)
