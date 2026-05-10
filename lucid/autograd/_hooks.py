"""
Tensor-level gradient hooks (``Tensor.register_hook``).

Hooks fire after :meth:`~lucid.Tensor.backward` completes, receiving the
accumulated gradient of the tensor they are registered on.  Hooks may
return a replacement gradient; returning ``None`` leaves the gradient
unchanged.

The module exposes two public helpers used by ``Tensor.register_hook``:

* :func:`_register_tensor_hook` — add a hook and return a removable handle.
* :func:`_dispatch_tensor_grad_hooks` — called by ``Tensor.backward`` to
  fire every pending hook after ``engine_backward`` has run.

Design notes
------------
We keep a module-level registry ``_TENSOR_HOOKS`` keyed by the Python
``id()`` of the owning tensor.  A ``weakref.finalize`` callback prunes the
entry automatically when the tensor is garbage-collected.  This is the
"post-backward" model — hooks fire with the *final* accumulated gradient
stored in ``.grad``.  This is semantically correct for leaf tensors (the
dominant use case: parameter gradient logging/clipping).  For non-leaf
tensors, use ``retain_grad()`` before the forward pass so the grad is
preserved.
"""

import weakref
from typing import Callable, TYPE_CHECKING, cast

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── Registry ─────────────────────────────────────────────────────────────────

# id(tensor) → (hooks_list, weakref_to_tensor)
_TENSOR_HOOKS: dict[int, tuple[list[Callable[..., object]], weakref.ref[object]]] = {}


class RemovableHandle:
    """Handle returned by :meth:`~lucid.Tensor.register_hook`.

    Call :meth:`remove` to de-register the hook; the handle can also be
    used as a context manager:

    .. code-block:: python

        with x.register_hook(lambda g: print(g)) as h:
            y = x.sum()
            y.backward()
        # hook is removed here
    """

    def __init__(
        self, hooks_list: list[Callable[..., object]], hook: Callable[..., object]
    ) -> None:
        self._hooks_list = hooks_list
        self._hook = hook

    def remove(self) -> None:
        """Remove the hook from the tensor."""
        try:
            self._hooks_list.remove(self._hook)
        except ValueError:
            pass  # already removed

    def __enter__(self) -> RemovableHandle:
        return self

    def __exit__(self, *args: object) -> None:
        self.remove()


def _register_tensor_hook(
    tensor: Tensor, hook: Callable[..., object]
) -> RemovableHandle:
    """Register *hook* on *tensor*'s gradient and return a removable handle.

    Called by :meth:`~lucid.Tensor.register_hook`.
    """
    tid = id(tensor)
    if tid not in _TENSOR_HOOKS:
        hooks_list: list[Callable[..., object]] = []

        # Clean up the registry entry when the tensor is GC'd.
        def _finalizer(_tid: int = tid) -> None:
            _TENSOR_HOOKS.pop(_tid, None)

        _TENSOR_HOOKS[tid] = (hooks_list, weakref.ref(tensor, lambda _: _finalizer()))
    else:
        hooks_list, _ = _TENSOR_HOOKS[tid]

    hooks_list.append(hook)
    return RemovableHandle(hooks_list, hook)


def _dispatch_tensor_grad_hooks() -> None:
    """Fire all registered tensor-gradient hooks after backward completes.

    Called by :meth:`~lucid.Tensor.backward` (and by the free-function
    ``lucid.autograd.backward``) immediately after ``engine_backward``
    returns.
    """
    from lucid._tensor.tensor import Tensor
    from lucid._dispatch import _wrap

    stale: list[int] = []
    for tid, (hooks, wr) in _TENSOR_HOOKS.items():
        t = wr()
        if t is None:
            stale.append(tid)
            continue
        if not hooks:
            continue

        t_tensor = cast(Tensor, t)
        # Retrieve the current gradient.
        g_impl = t_tensor._impl.grad_as_impl()
        if g_impl is None:
            g_raw = t_tensor._impl.grad_as_python()
            if g_raw is None:
                continue  # no grad for this tensor yet
            g_impl = _C_engine.TensorImpl(g_raw, t_tensor._impl.device, False)

        grad = _wrap(g_impl)

        # Call hooks in registration order; a non-None return replaces the grad.
        for hook in hooks:
            result = hook(grad)
            if result is not None:
                if not isinstance(result, Tensor):
                    raise TypeError(
                        f"Tensor grad hook must return a Tensor or None, "
                        f"got {type(result).__name__}"
                    )
                grad = result

        # Write the (possibly modified) gradient back.
        t_tensor._impl.set_grad(_unwrap(grad))

    for tid in stale:
        _TENSOR_HOOKS.pop(tid, None)
