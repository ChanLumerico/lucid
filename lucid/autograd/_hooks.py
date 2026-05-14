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
    r"""Handle returned by :meth:`~lucid.Tensor.register_hook`.

    A ``RemovableHandle`` keeps a reference to the list of hooks
    attached to a tensor's gradient and to the specific callable
    that was just registered, providing two ways to take the hook
    back off:

    * Imperative — call :meth:`remove` whenever the hook is no
      longer needed.
    * RAII — bind the handle in a ``with`` block; the hook is
      automatically removed when control leaves the block.

    This class is *distinct* from
    ``lucid.nn.hooks.RemovableHandle``: that one manages
    forward/backward hooks on :class:`~lucid.nn.Module`, whereas
    this one operates on the per-tensor hooks dispatched from
    :meth:`~lucid.Tensor.backward` after the gradient has been
    accumulated.

    Parameters
    ----------
    hooks_list : list of callable
        The underlying registry list owned by the tensor; on
        :meth:`remove` the hook is dropped from this list.
    hook : callable
        The exact callable to remove. Identity (``is``) is used
        for matching.

    Attributes
    ----------
    _hooks_list : list of callable
        The registry list (private).
    _hook : callable
        The registered hook (private).

    Notes
    -----
    The post-backward hook contract is

    .. math::

        \bar x \leftarrow h(\bar x),

    where :math:`\bar x = \partial \mathcal{L} / \partial x` is
    the accumulated gradient and :math:`h` is the user-supplied
    hook. A hook returning ``None`` leaves :math:`\bar x`
    unchanged; returning a tensor replaces it.

    :meth:`remove` is idempotent — calling it twice is harmless.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> handle = x.register_hook(lambda g: print('grad =', g))
    >>> (x * x).sum().backward()
    >>> handle.remove()

    As a context manager:

    >>> with x.register_hook(lambda g: g * 2) as h:
    ...     (x * x).sum().backward()
    """

    def __init__(
        self, hooks_list: list[Callable[..., object]], hook: Callable[..., object]
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._hooks_list = hooks_list
        self._hook = hook

    def remove(self) -> None:
        """Remove the hook from the tensor."""
        try:
            self._hooks_list.remove(self._hook)
        except ValueError:
            pass  # already removed

    def __enter__(self) -> RemovableHandle:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
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
