"""
Tensor-level gradient hooks (``Tensor.register_hook``).

A hook runs inside the backward pass, where the tensor's gradient is
complete: for a non-leaf, on its producer's output slot once every
consumer's contribution has arrived and before the producer runs; for a
leaf, just before the gradient is added into ``.grad``.  What a hook
returns is the gradient that flows on.  The engine owns where and when
(``_C_engine._tensor_hook_runner``); this module owns what a slot's hooks
are — their list, their order, their removal, and the check that each one
returned a tensor or ``None``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


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
    this one operates on the per-tensor hooks the engine runs on a
    tensor's gradient during backward.

    Parameters
    ----------
    hooks_list : list of callable
        The hook list of the tensor's gradient slot; on
        :meth:`remove` the hook is dropped from this list.
    hook : callable
        The exact callable to remove. Identity (``is``) is used
        for matching.

    Attributes
    ----------
    _hooks_list : list of callable
        The slot's hook list (private).
    _hook : callable
        The registered hook (private).

    Notes
    -----
    The hook contract is

    .. math::

        \bar x \leftarrow h(\bar x),

    where :math:`\bar x = \partial \mathcal{L} / \partial x` is
    the gradient reaching :math:`x` in the current backward pass
    and :math:`h` is the user-supplied hook. A hook returning
    ``None`` leaves :math:`\bar x` unchanged; returning a tensor
    replaces it for everything upstream of :math:`x`.

    :meth:`remove` is idempotent — calling it twice is harmless, and
    removes one registration even when the same callable was
    registered more than once.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> handle = x.register_hook(lambda g: print('grad =', g))
    >>> (x * x).sum().backward()
    grad = tensor([2., 4.])
    >>> handle.remove()

    As a context manager:

    >>> with x.register_hook(lambda g: g * 2) as h:
    ...     (x * x).sum().backward()
    >>> x.grad
    tensor([6., 12.])
    """

    def __init__(
        self, hooks_list: list[Callable[..., object]], hook: Callable[..., object]
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._hooks_list = hooks_list
        self._hook = hook
        self._removed = False

    def remove(self) -> None:
        """Remove the hook from the tensor."""
        if self._removed:
            return
        self._removed = True
        for i, registered in enumerate(self._hooks_list):
            if registered is self._hook:
                del self._hooks_list[i]
                return

    def __enter__(self) -> RemovableHandle:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
        self.remove()


class _HookRunner:
    """The hooks of one gradient slot, run in registration order.

    The engine holds one per slot and calls it with the slot's gradient.
    It holds no reference to the tensor: the engine keeps it on the
    producer node (or the leaf's autograd metadata), and a reference back
    would be a cycle the collector cannot see through the engine.
    """

    __slots__ = ("hooks",)

    def __init__(self) -> None:
        self.hooks: list[Callable[..., object]] = []

    def __call__(self, grad: _C_engine.TensorImpl) -> _C_engine.TensorImpl | None:
        """Run every hook on the gradient; the last tensor returned flows on.

        Returns ``None`` when no hook replaced the gradient, so the engine
        keeps the one it handed over, with any in-place write a hook made.

        Raises
        ------
        TypeError
            A hook returned something that is neither a tensor nor ``None``.
        """
        from lucid._tensor.tensor import Tensor

        g = _wrap(grad)
        replaced = False
        # A snapshot: a hook may register or remove hooks on this slot.
        for hook in tuple(self.hooks):
            result = hook(g)
            if result is None:
                continue
            if not isinstance(result, Tensor):
                raise TypeError(
                    f"a tensor hook must return a Tensor or None, but "
                    f"{getattr(hook, '__qualname__', repr(hook))} returned "
                    f"{type(result).__name__}"
                )
            g = result
            replaced = True
        return _unwrap(g) if replaced else None


def _register_tensor_hook(
    tensor: Tensor, hook: Callable[[Tensor], Tensor | None]
) -> RemovableHandle:
    """Add *hook* to *tensor*'s gradient slot and return a removable handle.

    Called by :meth:`~lucid.Tensor.register_hook`.  The slot's runner is
    installed on the first registration and shared by every later one.
    """
    runner = _C_engine._tensor_hook_runner(_unwrap(tensor), _HookRunner)
    runner.hooks.append(hook)
    return RemovableHandle(runner.hooks, hook)
