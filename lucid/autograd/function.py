"""
lucid.autograd.function — public autograd entry points.

Backward / grad pass-through:
  * The graph itself is built and traversed inside the C++ engine
    (`_C_engine.engine_backward`). The Python side merely seeds the
    output gradient (defaults to ones) and routes the call.

Grad-mode controls:
  * `no_grad` and `enable_grad` are context managers AND decorators that
    push/pop the engine's thread-local `GradMode` flag — when grad is
    disabled, ops skip building the autograd graph entirely.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, Iterable, Sequence
from types import TracebackType

import numpy as np

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of, to_engine_dtype


__all__ = [
    "backward", "grad",
    "no_grad", "enable_grad", "set_grad_enabled", "is_grad_enabled",
]


# --------------------------------------------------------------------------- #
# Backward / grad
# --------------------------------------------------------------------------- #

def backward(
    tensor: Tensor,
    /,
    grad: Tensor | None = None,
    retain_graph: bool = False,
) -> None:
    """Run backward starting at `tensor`.

    If `grad` is None, the engine seeds with `ones_like(tensor)`.
    If provided, it must be broadcast-compatible with `tensor`.
    """
    if grad is not None:
        # Stash user-supplied seed into the root tensor's .grad before
        # triggering the engine traversal. Engine treats a pre-existing
        # grad as the seed.
        tensor.grad = grad
    _C_engine.engine_backward(impl_of(tensor), bool(retain_graph))


def _as_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def grad(
    outputs: Tensor | Iterable[Tensor],
    inputs: Tensor | Iterable[Tensor],
    grad_outputs: Tensor | Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[Tensor, ...] | Tensor:
    """Compute gradients of `outputs` w.r.t. `inputs`.

    Mirrors PyTorch's `torch.autograd.grad`. Implemented by:
      1. Saving each input's existing `.grad`.
      2. Running backward from each output (with optional seed).
      3. Snapshotting the resulting input grads into a tuple.
      4. Restoring each input's pre-call `.grad`.
    """
    out_tensors = _as_tuple(outputs)
    in_tensors = _as_tuple(inputs)
    if grad_outputs is None:
        grad_outs: tuple[Tensor | None, ...] = (None,) * len(out_tensors)
    else:
        grad_outs = _as_tuple(grad_outputs)
        if len(grad_outs) != len(out_tensors):
            raise ValueError(
                "grad_outputs length must match outputs length.")

    for t in out_tensors:
        if not isinstance(t, Tensor):
            raise TypeError("All outputs must be Tensor instances.")
    for t in in_tensors:
        if not isinstance(t, Tensor):
            raise TypeError("All inputs must be Tensor instances.")

    # Snapshot prior grads so we can restore them.
    prev_in_grads = [t.grad for t in in_tensors]

    # Zero the input grads so we can isolate this call's contribution.
    for t in in_tensors:
        t.zero_grad()

    try:
        for i, (out, gout) in enumerate(zip(out_tensors, grad_outs)):
            if not out.requires_grad:
                if allow_unused:
                    continue
                raise RuntimeError("All outputs must require gradients.")
            backward(
                out, grad=gout,
                retain_graph=(i < len(out_tensors) - 1) or retain_graph,
            )

        results = tuple(t.grad for t in in_tensors)
        if not allow_unused and any(g is None for g in results):
            raise RuntimeError(
                "Some inputs did not receive gradients. "
                "Set allow_unused=True to skip this check.")
        if len(results) == 1:
            return results[0]
        return results
    finally:
        for t, g in zip(in_tensors, prev_in_grads):
            t.grad = g


# --------------------------------------------------------------------------- #
# Grad-mode control
# --------------------------------------------------------------------------- #

def is_grad_enabled() -> bool:
    return _C_engine.grad_enabled()


def set_grad_enabled(value: bool) -> None:
    _C_engine.set_grad_enabled(bool(value))


class _GradModeContext(AbstractContextManager):
    """Context manager + decorator that toggles engine grad mode.

    Used as both `no_grad` (target=False) and `enable_grad` (target=True).
    """

    __slots__ = ("_target", "_prev")

    def __init__(self, target: bool) -> None:
        self._target = bool(target)
        self._prev: bool | None = None

    def __enter__(self) -> "_GradModeContext":
        self._prev = _C_engine.grad_enabled()
        _C_engine.set_grad_enabled(self._target)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._prev is not None:
            _C_engine.set_grad_enabled(self._prev)
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        target = self._target

        @wraps(func)
        def wrapper(*args, **kwargs):
            with _GradModeContext(target):
                return func(*args, **kwargs)
        return wrapper


def no_grad() -> _GradModeContext:
    """Context manager / decorator that disables autograd graph building."""
    return _GradModeContext(False)


def enable_grad() -> _GradModeContext:
    """Context manager / decorator that re-enables autograd inside a no_grad scope."""
    return _GradModeContext(True)
