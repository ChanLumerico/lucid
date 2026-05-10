"""Helpers shared across the ``lucid._ops.composite`` submodules.

Composite ops are pure-Python — they layer on top of the registered
engine primitives via the ``lucid`` top-level namespace and Tensor
operator overloads.

Conventions used by every composite:

* prefer **operator overloads** (``+ - * / == < >``) — they go through
  ``Tensor.__add__`` etc., broadcast, and accept Python scalars;
* the free-function forms in ``lucid`` (``lucid.add``, ``lucid.sub``, ...)
  are thin engine wrappers and reject Python scalars — only use them
  when both operands are already tensors;
* for axis swaps, build a ``permute`` permutation — the engine's
  ``transpose`` reverses *all* dims, not just two.
"""

from typing import TypeGuard

from lucid._tensor.tensor import Tensor


def _is_tensor(x: object) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor)


def _swap_dims(t: Tensor, d0: int, d1: int) -> Tensor:
    """Swap exactly two dims by building a permute permutation.

    The engine's ``transpose`` reverses every dim — we don't want that.
    """
    n = t.ndim
    perm = list(range(n))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return t.permute(*perm)
