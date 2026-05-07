"""Predicates and zero-cost identity ops surfaced for PyTorch parity."""

from typing import TYPE_CHECKING

import lucid
from lucid._ops.composite._shared import _is_tensor
from lucid._types import TensorLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def numel(x: Tensor) -> int:
    return int(x.numel())


def is_storage(x: Tensor) -> bool:
    """Lucid has no separate Storage type — always False."""
    return False


def is_nonzero(x: Tensor) -> bool:
    if x.numel() != 1:
        raise RuntimeError("is_nonzero is defined only for scalar tensors (numel == 1)")
    return bool(x.item() != 0)


def is_same_size(a: Tensor, b: Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape)


def is_neg(x: Tensor) -> bool:
    return False


def is_conj(x: Tensor) -> bool:
    return False


def isin(
    elements: Tensor | TensorLike,
    test_elements: Tensor | TensorLike,
    *,
    invert: bool = False,
) -> Tensor:
    """Per-element membership test against ``test_elements``."""
    if not _is_tensor(elements):
        elements = lucid.tensor(elements)
    if not _is_tensor(test_elements):
        test_elements = lucid.tensor(test_elements)
    e_flat = elements.reshape(-1)
    t_flat = test_elements.reshape(-1)
    n, m = e_flat.shape[0], t_flat.shape[0]
    e_bc = e_flat.unsqueeze(1).expand(n, m)
    t_bc = t_flat.unsqueeze(0).expand(n, m)
    matches = (e_bc == t_bc).to(dtype=lucid.float32).sum(1)
    out = (matches > 0.0).reshape(elements.shape)
    return ~out if invert else out


def isneginf(x: Tensor) -> Tensor:
    return lucid.logical_and(lucid.isinf(x), x < 0.0)


def isposinf(x: Tensor) -> Tensor:
    return lucid.logical_and(lucid.isinf(x), x > 0.0)


def isreal(x: Tensor) -> Tensor:
    """Real tensors are always real — all-True bool tensor."""
    return lucid.isfinite(x) | lucid.isnan(x) | lucid.isinf(x)


def conj_physical(x: Tensor) -> Tensor:
    """Materialised conjugate.  Lucid never carries lazy conjugate metadata,
    so this is just an alias for :func:`lucid.conj` (the engine op)."""
    return lucid.conj(x)


def resolve_conj(x: Tensor) -> Tensor:
    """No-op: Lucid never lazily marks tensors as conjugated, so there is
    nothing to resolve.  Provided for API parity."""
    return x


def resolve_neg(x: Tensor) -> Tensor:
    """No-op: Lucid never lazily marks tensors as negated.  Provided for
    API parity."""
    return x


__all__ = [
    "numel",
    "is_storage",
    "is_nonzero",
    "is_same_size",
    "is_neg",
    "is_conj",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "conj_physical",
    "resolve_conj",
    "resolve_neg",
]
