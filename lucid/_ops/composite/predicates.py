"""Predicates and zero-cost identity ops surfaced for PyTorch parity."""

import lucid
from lucid._ops.composite._shared import _is_tensor


def numel(x) -> int:  # type: ignore[no-untyped-def]
    return int(x.numel())


def is_storage(x) -> bool:  # type: ignore[no-untyped-def]
    """Lucid has no separate Storage type — always False."""
    return False


def is_nonzero(x) -> bool:  # type: ignore[no-untyped-def]
    if x.numel() != 1:
        raise RuntimeError("is_nonzero is defined only for scalar tensors (numel == 1)")
    return bool(x.item() != 0)


def is_same_size(a, b) -> bool:  # type: ignore[no-untyped-def]
    return tuple(a.shape) == tuple(b.shape)


def is_neg(x) -> bool:  # type: ignore[no-untyped-def]
    return False


def is_conj(x) -> bool:  # type: ignore[no-untyped-def]
    return False


def isin(elements, test_elements, *, invert: bool = False):  # type: ignore[no-untyped-def]
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


def isneginf(x):  # type: ignore[no-untyped-def]
    return lucid.logical_and(lucid.isinf(x), x < 0.0)


def isposinf(x):  # type: ignore[no-untyped-def]
    return lucid.logical_and(lucid.isinf(x), x > 0.0)


def isreal(x):  # type: ignore[no-untyped-def]
    """Real tensors are always real — all-True bool tensor."""
    return lucid.isfinite(x) | lucid.isnan(x) | lucid.isinf(x)


def conj(x):  # type: ignore[no-untyped-def]
    return x


def conj_physical(x):  # type: ignore[no-untyped-def]
    return x


def resolve_conj(x):  # type: ignore[no-untyped-def]
    return x


def resolve_neg(x):  # type: ignore[no-untyped-def]
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
    "conj",
    "conj_physical",
    "resolve_conj",
    "resolve_neg",
]
