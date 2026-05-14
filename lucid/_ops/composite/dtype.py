"""Dtype-promotion helpers (``result_type``, ``promote_types``, ``can_cast``)."""

from typing import TYPE_CHECKING

import lucid
from lucid._dtype import dtype as _DType
from lucid._ops.composite._shared import _is_tensor
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _kind_width(d: _DType) -> tuple[int, int]:
    """Return ``(kind, width)`` keys used for ordering dtypes during promotion.

    ``kind`` orders families (``0`` boolean, ``1`` integer, ``2`` floating,
    ``3`` complex) and ``width`` orders bit-widths within a family.
    Unknown dtypes default to ``(2, 32)`` (i.e. ``float32``).
    """
    table: dict[str, tuple[int, int]] = {
        "lucid.bool": (0, 1),
        "lucid.int8": (1, 8),
        "lucid.int16": (1, 16),
        "lucid.int32": (1, 32),
        "lucid.int64": (1, 64),
        "lucid.float16": (2, 16),
        "lucid.bfloat16": (2, 16),
        "lucid.float32": (2, 32),
        "lucid.float64": (2, 64),
        "lucid.complex64": (3, 64),
    }
    return table.get(str(d), (2, 32))


def _promote(a_dtype: _DType, b_dtype: _DType) -> _DType:
    """Return the wider of two dtypes according to ``(kind, width)`` ordering.

    The higher-kind dtype wins outright (e.g. ``float`` beats ``int``); ties
    in kind are broken by the wider bit-width.
    """
    ka, wa = _kind_width(a_dtype)
    kb, wb = _kind_width(b_dtype)
    if ka != kb:
        return a_dtype if ka > kb else b_dtype
    return a_dtype if wa >= wb else b_dtype


def result_type(a: Tensor | Scalar, b: Tensor | Scalar) -> _DType:
    """Determine the promoted dtype of an operation on ``a`` and ``b``.

    Parameters
    ----------
    a, b : Tensor | Scalar
        Operands. Python scalars do not contribute a dtype and are ignored
        unless both arguments are scalar (in which case ``float32`` is used).

    Returns
    -------
    DType
        The dtype that the binary operation would produce.
    """
    da = a.dtype if _is_tensor(a) else None
    db = b.dtype if _is_tensor(b) else None
    if da is None and db is None:
        return lucid.float32  # type: ignore[return-value]
    if da is None:
        return db  # type: ignore[return-value]
    if db is None:
        return da
    if da == db:
        return da
    return _promote(da, db)


def promote_types(a_dtype: _DType, b_dtype: _DType) -> _DType:
    """Return the dtype that ``a_dtype`` and ``b_dtype`` jointly promote to.

    Parameters
    ----------
    a_dtype, b_dtype : DType
        Operand dtypes.

    Returns
    -------
    DType
        Promoted dtype, equal to either input or the wider of the two
        according to the standard kind/width ordering.
    """
    if a_dtype == b_dtype:
        return a_dtype
    return _promote(a_dtype, b_dtype)


def can_cast(from_dtype: _DType, to_dtype: _DType) -> bool:
    """Return ``True`` if ``from_dtype`` can be cast to ``to_dtype`` without loss.

    A cast is considered safe when promotion of the two dtypes equals
    ``to_dtype`` — i.e. ``to_dtype`` already includes the value range and
    precision of ``from_dtype``.
    """
    return promote_types(from_dtype, to_dtype) == to_dtype


__all__ = ["result_type", "promote_types", "can_cast"]
