"""Dtype-promotion helpers (``result_type``, ``promote_types``, ``can_cast``)."""

from typing import TYPE_CHECKING

import lucid
from lucid._dtype import dtype as _DType
from lucid._ops.composite._shared import _is_tensor
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _kind_width(d: _DType) -> tuple[int, int]:
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
    ka, wa = _kind_width(a_dtype)
    kb, wb = _kind_width(b_dtype)
    if ka != kb:
        return a_dtype if ka > kb else b_dtype
    return a_dtype if wa >= wb else b_dtype


def result_type(a: Tensor | Scalar, b: Tensor | Scalar) -> _DType:
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
    if a_dtype == b_dtype:
        return a_dtype
    return _promote(a_dtype, b_dtype)


def can_cast(from_dtype: _DType, to_dtype: _DType) -> bool:
    return promote_types(from_dtype, to_dtype) == to_dtype


__all__ = ["result_type", "promote_types", "can_cast"]
