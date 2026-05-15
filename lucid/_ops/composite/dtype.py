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
    r"""Compute the dtype that a binary operation on the inputs would produce.

    Implements NumPy-style type promotion: tensors contribute their
    declared dtype, while Python scalars are treated as having no dtype
    of their own and follow whichever tensor operand they appear with.

    Parameters
    ----------
    a : Tensor | Scalar
        First operand. Tensor operands contribute their dtype; Python
        scalars do not.
    b : Tensor | Scalar
        Second operand. Same convention as ``a``.

    Returns
    -------
    DType
        The dtype that an arithmetic operation on ``a`` and ``b`` would
        produce.

    Notes
    -----
    The promotion algorithm groups dtypes into four kinds — bool
    :math:`< \text{int} < \text{float} < \text{complex}` — and uses
    bit-width as a tiebreaker. Concretely:

    .. math::

        \text{result\_type}(a, b) =
        \begin{cases}
            \text{dtype}(a),                        & b\ \text{is scalar}, \\
            \text{dtype}(b),                        & a\ \text{is scalar}, \\
            \text{float32},                          & \text{both scalar}, \\
            \operatorname{promote}(\text{dtype}(a), \text{dtype}(b)), & \text{otherwise}.
        \end{cases}

    See also :func:`promote_types` (operates directly on dtype objects).

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1, 2], dtype=lucid.int32)
    >>> b = lucid.tensor([1., 2.], dtype=lucid.float32)
    >>> lucid.result_type(a, b)
    lucid.float32
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
    r"""Compute the joint promotion of two dtypes.

    Operates on dtype objects directly (no tensor required). Combines
    the two dtypes according to the standard kind/width ordering:
    higher kind (bool :math:`<` int :math:`<` float :math:`<` complex)
    wins outright, and ties in kind are broken by the wider bit-width.

    Parameters
    ----------
    a_dtype : DType
        First dtype.
    b_dtype : DType
        Second dtype.

    Returns
    -------
    DType
        The dtype that values of either input dtype would be promoted to
        in a binary operation.

    Notes
    -----
    With ``kind(d)`` and ``width(d)`` denoting the kind and bit-width
    of a dtype:

    .. math::

        \text{promote}(d_a, d_b) =
        \begin{cases}
            d_a, & \text{kind}(d_a) > \text{kind}(d_b), \\
            d_b, & \text{kind}(d_a) < \text{kind}(d_b), \\
            d_a, & \text{kind equal and width}(d_a) \geq \text{width}(d_b), \\
            d_b, & \text{otherwise}.
        \end{cases}

    The relation is symmetric (up to ties) and commutative on the
    output dtype.

    Examples
    --------
    >>> import lucid
    >>> lucid.promote_types(lucid.int32, lucid.float32)
    lucid.float32
    >>> lucid.promote_types(lucid.int8, lucid.int64)
    lucid.int64
    """
    if a_dtype == b_dtype:
        return a_dtype
    return _promote(a_dtype, b_dtype)


def can_cast(from_dtype: _DType, to_dtype: _DType) -> bool:
    r"""Predicate: can ``from_dtype`` be safely cast to ``to_dtype``?

    Returns ``True`` iff every value representable in ``from_dtype`` is
    also representable in ``to_dtype`` without loss of range or
    precision.  This is the "safe" casting policy of NumPy: it admits
    widening conversions (e.g. ``int8 → int32``, ``float32 → float64``)
    and rejects narrowing or sign-changing conversions.

    Parameters
    ----------
    from_dtype : dtype-like
        Source dtype.
    to_dtype : dtype-like
        Destination dtype.

    Returns
    -------
    bool
        ``True`` if the cast is safe, ``False`` otherwise.

    Notes
    -----
    Implemented in terms of :func:`promote_types`: the cast is safe when

    .. math::

        \operatorname{promote\_types}(\text{from},\;\text{to})
        \;=\; \text{to},

    i.e. ``to_dtype`` already dominates ``from_dtype`` in the
    promotion lattice.  Contrast with :func:`result_type`, which
    returns the *target* dtype for a mixed-type expression rather than
    a boolean — use ``can_cast`` for pre-flight checks, ``result_type``
    for selecting an output dtype.

    Examples
    --------
    >>> import lucid
    >>> lucid.can_cast(lucid.int8, lucid.int32)
    True
    >>> lucid.can_cast(lucid.float32, lucid.int32)
    False
    >>> lucid.can_cast(lucid.float32, lucid.float64)
    True
    """
    return promote_types(from_dtype, to_dtype) == to_dtype


__all__ = ["result_type", "promote_types", "can_cast"]
