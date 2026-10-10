"""Dtype-promotion helpers (``result_type``, ``promote_types``, ``can_cast``)."""

from typing import TYPE_CHECKING

from lucid import _promotion
from lucid._dtype import _ENGINE_TO_DTYPE, to_engine_dtype
from lucid._dtype import dtype as _DType
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def result_type(a: Tensor | Scalar, b: Tensor | Scalar) -> _DType:
    r"""Compute the dtype that a binary operation on the inputs would produce.

    The dtype ``a + b`` (and every other binary op) computes in: the same
    rule, from the same place, as the operators themselves.

    Parameters
    ----------
    a : Tensor | Scalar
        First operand — a tensor or a Python number.
    b : Tensor | Scalar
        Second operand. Same convention as ``a``.

    Returns
    -------
    DType
        The dtype that an arithmetic operation on ``a`` and ``b`` would
        produce.

    Notes
    -----
    Operands fall into three tiers: tensors with at least one dimension,
    0-d tensors, and Python numbers.  Each tier's dtypes meet on the
    :func:`promote_types` lattice, and a lower tier can lift the result's
    category (bool :math:`<` int :math:`<` float :math:`<` complex) but
    never its width within a category:

    .. math::

        \text{result\_type}(a, b) =
        \operatorname{combine}\bigl(\text{dim},\;
        \operatorname{combine}(\text{zero-dim},\; \text{number})\bigr)

    A Python ``bool`` counts as bool, an ``int`` as int64, a ``float`` as
    the default dtype and a ``complex`` as its complex counterpart.  Where
    the reference would answer complex32 (float16 beside a complex
    operand), Lucid answers complex64.

    See also :func:`promote_types` (operates directly on dtype objects).

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1, 2], dtype=lucid.int32)
    >>> b = lucid.tensor([1., 2.], dtype=lucid.float32)
    >>> lucid.result_type(a, b)
    lucid.float32
    >>> lucid.result_type(a, 2.5)
    lucid.float32
    >>> lucid.result_type(lucid.ones(2, dtype=lucid.float16), lucid.tensor(0.5))
    lucid.float16
    """
    da = a._impl if isinstance(a, Tensor) else a
    db = b._impl if isinstance(b, Tensor) else b
    return _ENGINE_TO_DTYPE[_promotion.result_type(da, db)]


def promote_types(a_dtype: _DType, b_dtype: _DType) -> _DType:
    r"""Compute the joint promotion of two dtypes.

    Operates on dtype objects directly (no tensor required). Combines
    the two dtypes according to the standard kind/width ordering:
    higher kind (bool :math:`<` int :math:`<` float :math:`<` complex)
    wins outright, and ties in kind are broken by the wider bit-width.
    A complex dtype beside float64 gives complex128, and float16 beside
    bfloat16 — neither holds the other — gives float32.

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
    return _ENGINE_TO_DTYPE[
        _promotion.promote_types(to_engine_dtype(a_dtype), to_engine_dtype(b_dtype))
    ]


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

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor  # noqa: E402,F811
