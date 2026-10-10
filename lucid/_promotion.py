"""The one owner of binary dtype promotion.

Every spelling of a binary op — ``a + b``, ``lucid.add``, ``Tensor.add``,
the in-place forms, the comparisons, the bitwise and shift operators,
``where`` / ``cat`` / ``einsum``, and the public ``result_type`` /
``promote_types`` / ``can_cast`` — takes its result dtype from here.  This is
the only module under ``lucid/`` allowed to hold a dtype kind/width table;
``test_promotion_single_source`` fails if another one appears.

The rule is the reference framework's.  Operands fall into three tiers:

1. tensors with at least one dimension,
2. 0-d tensors,
3. Python numbers.

Within a tier the dtypes meet on the lattice :func:`promote_types`.  Across
tiers a lower tier can raise the result's *category* (bool < integer <
floating < complex) but never its width inside a category, so
``float16_tensor * tensor(0.5)`` stays float16 and ``int8_tensor + 1`` stays
int8, while ``int32_tensor + 2.5`` becomes floating.

A Python ``bool`` is bool, an ``int`` is int64, a ``float`` is the default
dtype and a ``complex`` the complex dtype of the default (complex64, or
complex128 under a float64 default).  Off the CPU stream, which cannot hold
float64, a float64 default yields float32 / complex64 instead.

One documented deviation: the reference promotes float16 against a complex
operand to complex32, which Lucid lacks; Lucid answers complex64.

The module is a leaf — it imports only the engine and ``lucid._globals`` —
so the dispatch layer, the dunders and the op adapters
can all import it without a cycle.  It decides dtypes only; casting the
operands and building scalar operands is :mod:`lucid._dispatch`'s job.
"""

from lucid._C import engine as _C_engine
from lucid._globals import get_default_dtype

_D = _C_engine.Dtype

#: Category of each dtype; a larger category wins a mixed-category pair.
BOOL, INTEGER, FLOATING, COMPLEX = 0, 1, 2, 3

# (category, bit width) for every engine dtype — the single promotion table.
_KIND_WIDTH: dict[_C_engine.Dtype, tuple[int, int]] = {
    _D.Bool: (BOOL, 1),
    _D.I8: (INTEGER, 8),
    _D.I16: (INTEGER, 16),
    _D.I32: (INTEGER, 32),
    _D.I64: (INTEGER, 64),
    _D.F16: (FLOATING, 16),
    _D.BF16: (FLOATING, 16),
    _D.F32: (FLOATING, 32),
    _D.F64: (FLOATING, 64),
    _D.C64: (COMPLEX, 64),
    _D.C128: (COMPLEX, 128),
}

#: The half-precision floating dtypes, the ones mul/div/floordiv op-math is for.
HALF: frozenset[_C_engine.Dtype] = frozenset({_D.F16, _D.BF16})

#: A Python number, the third tier.
NUMBER = (bool, int, float, complex)


def bit_width(d: _C_engine.Dtype) -> int:
    """The width of one element of ``d`` in bits, read from the one table.

    Used to wrap a Python integer to an integer dtype's width exactly.
    """
    return _KIND_WIDTH[d][1]


def is_integral(d: _C_engine.Dtype) -> bool:
    """Whether ``d`` is bool or an integer dtype — the dtypes that hold no
    fraction, which true division moves to the default float."""
    return _KIND_WIDTH[d][0] <= INTEGER


def is_integer(d: _C_engine.Dtype) -> bool:
    """Whether ``d`` is an integer dtype, bool excluded — the dtypes a Python
    integer can be written into exactly."""
    return _KIND_WIDTH[d][0] == INTEGER


def _complex_of(d: _C_engine.Dtype) -> _C_engine.Dtype:
    """The complex dtype whose parts are the floating dtype ``d``."""
    return _D.C128 if d == _D.F64 else _D.C64


def promote_types(da: _C_engine.Dtype, db: _C_engine.Dtype) -> _C_engine.Dtype:
    """The lattice join of two dtypes, tiers aside.

    Parameters
    ----------
    da, db : lucid._C.engine.Dtype
        The two dtypes.

    Returns
    -------
    lucid._C.engine.Dtype
        The higher category's dtype; within one category the wider one.
        Two complex dtypes give the wider, a complex beside float64 gives
        complex128, and float16 beside bfloat16 — neither holds the other —
        gives float32.
    """
    if da == db:
        return da
    ka, wa = _KIND_WIDTH[da]
    kb, wb = _KIND_WIDTH[db]
    if ka == COMPLEX and kb != COMPLEX:
        return _D.C128 if db == _D.F64 else da
    if kb == COMPLEX and ka != COMPLEX:
        return _D.C128 if da == _D.F64 else db
    if ka != kb:
        return da if ka > kb else db
    if wa == wb:
        return _D.F32
    return da if wa > wb else db


def default_float(device: _C_engine.Device) -> _C_engine.Dtype:
    """The default floating dtype, narrowed to what ``device`` can hold.

    The GPU stream cannot hold float64, so a float64 default answers
    float32 there.
    """
    d = get_default_dtype()._engine
    if d == _D.F64 and device != _C_engine.Device.CPU:
        return _D.F32
    return d


def scalar_dtype(
    x: bool | int | float | complex, device: _C_engine.Device
) -> _C_engine.Dtype:
    """The dtype a Python number stands for beside tensors on ``device``.

    Parameters
    ----------
    x : bool or int or float or complex
        The number.
    device : lucid._C.engine.Device
        Where the tensors it meets live.

    Returns
    -------
    lucid._C.engine.Dtype
        bool for a bool, int64 for an int, :func:`default_float` for a
        float and its complex counterpart for a complex — the number's
        tier, before any tensor is seen.
    """
    if isinstance(x, bool):
        return _D.Bool
    if isinstance(x, int):
        return _D.I64
    if isinstance(x, float):
        return default_float(device)
    return _complex_of(default_float(device))


def _combine(
    higher: _C_engine.Dtype | None, lower: _C_engine.Dtype | None
) -> _C_engine.Dtype | None:
    """Fold a lower tier's dtype into a higher tier's.

    The lower tier can lift the category only: a complex lower makes a
    floating higher its complex counterpart; a floating lower meets an
    integral higher on the lattice, as does anything beside a bool higher.
    Otherwise the higher tier's dtype stands.
    """
    if higher is None or lower is None:
        return lower if higher is None else higher
    kh = _KIND_WIDTH[higher][0]
    kl = _KIND_WIDTH[lower][0]
    if kl == COMPLEX and kh != COMPLEX:
        return _complex_of(higher) if kh == FLOATING else lower
    if kh >= FLOATING or not (kh == BOOL or kl == FLOATING):
        return higher
    return promote_types(higher, lower)


Operand = _C_engine.TensorImpl | bool | int | float | complex


def result_type(*operands: Operand) -> _C_engine.Dtype:
    """The dtype a binary or n-ary op over ``operands`` computes in.

    Parameters
    ----------
    *operands : TensorImpl or bool or int or float or complex
        Engine tensors and Python numbers, in any order.  A tensor is read
        through ``dtype``, ``shape`` and ``device`` only, so shadow
        allocation's phantom impls classify as real ones do.

    Returns
    -------
    lucid._C.engine.Dtype
        ``combine(dimensioned, combine(zero_dim, numbers))``, each tier the
        lattice join of its members.

    Raises
    ------
    TypeError
        If no operand is given.
    """
    dim: _C_engine.Dtype | None = None
    zero: _C_engine.Dtype | None = None
    device: _C_engine.Device | None = None
    numbers: list[bool | int | float | complex] = []
    for x in operands:
        if isinstance(x, NUMBER):
            numbers.append(x)
            continue
        d = x.dtype
        if device is None:
            device = x.device
        if x.shape:
            dim = d if dim is None else promote_types(dim, d)
        else:
            zero = d if zero is None else promote_types(zero, d)
    wrapped: _C_engine.Dtype | None = None
    for n in numbers:
        d = scalar_dtype(n, device if device is not None else _C_engine.Device.CPU)
        wrapped = d if wrapped is None else promote_types(wrapped, d)
    out = _combine(dim, _combine(zero, wrapped))
    if out is None:
        raise TypeError("result_type: expected at least one operand")
    return out
