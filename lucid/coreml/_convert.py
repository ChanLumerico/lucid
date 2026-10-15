"""Every conversion between element types a package makes (LCD-388).

MIL does not promote, so operands are converted explicitly — and two
conversions are not what they look like. A ``cast`` to or from bool is
folded wrongly by Core ML when its operand is constant, and the gathers
and scatters take no bool data at all. :func:`emit_cast` is the one place
a conversion is spelled, so neither reaches a package.
"""

from typing import TYPE_CHECKING

from lucid.coreml import _spec

if TYPE_CHECKING:
    from lucid.coreml._build import Builder
    from lucid.coreml._emit import Bindings, EmitResult, TracedOp

__all__ = [
    "MIL_CAST_NAMES",
    "emit_cast",
    "emit_opmath_binary",
    "moved",
    "scalar_like",
    "widened",
]

_MIL_BOOL = _spec.BOOL
_MIL_INT32 = _spec.INT32

#: MIL's spelling of each element type a conversion can land on — the
#: ``out_dtype`` :func:`emit_cast` takes.
MIL_CAST_NAMES = {
    _spec.FLOAT16: "fp16",
    _spec.FLOAT32: "fp32",
    _MIL_INT32: "int32",
    _MIL_BOOL: "bool",
}


def scalar_like(b: Builder, value: str, number: int) -> str:
    """A scalar ``number`` of ``value``'s type: MIL refuses int32 beside a float.

    Parameters
    ----------
    b : Builder
        Builder minting the constant.
    value : str
        The operand the scalar will meet.
    number : int
        The scalar.

    Returns
    -------
    str
        The constant's name: int32 beside an int32 or bool value, otherwise
        a float at the body's precision.
    """
    if b.dtype_of(value) in (_MIL_INT32, _MIL_BOOL):
        return b.const_int(number)
    return b.const_float(float(number))


def emit_cast(b: Builder, value: str, out_dtype: str) -> EmitResult:
    """``value`` converted to ``out_dtype``: every conversion the program makes.

    The driver's own casts bracket an fp16 body, which still presents fp32
    inputs and outputs; ``astype`` and the operations that need another
    operand type make the rest.

    A conversion to or from bool is not a MIL ``cast``. Core ML folds such
    a cast wrongly when its operand is constant — built from the model's
    own constants, as a causal mask is: ``ones(5, 5).triu(1).bool()`` came
    back all false in a float16 package, and a float32 package lost the
    mask once a scatter cast it on to int32 (LCD-388). ``not_equal`` and
    ``select`` say the same thing and are folded correctly.

    Parameters
    ----------
    b : Builder
        Builder minting the constants.
    value : str
        Name of the value to convert.
    out_dtype : str
        MIL's spelling of the target type — ``"fp16"``, ``"fp32"``,
        ``"int32"`` or ``"bool"``.

    Returns
    -------
    tuple[str, list]
        The last operation, for the caller to append and declare.
    """
    from_bool = b.dtype_of(value) == _MIL_BOOL
    if out_dtype == "bool" and not from_bool:
        return "not_equal", [("x", value), ("y", scalar_like(b, value, 0))]
    if from_bool and out_dtype != "bool":
        picked: Bindings = [
            ("cond", value),
            ("a", b.const_int(1)),
            ("b", b.const_int(0)),
        ]
        if out_dtype == "int32":
            return "select", picked
        # On through int32, whose cast to a float is not a bool cast.
        value = b.emit("select", picked, b.shape_of(value), dtype=_MIL_INT32)
    return "cast", [("x", value), ("dtype", b.const_str(out_dtype))]


def widened(b: Builder, value: str) -> str:
    """``value`` as int32 1 / 0 when it is bool, else itself.

    MIL's gathers and scatters take no bool data: the package fails to
    parse, naming the operand (LCD-388).  A bool tensor is moved as int32
    and turned back into bool afterwards, which 1 and 0 survive exactly.

    Parameters
    ----------
    b : Builder
        Builder appending the conversion.
    value : str
        The operand.

    Returns
    -------
    str
        ``value`` itself, or an int32 copy of it.
    """
    if b.dtype_of(value) != _MIL_BOOL:
        return value
    return b.emit(*emit_cast(b, value, "int32"), b.shape_of(value), dtype=_MIL_INT32)


def moved(
    b: Builder, op: TracedOp, mil_type: str, bindings: Bindings, data: tuple[str, ...]
) -> EmitResult:
    """``mil_type`` with its ``data`` parameters widened when the result is bool.

    Parameters
    ----------
    b : Builder
        Builder appending the int32 operation.
    op : TracedOp
        The traced gather or scatter; its declared result type decides.
    mil_type : str
        The MIL operation.
    bindings : Bindings
        Its operands, as an emitter returns them.
    data : tuple of str
        The parameters that carry the tensor being moved.

    Returns
    -------
    tuple[str, list]
        The operation itself for a non-bool result; otherwise the
        conversion of its int32 result back to bool.
    """
    if b.result_mil_dtype(op) != _MIL_BOOL:
        return mil_type, bindings
    operands: Bindings = [
        (
            param,
            widened(b, value) if param in data and isinstance(value, str) else value,
        )
        for param, value in bindings
    ]
    result = b.emit(mil_type, operands, b.result_dims(op), dtype=_MIL_INT32)
    return emit_cast(b, result, "bool")


def emit_opmath_binary(
    b: Builder,
    op: TracedOp,
    lucid_name: str,
    mil_name: str,
    lhs: str,
    rhs: str,
    target: int | None,
) -> EmitResult | None:
    """Upcast a float16 + 0-d float32 ``mul``/``div``/``floordiv`` pair to fp32.

    Matches eager op-math promotion by casting the half input to float32,
    emitting the binary op in float32, and casting the result back to fp16.

    Parameters
    ----------
    b : Builder
        Active MIL builder.
    op : TracedOp
        Traced binary operation.
    lucid_name : str
        Lucid operation name.
    mil_name : str
        MIL binary operation name.
    lhs : str
        Left-hand MIL value name.
    rhs : str
        Right-hand MIL value name.
    target : int or None
        Target MIL output data type.

    Returns
    -------
    EmitResult or None
        Cast-back MIL result name if op-math promotion applies, else ``None``.
    """
    if target != _spec.FLOAT16 or lucid_name not in ("mul", "div", "floordiv"):
        return None
    is_pair = (b.dtype_of(lhs) == _spec.FLOAT16 and b.is_0d_float32(rhs)) or (
        b.dtype_of(rhs) == _spec.FLOAT16 and b.is_0d_float32(lhs)
    )
    if not is_pair:
        return None
    x_f32, y_f32 = b.agree_on_dtype([lhs, rhs], _spec.FLOAT32)
    res_f32 = b.emit(
        mil_name,
        [("x", x_f32), ("y", y_f32)],
        b.result_dims(op),
        dtype=_spec.FLOAT32,
    )
    return emit_cast(b, res_f32, "fp16")
