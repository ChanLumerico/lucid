"""Core ML's vocabulary: dtypes, compute units, precisions, targets.

Everything Core ML numbers or names lives here rather than being spelled
inline, so a specification bump is a change in one file.  The values that
end up in the emitted bytes come from the engine (which got them from the
generated schema header); what this module adds is the Lucid-facing names
and the mapping from Lucid's own dtypes.
"""

import enum
from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

__all__ = ["ComputeUnits", "Precision", "WeightPrecision"]


class ComputeUnits(enum.Enum):
    """Processors Core ML may schedule the model on.

    ``CPU_AND_NE`` is the reason this package exists: it is the only way
    to reach the Neural Engine, which neither of Lucid's own backends
    (Accelerate, MLX) targets.
    """

    ALL = "ALL"
    CPU_ONLY = "CPU_ONLY"
    CPU_AND_GPU = "CPU_AND_GPU"
    CPU_AND_NE = "CPU_AND_NE"


class Precision(enum.Enum):
    """Weight and activation precision of the exported program.

    ``FLOAT32`` is the default because an export should agree with the
    model it came from; Core ML's own default is ``FLOAT16``, which the
    Neural Engine wants and which costs roughly 1e-4 against the eager
    model.
    """

    FLOAT32 = "FLOAT32"
    FLOAT16 = "FLOAT16"


class WeightPrecision(enum.Enum):
    """How a weight is stored, as distinct from how the body computes.

    ``INT8`` stores each weight as an integer code plus one scale per
    output channel, and Core ML dequantizes it on the way into the
    operation that uses it. The arithmetic still runs at the body's
    precision — this is a storage decision, not a different network — so
    the package halves in size against float16 and the Neural Engine
    moves half as much memory to do the same work.

    The cost is real and one-directional: eight bits per weight cannot
    represent what sixteen did, so a quantized export is further from the
    eager model than a float16 one. ``verify`` will say by how much.
    """

    FLOAT = "FLOAT"
    INT8 = "INT8"


# Lucid dtype -> (MIL dtype, blob dtype).  The two numberings disagree —
# MIL calls float32 11, the blob calls it 2 — so both are carried rather
# than derived from each other.
_DTYPES: dict[object, tuple[int, int]] = {
    lucid.float32: (_C_engine.coreml.DTYPE_FLOAT32, _C_engine.coreml.BLOB_FLOAT32),
    lucid.float16: (_C_engine.coreml.DTYPE_FLOAT16, _C_engine.coreml.BLOB_FLOAT16),
    # Integer inputs (token ids, masks) reach the interface but never the
    # weight blob, so they have a MIL type and no blob type.
    lucid.int64: (_C_engine.coreml.DTYPE_INT32, -1),
    lucid.int32: (_C_engine.coreml.DTYPE_INT32, -1),
}

INT8 = _C_engine.coreml.DTYPE_INT8
BLOB_INT8 = _C_engine.coreml.BLOB_INT8
INT32 = _C_engine.coreml.DTYPE_INT32
BOOL = _C_engine.coreml.DTYPE_BOOL
STRING = _C_engine.coreml.DTYPE_STRING
FLOAT32 = _C_engine.coreml.DTYPE_FLOAT32


def mil_dtype(dtype: object) -> int:
    """MIL element-type number for a Lucid dtype.

    Raises
    ------
    TypeError
        Core ML has no storage for this dtype. ``float64`` is the common
        case and it is genuine — the format has no double.
    """
    entry = _DTYPES.get(dtype)
    if entry is None:
        raise TypeError(
            f"lucid.coreml: dtype {dtype} has no Core ML equivalent "
            f"(supported: {', '.join(str(d) for d in _DTYPES)})"
        )
    return entry[0]


def blob_dtype(dtype: object) -> int:
    """Blob-file element type for a Lucid dtype.

    Not the same numbering as :func:`mil_dtype`: the weight blob calls
    float32 ``2`` while the program calls it ``11``. Both are carried
    explicitly rather than derived from one another.

    Parameters
    ----------
    dtype : lucid dtype
        Element type to translate.

    Returns
    -------
    int
        Blob-format dtype code.

    Raises
    ------
    TypeError
        No Core ML storage exists for this dtype.
    """
    entry = _DTYPES.get(dtype)
    if entry is None:
        raise TypeError(
            f"lucid.coreml: dtype {dtype} cannot be stored in a Core ML weight blob"
        )
    return entry[1]


# MIL spells the cast target as a string, not the numeric dtype.
CAST_NAMES = {FLOAT32: "fp32", _C_engine.coreml.DTYPE_FLOAT16: "fp16"}

FLOAT16 = _C_engine.coreml.DTYPE_FLOAT16
BLOB_FLOAT16 = _C_engine.coreml.BLOB_FLOAT16
BLOB_FLOAT32 = _C_engine.coreml.BLOB_FLOAT32


def body_dtypes(precision: Precision) -> tuple[int, int]:
    """Element types the program body computes in.

    Parameters
    ----------
    precision : Precision
        Requested body precision. ``FLOAT16`` is what the Neural Engine
        runs; ``FLOAT32`` keeps the export faithful to the source model.

    Returns
    -------
    tuple[int, int]
        ``(MIL dtype, blob dtype)`` — the two numberings disagree.
    """
    if precision is Precision.FLOAT16:
        return FLOAT16, BLOB_FLOAT16
    return FLOAT32, BLOB_FLOAT32


def trace_dtype(name: str, body_mil: int) -> int:
    """MIL element type for a value the tracer produced.

    Float results follow the program body — an fp16 export wants fp16
    intermediates — but integer and boolean ones must not. Declaring an
    index tensor as float is not a rounding difference; MIL rejects the
    program, and where it does not the operation reads the wrong values.

    Parameters
    ----------
    name : str
        Lucid dtype name as the trace spells it (``"F32"``, ``"I64"``…).
    body_mil : int
        MIL dtype the body computes in, used for float results.

    Returns
    -------
    int
        MIL element-type number.

    Raises
    ------
    TypeError
        Core ML has no equivalent — ``float64`` is the reachable case.
    """
    if name in ("F32", "F16"):
        return body_mil
    if name in ("I64", "I32", "I16", "I8"):
        # Core ML's program dialect works in int32; a vocabulary index
        # never approaches its range.
        return INT32
    if name == "Bool":
        return BOOL
    raise TypeError(f"lucid.coreml: no Core ML element type for a {name} intermediate")


def type_spec(tensor: Tensor) -> tuple[int, list[int]]:
    """Describe a tensor the way the engine binding expects.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose type and shape describe a MIL value.

    Returns
    -------
    tuple[int, list[int]]
        ``(MIL dtype, shape)``. An empty shape means a scalar.
    """
    return (mil_dtype(tensor.dtype), [int(d) for d in tensor.shape])
