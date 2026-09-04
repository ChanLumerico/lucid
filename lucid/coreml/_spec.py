"""Core ML's vocabulary: dtypes, compute units, precisions, targets.

Everything Core ML numbers or names lives here rather than being spelled
inline, so a specification bump is a change in one file.  The values that
end up in the emitted bytes come from the engine (which got them from the
generated schema header); what this module adds is the Lucid-facing names
and the mapping from Lucid's own dtypes.
"""

import dataclasses
import enum
from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

__all__ = [
    "Classifier",
    "State",
    "ColorSpace",
    "ComputeUnits",
    "ImageInput",
    "Metadata",
    "Precision",
    "WeightPrecision",
]


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


@dataclasses.dataclass(frozen=True)
class Palettize:
    """Store each weight as an index into a small table of values.

    A layer's weights are clustered into ``2 ** bits`` representative
    values; the package keeps the table and one key per weight, and Core
    ML expands it on the way into the operation that uses it. At four
    bits that is a quarter of the space float16 takes and an eighth of
    float32, for weights that were never using their full range to begin
    with.

    Unlike ``WeightPrecision.INT8``, which spaces its levels evenly and
    spends them wherever the range happens to be, the table is fitted to
    the weights — so a layer whose values crowd around zero keeps its
    resolution there. The cost is the same in kind: fewer distinct values
    than the model was trained with, and ``verify`` will say how much.

    Attributes
    ----------
    bits : int
        One of ``1, 2, 3, 4, 6, 8`` — palette sizes ``2, 4, 8, 16, 64,
        256``. Anything else is refused rather than rounded, since the
        choice is a size/accuracy trade the caller is making
        deliberately.

        At eight bits, ``WeightPrecision.INT8`` is usually the better
        instrument: it stores the same byte per weight but carries a
        scale per output channel, and on a trained ResNet-50 it measured
        both smaller and closer to the model.
    """

    bits: int = 4

    def __post_init__(self) -> None:
        if self.bits not in (1, 2, 3, 4, 6, 8):
            raise ValueError(
                f"lucid.coreml: palettization takes 1, 2, 3, 4, 6 or 8 bits "
                f"(palette sizes 2, 4, 8, 16, 64, 256); got {self.bits}"
            )


@dataclasses.dataclass(frozen=True)
class Sparsify:
    """Keep only the largest weights, and a bit saying where they were.

    The smallest ``ratio`` of each weight tensor by magnitude is set to
    zero, and the package stores the survivors plus one bit per element.
    Below about half sparsity that costs more than it saves — the mask is
    an eighth of a byte per weight whether the weight survives or not —
    so the useful settings start around ``0.5`` and the saving grows from
    there.

    This is magnitude pruning applied at export, not training: the model
    is not fine-tuned afterwards, so accuracy falls faster than it would
    for a network pruned and then retrained. ``verify`` measures it.

    Attributes
    ----------
    ratio : float
        Fraction of each weight set to zero, in ``[0, 1)``.
    """

    ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError(
                f"lucid.coreml: sparsity is a fraction in [0, 1); got {self.ratio}"
            )


class ColorSpace(enum.Enum):
    """Pixel layout Core ML should hand the model.

    Names the order the model's own channel 0, 1, 2 mean, so that the
    runtime writes a pixel buffer the right way round. Getting this wrong
    is silent: the model runs and answers badly, which is the same
    failure as feeding it an image with the red and blue channels
    swapped, because that is exactly what it is.
    """

    GRAYSCALE = "GRAYSCALE"
    RGB = "RGB"
    BGR = "BGR"


@dataclasses.dataclass(frozen=True)
class ImageInput:
    """Present an input as an image, with the normalisation it expects.

    An app holding a ``CVPixelBuffer`` cannot feed a multi-array without
    converting the pixels itself, and getting that conversion subtly wrong
    — a missed scale, the wrong channel order — produces a model that runs
    and answers badly. Declaring the input as an image moves both the
    conversion and the normalisation into the package.

    ``scale`` and ``bias`` are applied as ``pixel * scale + bias``, which
    is where a mean-and-standard-deviation normalisation lands: a channel
    normalised by ``(p/255 - m) / s`` has ``scale = 1/(255 * s)`` and
    ``bias = -m / s``.

    Attributes
    ----------
    scale : float
        Multiplier applied to every pixel, before ``bias``.
    bias : tuple[float, ...]
        One offset per channel. Empty adds nothing.
    color : ColorSpace
        Pixel layout. ``GRAYSCALE`` expects one channel, the others three.
    """

    scale: float = 1.0
    bias: tuple[float, ...] = ()
    color: ColorSpace = ColorSpace.RGB


@dataclasses.dataclass(frozen=True)
class Classifier:
    """Turn the exported scores into labels Core ML knows how to name.

    Without this a package returns a score array and the app does its own
    argmax and label lookup. Vision's ``VNCoreMLRequest`` does not even
    get that far: it reads the package's ``predictedFeatureName``, and an
    unset one means it returns nothing.

    The two feature names default to what the reference tooling emits,
    because that is what Xcode's preview and most sample code look for.

    ``classify`` does **not** normalise: whatever the model produces is
    what the map contains, so a network ending in a linear layer yields
    raw scores under a name that says "probabilities". Add a softmax to
    the model if the values need to be probabilities — Core ML will not
    do it, and the name will not tell you it did not.

    Attributes
    ----------
    labels : tuple[str, ...]
        One label per score, in the order the model produces them.
    label_name : str
        Feature the winning label is returned under.
    probabilities_name : str
        Feature the label-to-probability map is returned under.
    """

    labels: tuple[str, ...]
    label_name: str = "classLabel"
    probabilities_name: str = "classLabel_probs"


@dataclasses.dataclass(frozen=True)
class State:
    """A value the package carries from one prediction to the next.

    Core ML keeps it: the caller neither passes it in nor gets it back,
    and each prediction sees what the last one wrote. A decoder's
    key-value cache is the case this exists for.

    Lucid's side has to be a pair — an input the model reads and an output
    it returns — rather than a buffer it mutates. The tracer records a
    pure graph, so an in-place buffer write does not appear in it at all;
    a package built from that would agree on the first call and stop
    accumulating on every one after, which ``export`` refuses.

    The state begins at zero. Nothing carries the example's values into
    it, so a model that needs a different starting point has to be given
    one through an ordinary input.

    Attributes
    ----------
    input : str
        Feature name of the input the state replaces.
    output : str
        Output field whose value is written back into it.
    """

    input: str
    output: str


@dataclasses.dataclass(frozen=True)
class Metadata:
    """What the package says about itself.

    Empty fields are left out of the description rather than written
    blank, so a package carries only what someone actually stated.
    """

    description: str = ""
    author: str = ""
    license: str = ""
    version: str = ""


_COLOR_SPACES: dict[ColorSpace, int] = {
    ColorSpace.GRAYSCALE: _C_engine.coreml.COLOR_GRAYSCALE,
    ColorSpace.RGB: _C_engine.coreml.COLOR_RGB,
    ColorSpace.BGR: _C_engine.coreml.COLOR_BGR,
}


def color_space(color: ColorSpace) -> int:
    """Engine constant for a colour space.

    Parameters
    ----------
    color : ColorSpace
        Layout to translate.

    Returns
    -------
    int
        The ``ImageFeatureType.ColorSpace`` value.
    """
    return _COLOR_SPACES[color]


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
BLOB_UINT8 = _C_engine.coreml.BLOB_UINT8
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
