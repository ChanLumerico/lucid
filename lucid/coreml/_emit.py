"""Lucid trace op → Core ML MIL operation.

One entry per Lucid op name.  An emitter receives a builder context (for
minting the constants MIL requires every scalar operand to be) and the
already-emitted value names of the op's inputs, and returns the MIL op
type plus its ``(parameter, value name)`` bindings.  The driver in
:mod:`lucid.coreml._build` supplies the output name and type from the
trace, so an emitter never has to reason about shapes.

Operand order is Lucid's, and it is not the order habit suggests.  The
table below was read off a trace rather than assumed:

    conv2d           (x, weight, bias)
    linear           (x, weight, bias)
    batch_norm_eval  (x, running_mean, running_var, weight, bias)

The third line is the dangerous one: the statistics come *before* the
affine parameters.  An emitter written from the usual
``(gamma, beta, mean, var)`` habit yields a model with correct shapes,
correct output ranks, and wrong values everywhere — nothing about it
fails loudly.
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from lucid.coreml._build import Builder

__all__ = ["EMITTERS", "MIL_OPS"]

# Lucid op name -> emitter.
EMITTERS: dict[str, Callable[..., tuple[str, list[tuple[str, str]]]]] = {}


def _emitter(name: str) -> Callable[..., object]:
    def register(fn: object) -> object:
        EMITTERS[name] = fn  # type: ignore[assignment]
        return fn

    return register


def _pair(value: object) -> list[int]:
    return [int(value[0]), int(value[1])]  # type: ignore[index]


def _pad4(padding: object) -> list[int]:
    """Lucid's ``[pad_h, pad_w]`` as MIL's ``[top, bottom, left, right]``."""
    ph, pw = int(padding[0]), int(padding[1])  # type: ignore[index]
    return [ph, ph, pw, pw]


def _flag(value: object, default: bool = False) -> bool:
    # Lucid records these as single-element int lists in the trace.
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


@_emitter("conv2d")
def _conv2d(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    attrs = op.attrs  # type: ignore[attr-defined]
    bindings = [("x", ins[0]), ("weight", ins[1])]
    if len(ins) > 2:
        bindings.append(("bias", ins[2]))
    bindings += [
        ("strides", b.const_ints(_pair(attrs["stride"]))),
        ("pad_type", b.const_str("custom")),
        ("pad", b.const_ints(_pad4(attrs["padding"]))),
        ("dilations", b.const_ints(_pair(attrs["dilation"]))),
        ("groups", b.const_int(int(attrs["groups"]))),
    ]
    return "conv", bindings


@_emitter("linear")
def _linear(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    bindings = [("x", ins[0]), ("weight", ins[1])]
    if len(ins) > 2:
        bindings.append(("bias", ins[2]))
    return "linear", bindings


@_emitter("batch_norm_eval")
def _batch_norm_eval(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    x, mean, variance, gamma, beta = ins[0], ins[1], ins[2], ins[3], ins[4]
    return "batch_norm", [
        ("x", x),
        ("mean", mean),
        ("variance", variance),
        ("gamma", gamma),
        ("beta", beta),
        ("epsilon", b.const_float(float(op.attrs["eps"]))),  # type: ignore[attr-defined]
    ]


@_emitter("relu")
def _relu(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    return "relu", [("x", ins[0])]


@_emitter("relu6")
def _relu6(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    return "relu6", [("x", ins[0])]


@_emitter("sigmoid")
def _sigmoid(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    return "sigmoid", [("x", ins[0])]


@_emitter("tanh")
def _tanh(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    return "tanh", [("x", ins[0])]


@_emitter("add")
def _add(b: "Builder", op: object, ins: list[str]) -> tuple[str, list[tuple[str, str]]]:
    return "add", [("x", ins[0]), ("y", ins[1])]


@_emitter("sub")
def _sub(b: "Builder", op: object, ins: list[str]) -> tuple[str, list[tuple[str, str]]]:
    return "sub", [("x", ins[0]), ("y", ins[1])]


@_emitter("mul")
def _mul(b: "Builder", op: object, ins: list[str]) -> tuple[str, list[tuple[str, str]]]:
    return "mul", [("x", ins[0]), ("y", ins[1])]


@_emitter("div")
def _div(b: "Builder", op: object, ins: list[str]) -> tuple[str, list[tuple[str, str]]]:
    return "real_div", [("x", ins[0]), ("y", ins[1])]


@_emitter("reshape")
def _reshape(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    # Lucid keeps the target shape on the result, not in the attributes.
    shape = [int(d) for d in op.outputs[0].shape]  # type: ignore[attr-defined]
    return "reshape", [("x", ins[0]), ("shape", b.const_ints(shape))]


@_emitter("max_pool2d")
def _max_pool2d(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    attrs = op.attrs  # type: ignore[attr-defined]
    return "max_pool", [
        ("x", ins[0]),
        ("kernel_sizes", b.const_ints(_pair(attrs["kernel_size"]))),
        ("strides", b.const_ints(_pair(attrs["stride"]))),
        ("pad_type", b.const_str("custom")),
        ("pad", b.const_ints(_pad4(attrs["padding"]))),
        ("ceil_mode", b.const_bool(_flag(attrs.get("ceil_mode")))),
    ]


@_emitter("avg_pool2d")
def _avg_pool2d(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    attrs = op.attrs  # type: ignore[attr-defined]
    return "avg_pool", [
        ("x", ins[0]),
        ("kernel_sizes", b.const_ints(_pair(attrs["kernel_size"]))),
        ("strides", b.const_ints(_pair(attrs["stride"]))),
        ("pad_type", b.const_str("custom")),
        ("pad", b.const_ints(_pad4(attrs["padding"]))),
        (
            "exclude_padding_from_average",
            b.const_bool(not _flag(attrs.get("count_include_pad"), default=True)),
        ),
        ("ceil_mode", b.const_bool(_flag(attrs.get("ceil_mode")))),
    ]


@_emitter("dropout")
def _dropout(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    # An exported graph is an inference graph.  Lucid still records the op
    # under ``eval()``; a training-mode one is refused rather than
    # silently turned into an identity the caller did not ask for.
    if op.attrs.get("training"):  # type: ignore[attr-defined]
        raise NotImplementedError(
            "lucid.coreml: dropout was traced in training mode — call model.eval() "
            "before exporting, or the exported graph would differ from the traced one"
        )
    return "identity", [("x", ins[0])]


def _out_shape(op: object) -> list[int]:
    return [int(d) for d in op.outputs[0].shape]  # type: ignore[attr-defined]


# ── shape ops ────────────────────────────────────────────────────────
#
# ``squeeze`` / ``unsqueeze`` / ``contiguous`` all become a ``reshape`` to
# the shape the trace already recorded.  Lucid materialises every view, so
# the result shape is static and a reshape expresses each of them exactly,
# which avoids carrying axis bookkeeping that could disagree with the trace.


@_emitter("squeeze")
def _squeeze(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "reshape", [("x", ins[0]), ("shape", b.const_ints(_out_shape(op)))]


@_emitter("unsqueeze")
def _unsqueeze(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "reshape", [("x", ins[0]), ("shape", b.const_ints(_out_shape(op)))]


@_emitter("contiguous")
def _contiguous(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "identity", [("x", ins[0])]


@_emitter("permute")
def _permute(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    perm = [int(a) for a in op.attrs["permutation"]]  # type: ignore[attr-defined]
    return "transpose", [("x", ins[0]), ("perm", b.const_ints(perm))]


@_emitter("concatenate")
def _concatenate(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    axis = int(op.attrs.get("dim", 0))  # type: ignore[attr-defined]
    # ``values`` is variadic: one parameter bound to every input.
    return "concat", [
        ("values", list(ins)),
        ("axis", b.const_int(axis)),
        ("interleave", b.const_bool(False)),
    ]


# ── activations and arithmetic ───────────────────────────────────────


@_emitter("silu")
def _silu(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "silu", [("x", ins[0])]


@_emitter("gelu_exact")
def _gelu_exact(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    # Lucid's exact GELU is the erf form, which MIL calls EXACT; MIL's
    # default is the tanh approximation, a different function.
    return "gelu", [("x", ins[0]), ("mode", b.const_str("EXACT"))]


@_emitter("leaky_relu")
def _leaky_relu(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    alpha = float(op.attrs.get("negative_slope", 0.01))  # type: ignore[attr-defined]
    return "leaky_relu", [("x", ins[0]), ("alpha", b.const_float(alpha))]


@_emitter("softmax")
def _softmax(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    axis = int(op.attrs.get("axis", -1))  # type: ignore[attr-defined]
    return "softmax", [("x", ins[0]), ("axis", b.const_int(axis))]


@_emitter("exp")
def _exp(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "exp", [("x", ins[0])]


@_emitter("matmul")
def _matmul(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return "matmul", [
        ("x", ins[0]),
        ("y", ins[1]),
        ("transpose_x", b.const_bool(False)),
        ("transpose_y", b.const_bool(False)),
    ]


@_emitter("layer_norm")
def _layer_norm(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    """Normalised over the trailing axes the weight covers.

    Lucid's trace does not record which axes were normalised, but the
    affine weight's rank determines them: it broadcasts over exactly the
    normalised tail. MIL wants them as explicit (negative) axis indices.
    """
    x, weight = ins[0], ins[1]
    rank = len(b.shape_of(weight))
    axes = list(range(-rank, 0))
    bindings: list[tuple[str, object]] = [
        ("x", x),
        ("axes", b.const_ints(axes)),
        ("gamma", weight),
    ]
    if len(ins) > 2:
        bindings.append(("beta", ins[2]))
    bindings.append(("epsilon", b.const_float(float(op.attrs["eps"]))))  # type: ignore[attr-defined]
    return "layer_norm", bindings


def emit_cast(
    b: "Builder", value: str, out_dtype: str
) -> tuple[str, list[tuple[str, str]]]:
    """A ``cast`` the driver inserts, not one a Lucid op asks for.

    An fp16 program still presents fp32 inputs and outputs, so the body is
    bracketed by casts rather than forcing callers to hand over half
    precision. These two casts are also the only operations that stay on
    the CPU when the rest of the model runs on the Neural Engine.

    Parameters
    ----------
    b : Builder
        Builder minting the dtype constant.
    value : str
        Name of the value to cast.
    out_dtype : str
        MIL's spelling of the target type — ``"fp16"`` or ``"fp32"``.

    Returns
    -------
    tuple[str, list]
        ``("cast", bindings)`` for the driver to append.
    """
    return "cast", [("x", value), ("dtype", b.const_str(out_dtype))]


@_emitter("identity")
def _identity(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, str]]]:
    return "identity", [("x", ins[0])]


# Names of the MIL ops this package can produce, for diagnostics.
MIL_OPS = (
    "add",
    "concat",
    "exp",
    "gelu",
    "layer_norm",
    "leaky_relu",
    "matmul",
    "silu",
    "softmax",
    "transpose",
    "avg_pool",
    "batch_norm",
    "conv",
    "identity",
    "linear",
    "max_pool",
    "mul",
    "real_div",
    "relu",
    "relu6",
    "reshape",
    "sigmoid",
    "sub",
    "tanh",
)
