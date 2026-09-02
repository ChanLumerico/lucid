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

from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    from lucid.coreml._build import Builder

__all__ = ["EMITTERS", "MIL_OPS", "MultiOutput"]

# Lucid dtype name -> the spelling MIL's ``cast`` expects.
_CAST_TARGETS = {
    "F32": "fp32",
    "F16": "fp16",
    "I64": "int32",
    "I32": "int32",
    "Bool": "bool",
}


class MultiOutput(NamedTuple):
    """One MIL operation that produces every one of a Lucid op's outputs.

    ``split`` is the case: the trace records one op with several
    results, and ``Operation.outputs`` is repeated to match.
    """

    mil_type: str
    bindings: list


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


@_emitter("mean")
def _mean(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    attrs = op.attrs  # type: ignore[attr-defined]
    axes = [int(d) for d in attrs["dims"]]
    return "reduce_mean", [
        ("x", ins[0]),
        ("axes", b.const_ints(axes)),
        ("keep_dims", b.const_bool(bool(attrs.get("keepdim", False)))),
    ]


@_emitter("stack")
def _stack(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    axis = int(op.attrs.get("axis", 0))  # type: ignore[attr-defined]
    return "stack", [("values", list(ins)), ("axis", b.const_int(axis))]


@_emitter("split_at")
def _split_at(b: "Builder", op: object, ins: list[str]) -> MultiOutput:
    """One MIL ``split`` producing every section the trace recorded.

    Lucid records the cut points; MIL wants the section sizes, which the
    output shapes already give — and taking them from the outputs keeps
    the two descriptions from disagreeing.
    """
    axis = int(op.attrs["axis"])  # type: ignore[attr-defined]
    sizes = [int(o.shape[axis]) for o in op.outputs]  # type: ignore[attr-defined]
    return MultiOutput(
        "split",
        [
            ("x", ins[0]),
            ("split_sizes", b.const_ints(sizes)),
            ("axis", b.const_int(axis)),
        ],
    )


@_emitter("broadcast_to")
def _broadcast_to(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    """Expressed as ``tile``, which MIL has, since ``broadcast_to`` it does not.

    A broadcast repeats each size-1 axis; the repetition counts come from
    dividing the result shape by the operand's. Ranks are matched first by
    reshaping with leading ones, the way broadcasting aligns them.
    """
    src = b.shape_of(ins[0])
    out = _out_shape(op)
    value = ins[0]
    if len(src) < len(out):
        src = [1] * (len(out) - len(src)) + src
        value = b.emit("reshape", [("x", ins[0]), ("shape", b.const_ints(src))], src)
    reps = [o // s if s else 1 for s, o in zip(src, out)]
    return "tile", [("x", value), ("reps", b.const_ints(reps))]


@_emitter("scaled_dot_product_attention")
def _sdpa(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    """Decomposed rather than mapped to ``ios18.scaled_dot_product_attention``.

    The fused operation exists only in a newer opset than the one this
    writer emits, and the decomposition is what Core ML's own converter
    produces for older targets: scores, scale, softmax, weighted sum.
    """
    attrs = op.attrs  # type: ignore[attr-defined]
    if attrs.get("has_mask") and len(ins) < 4:
        raise NotImplementedError(
            "lucid.coreml: scaled_dot_product_attention declares a mask but the "
            "trace carries no mask operand"
        )
    query, key, value = ins[0], ins[1], ins[2]
    q_shape = b.shape_of(query)
    k_shape = b.shape_of(key)
    scores_shape = q_shape[:-1] + [k_shape[-2]]

    scores = b.emit(
        "matmul",
        [
            ("x", query),
            ("y", key),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(True)),
        ],
        scores_shape,
    )
    scaled = b.emit(
        "mul",
        [("x", scores), ("y", b.const_float(float(attrs["scale"])))],
        scores_shape,
    )
    if attrs.get("has_mask"):
        # The mask is already an additive float tensor in the trace — it
        # broadcasts over the head axis — so it goes straight onto the
        # scores rather than being rebuilt here.
        scaled = b.emit("add", [("x", scaled), ("y", ins[3])], scores_shape)
    if attrs.get("is_causal"):
        # A causal mask is a constant here: the trace fixed the sequence
        # length, so the forbidden positions are known.  ``-1e4`` rather
        # than ``-inf`` because the softmax may run in float16, where
        # ``inf - inf`` is a NaN rather than a zero weight.
        import lucid

        rows, cols = scores_shape[-2], scores_shape[-1]
        offset = cols - rows
        mask = lucid.tensor(
            [
                [0.0 if j <= i + offset else -1.0e4 for j in range(cols)]
                for i in range(rows)
            ]
        )
        scaled = b.emit(
            "add", [("x", scaled), ("y", b.const_from_tensor(mask))], scores_shape
        )
    weights = b.emit(
        "softmax", [("x", scaled), ("axis", b.const_int(-1))], scores_shape
    )
    return "matmul", [
        ("x", weights),
        ("y", value),
        ("transpose_x", b.const_bool(False)),
        ("transpose_y", b.const_bool(False)),
    ]


# ── values produced out of nothing ───────────────────────────────────
#
# These take no operands: the trace already fixed their contents, so each
# becomes a constant that an ``identity`` names as the op's output.


@_emitter("zeros")
def _zeros(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    import lucid

    return "identity", [("x", b.const_from_tensor(lucid.zeros(*_out_shape(op))))]


@_emitter("full")
def _full(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    import lucid

    value = float(op.attrs.get("fill_value", 0.0))  # type: ignore[attr-defined]
    shape = _out_shape(op)
    filled = lucid.zeros(*shape) + value if shape else lucid.tensor(value)
    return "identity", [("x", b.const_from_tensor(filled))]


@_emitter("arange")
def _arange(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    import lucid

    attrs = op.attrs  # type: ignore[attr-defined]
    start = float(attrs.get("start", 0.0))
    step = float(attrs.get("step", 1.0))
    count = int(_out_shape(op)[0])
    values = lucid.tensor([start + step * i for i in range(count)])
    return "identity", [("x", b.const_from_tensor(values))]


# ── indexing, casting, reductions ────────────────────────────────────


@_emitter("embedding")
def _embedding(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    """A row lookup, which MIL spells ``gather`` along the table's first axis.

    ``padding_idx`` is not honoured here: it only matters to the backward
    pass, and an exported graph has none.
    """
    table, indices = ins[0], ins[1]
    return "gather", [
        ("x", table),
        ("indices", indices),
        ("axis", b.const_int(0)),
        # Required by this opset.  False: the trace's indices are already
        # in range, and checking them would cost a comparison per lookup.
        ("validate_indices", b.const_bool(False)),
    ]


@_emitter("astype")
def _astype(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    name = str(op.outputs[0].dtype).split(".")[-1]  # type: ignore[attr-defined]
    target = _CAST_TARGETS.get(name)
    if target is None:
        raise NotImplementedError(f"lucid.coreml: no Core ML cast target for {name}")
    return "cast", [("x", ins[0]), ("dtype", b.const_str(target))]


@_emitter("max")
def _max(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return _reduce(b, op, ins, "reduce_max")


@_emitter("min")
def _min(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    return _reduce(b, op, ins, "reduce_min")


def _reduce(
    b: "Builder", op: object, ins: list[str], mil_type: str
) -> tuple[str, list[tuple[str, object]]]:
    attrs = op.attrs  # type: ignore[attr-defined]
    return mil_type, [
        ("x", ins[0]),
        ("axes", b.const_ints([int(d) for d in attrs["dims"]])),
        ("keep_dims", b.const_bool(bool(attrs.get("keepdim", False)))),
    ]


@_emitter("gelu")
def _gelu(
    b: "Builder", op: object, ins: list[str]
) -> tuple[str, list[tuple[str, object]]]:
    # Lucid's plain ``gelu`` is the tanh approximation; ``gelu_exact`` is
    # the erf form and maps to MIL's EXACT mode.
    return "gelu", [("x", ins[0]), ("mode", b.const_str("TANH_APPROXIMATION"))]


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
    "cast",
    "gather",
    "gelu",
    "reduce_max",
    "reduce_min",
    "layer_norm",
    "leaky_relu",
    "matmul",
    "reduce_mean",
    "silu",
    "split",
    "stack",
    "tile",
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
