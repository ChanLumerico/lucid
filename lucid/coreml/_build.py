"""Trace a Lucid model and build the Core ML package it describes.

The tracer that :mod:`lucid.compile` uses to lower a forward pass into
MPSGraph produces a complete graph IR.  Core ML is a *second backend* for
that IR — MIL instead of MPSGraph — not a second front end, so nothing
here re-walks modules or re-derives shapes.

Two things the trace hands over that the driver has to sort out:

* **which external feed is the input.**  Everything a traced graph reads
  from outside is an "external feed": the model's parameters and buffers
  as well as the argument.  The argument is identified by object
  identity, the rest become constants.
* **which value is the output.**  Zoo models return an output dataclass
  (``ImageClassificationOutput`` and friends) rather than a tensor, so a
  bare ``isinstance`` check would reject every model in the zoo.
"""

from typing import TYPE_CHECKING, Any, NamedTuple

import dataclasses
import math
import struct

import lucid
import lucid.compile as _compile
import lucid.quantization as _quant
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.coreml import _spec
from lucid.coreml._emit import (
    EMITTERS,
    Bindings,
    Bound,
    Constant,
    MultiOutput,
    TracedOp,
    _as_float,
    _as_int,
    emit_cast,
)
from lucid.coreml._spec import (
    Classifier,
    State,
    ColorSpace,
    ImageInput,
    Metadata,
    Precision,
    WeightPrecision,
)

if TYPE_CHECKING:
    from lucid._C.engine import BlobWriter, MilProgram
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["Builder", "UnsupportedOp", "build_package", "trace"]


# Core ML's ML Program dialect refuses tensors above this rank.
_MAX_RANK = 5


class UnsupportedRank(NotImplementedError):
    """A tensor exceeds the rank Core ML's program dialect allows.

    Not something the writer can work around: the limit is the format's.
    Window-attention models hit it — Swin partitions into
    ``(B, H/w, w, W/w, w, C)``, which is rank six — and the honest answer
    is to say which operation and which shape rather than to reshape
    around it and hope the semantics survive.
    """

    def __init__(self, op_name: str, shape: tuple[int, ...]) -> None:
        super().__init__(
            f"lucid.coreml: op {op_name!r} produces a rank-{len(shape)} tensor "
            f"{shape}, and Core ML's program dialect allows at most rank "
            f"{_MAX_RANK}. This is a format limit, not a missing emitter."
        )
        self.op_name = op_name
        self.shape = shape


class UnsupportedOp(NotImplementedError):
    """A traced op has no MIL translation, or a refused one.

    Named rather than generic because the alternative failure — emitting
    a package that quietly lacks a layer — produces a model that loads
    and returns plausible numbers.

    Parameters
    ----------
    op_name : str
        Lucid operation that could not be emitted.
    reason : str or None, optional, default=None
        Why, when the answer is something other than "nobody has written
        the emitter yet". An operation Core ML translates *badly* — one
        whose package would load and be wrong — is refused deliberately,
        and pointing the reader at ``_emit.py`` to add a mapping would
        send them to write the very thing that was rejected.
    """

    def __init__(self, op_name: str, reason: str | None = None) -> None:
        super().__init__(
            f"lucid.coreml: {op_name} — {reason}"
            if reason is not None
            else (
                f"lucid.coreml: no Core ML translation for Lucid op "
                f"{op_name!r}. Mapped ops: {', '.join(sorted(EMITTERS))}. "
                f"Add an emitter in lucid/coreml/_emit.py."
            )
        )
        self.op_name = op_name
        self.reason = reason


def _select_outputs(output: object, field: str | None) -> list[tuple[str, Tensor]]:
    """Every tensor the model returns, named, in the order it declares them.

    Exporting one field of a several-field output and calling the result
    the model is the same failure as dropping a layer, one level up: a
    detector reduced to its class scores loads, runs, and returns
    plausible numbers with no boxes in them. So the default is all of
    them, and ``field`` is how a caller asks for less.
    """
    if isinstance(output, lucid.Tensor):
        return [("output", output)]
    if field is not None:
        picked = getattr(output, field, None)
        if not isinstance(picked, lucid.Tensor):
            raise TypeError(
                f"lucid.coreml: {type(output).__name__}.{field} is "
                f"{type(picked).__name__}, not a Tensor"
            )
        return [(field, picked)]

    named: list[tuple[str, Tensor]] = []
    if isinstance(output, tuple):
        # A named tuple brings its own names; a plain one is positional,
        # and a state pair has to be able to refer to one of them.
        labels = getattr(output, "_fields", None)
        for position, value in enumerate(output):
            if not isinstance(value, lucid.Tensor):
                continue
            named.append(
                (
                    labels[position] if labels is not None else f"output_{position}",
                    value,
                )
            )
        if named:
            return named
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        for spec in dataclasses.fields(output):
            value = getattr(output, spec.name)
            if isinstance(value, lucid.Tensor):
                named.append((spec.name, value))
    if named:
        return named
    raise TypeError(
        f"lucid.coreml: the model returned {type(output).__name__}, which carries "
        "no tensor to export. Name the field with output_field=..., or return a "
        "Tensor."
    )


def _named_examples(example: object) -> tuple[list[tuple[str, Tensor]], bool]:
    """The example input(s), named, and whether they are passed by keyword.

    A model with more than one input is the common case, not the exotic
    one — a transformer takes an attention mask, an encoder-decoder takes
    both halves — so a tuple is passed positionally and a mapping by
    keyword, which is how the model would be called anyway.
    """
    if isinstance(example, lucid.Tensor):
        return [("input", example)], False
    if isinstance(example, dict):
        for name, value in example.items():
            if not isinstance(value, lucid.Tensor):
                raise TypeError(
                    f"lucid.coreml: example {name!r} is {type(value).__name__}, "
                    "not a Tensor"
                )
        return list(example.items()), True
    if isinstance(example, (tuple, list)):
        named: list[tuple[str, Tensor]] = []
        for position, value in enumerate(example):
            if not isinstance(value, lucid.Tensor):
                raise TypeError(
                    f"lucid.coreml: example {position} is {type(value).__name__}, "
                    "not a Tensor"
                )
            named.append((f"input_{position}", value))
        if not named:
            raise ValueError("lucid.coreml: no example inputs given")
        return named, False
    raise TypeError(
        f"lucid.coreml: example must be a Tensor, a tuple of Tensors, or a "
        f"mapping of name to Tensor — got {type(example).__name__}"
    )


def trace(model: Module, example: object, *, output_field: str | None = None) -> Any:
    """Run one traced forward pass.

    Parameters
    ----------
    model : nn.Module
        Model to trace. Should already be in ``eval()`` mode.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Input(s) to trace with; their identity is how the driver tells the
        arguments apart from the parameters, which are external feeds too.
        A tuple is passed positionally, a mapping by keyword.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to take when the model returns an output
        dataclass. ``None`` takes every tensor field it declares.

    Returns
    -------
    tuple
        ``(graph, external_feeds, inputs, outputs)``, where ``inputs`` is
        a list of ``(feature name, value id, tensor)`` and ``outputs`` a
        list of ``(feature name, value id, tensor)``.

    Raises
    ------
    ValueError
        An input never reached an op, or an output is not in the trace.
    TypeError
        The return value carries no tensor, or lacks the named field.
    """
    examples, by_keyword = _named_examples(example)
    before = _buffer_marks(model)
    with _compile._tracing() as tracer:
        if by_keyword:
            result = model(**dict(examples))
        else:
            result = model(*(tensor for _, tensor in examples))
        selected = _select_outputs(result, output_field)

    after = _buffer_marks(model)
    moved = sorted(
        name for name, mark in after.items() if before.get(name, mark) != mark
    )
    if moved:
        raise StatefulModel(moved)

    inputs: list[tuple[str, int, Tensor]] = []
    for name, tensor in examples:
        value_id = tracer.lookup_id(_unwrap(tensor))
        if value_id is None:
            raise ValueError(
                f"lucid.coreml: example input {name!r} never reached an op — the "
                "model ignored it, or copied it before use"
            )
        inputs.append((name, value_id, tensor))

    outputs: list[tuple[str, int, Tensor]] = []
    for name, tensor in selected:
        value_id = tracer.lookup_id(_unwrap(tensor))
        if value_id is None:
            raise ValueError(
                f"lucid.coreml: output {name!r} did not come out of the traced "
                "graph — no operation recorded it as a result. A model that "
                "reads tensor values while it runs computes that part on the "
                "host, where an exported graph cannot follow: ``.item()``, a "
                "Python ``if`` on a comparison, or an index computed from the "
                "data. ``lucid.histogram`` is one such composite. Whatever "
                "reads values has to become tensor operations before this "
                "model can be exported."
            )
        outputs.append((name, value_id, tensor))
    return tracer.graph, tracer.external_feeds, inputs, outputs, tracer.retained_values


def _flatten_ints(tensor: Tensor) -> list[int]:
    """Every element of an integer tensor, in row-major order.

    ``tolist`` is numpy-free, so reading a constant's values here keeps an
    external import out of this package (H4).

    Parameters
    ----------
    tensor : Tensor
        Integer tensor to read.

    Returns
    -------
    list[int]
        Flattened values.
    """
    flat: list[int] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(_as_int(value))

    walk(tensor.tolist())
    return flat


def _flatten_floats(tensor: Tensor) -> list[float]:
    """A float tensor's values, flattened, for a packed inline payload.

    Parameters
    ----------
    tensor : Tensor
        Float tensor to flatten.

    Returns
    -------
    list[float]
        Flattened values.
    """
    flat: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(_as_float(value))

    walk(tensor.tolist())
    return flat


def _flatten_bools(tensor: Tensor) -> list[bool]:
    """A boolean tensor's values, flattened, for an inline MIL constant.

    Parameters
    ----------
    tensor : Tensor
        Boolean tensor to flatten.

    Returns
    -------
    list[bool]
        Flattened values.
    """
    flat: list[bool] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(bool(value))

    walk(tensor.tolist())
    return flat


# Below this many elements a weight is not worth quantizing: the
# per-channel scale is itself stored uncompressed, so a small tensor can
# come out larger, and the error it adds buys nothing. The same threshold
# the reference tooling uses, stated here rather than inherited.
_QUANTIZE_MIN_ELEMENTS = 2048


#: MIL's spelling of each element type, for the ``cast`` operation's
#: string parameter. Only the types a promotion can land on: casting a
#: bool or a string is a different question, and an emitter that reaches
#: one leaves the operand alone rather than inventing an answer.
def _settle_target(
    wanted: _spec.DeploymentTarget | None,
    *,
    state: object,
    weights: object,
    functions: bool = False,
) -> _spec.DeploymentTarget:
    """The floor this package will actually have, refusing a lower ask.

    Three features move a program from ``CoreML7`` to ``CoreML8``, and
    until this existed they moved it without saying so: a caller who
    needed iOS 17 found out from a device. Asked for a floor the package
    cannot meet, the export stops here and names which feature raised
    it and how the call asked for it.

    Parameters
    ----------
    wanted : DeploymentTarget or None
        What the caller asked for. ``None`` accepts whatever the
        features require.
    state : list of State or None
        The state specification, if any.
    weights : WeightPrecision or Palettize or Sparsify
        How weights are stored; palettization needs the newer opset.
    functions : bool, optional, default=False
        Whether the package carries several entry points.

    Returns
    -------
    DeploymentTarget
        The floor the package has.

    Raises
    ------
    ValueError
        When a requested floor is lower than the features allow.
    """
    reasons: list[str] = []
    if state:
        reasons.append("state")
    if isinstance(weights, _spec.Palettize):
        reasons.append("palettization")
    if functions:
        reasons.append("multiple functions")

    if not reasons:
        return wanted or _spec.DeploymentTarget.IOS17
    if wanted is None or wanted is _spec.DeploymentTarget.IOS18:
        return _spec.DeploymentTarget.IOS18

    named = ", ".join(f"{r} ({_spec._NEEDS_IOS18[r]})" for r in reasons)
    raise ValueError(
        f"lucid.coreml: {wanted.name} was asked for, and this export uses "
        f"{named}, which Core ML expresses only from IOS18. The package "
        "would load on iOS 18 and nowhere earlier — drop the feature or "
        "raise the target, rather than finding out on a device"
    )


def _refuse_if_empty(tensor: Tensor) -> None:
    """Stop on a constant with no elements, and say where it came from.

    An empty constant is not a weight a model happens to leave blank; it
    is a shape the trace could not know. A two-stage detector sizes its
    region features by how many proposals survived, and on an untrained
    model with random input that is often none — so the package would be
    built around this one run's answer and give it for every input.

    The blob writer refuses it too, with "the tensor has no host
    storage", which is true and says nothing about why.

    Parameters
    ----------
    tensor : Tensor
        Constant about to be written.

    Raises
    ------
    UnsupportedOp
        When the tensor has no elements.
    """
    if int(tensor.numel()) != 0:
        return
    raise UnsupportedOp(
        "a constant with no elements",
        f"the traced graph contains an empty tensor of shape "
        f"{tuple(int(d) for d in tensor.shape)}. That shape came from the "
        "example input rather than from the model — a region proposal count, "
        "a non-maximum suppression result — so the package would be built "
        "around this run's answer and give it for every input. Export the "
        "part of the model whose shapes are fixed, and keep the "
        "data-dependent stage in the caller",
    )


_MIL_CAST_NAMES = {
    _C_engine.coreml.DTYPE_FLOAT16: "fp16",
    _C_engine.coreml.DTYPE_FLOAT32: "fp32",
    _C_engine.coreml.DTYPE_INT32: "int32",
    _C_engine.coreml.DTYPE_BOOL: "bool",
}


def _quantize_weight(
    tensor: Tensor, scale_mil: int
) -> tuple[Tensor, bytes, bytes, int] | None:
    """A weight as int8 codes with one scale per output channel.

    Returns ``None`` when the tensor should stay in floating point — a
    bias, a norm's parameters, anything below the threshold, or a rank-1
    value that has no channel axis to scale along.

    Symmetric rather than affine: a symmetric grid puts zero exactly on a
    lattice point, which matters because padding and masking write real
    zeros that an affine grid would round to something else.

    Parameters
    ----------
    tensor : Tensor
        Float weight to quantize.
    scale_mil : int
        MIL dtype the scale is stored as — the body's precision, so the
        dequantized value lands in the type the operation wants.

    Returns
    -------
    tuple or None
        ``(codes, scale bytes, zero-point bytes, channels)``.
    """
    if len(tensor.shape) < 2:
        return None
    count = 1
    for dim in tensor.shape:
        count *= int(dim)
    if count < _QUANTIZE_MIN_ELEMENTS:
        return None

    channels = int(tensor.shape[0])
    flat = tensor.reshape(channels, -1)
    scale, zero_point = _quant.calculate_qparams(
        flat.min(dim=1), flat.max(dim=1), _quant.per_channel_symmetric, _quant.qint8
    )
    codes = _quant.quantize(tensor, scale, zero_point, _quant.qint8, ch_axis=0)

    # MIL has no float16 or int8 immediate list; both travel as raw
    # little-endian bytes, so they are packed here rather than handed to
    # the engine as numbers it would have to re-encode.
    fmt = "<e" if scale_mil == _spec.FLOAT16 else "<f"
    scale_bytes = b"".join(struct.pack(fmt, v) for v in _flatten_floats(scale))
    zero_bytes = b"".join(
        struct.pack("<b", int(v)) for v in _flatten_floats(zero_point)
    )
    return codes, scale_bytes, zero_bytes, channels


#: A palette is fitted from at most this many weights.  The table has at
#: most 256 entries, so past a certain sample the centroids stop moving
#: and every extra element is time spent for nothing — an AlexNet
#: classifier weight has 37 million of them.
_PALETTE_SAMPLE = 1 << 16


def _assign(rows: Tensor, table: Tensor, count: int) -> Tensor:
    """Which palette entry each value falls in, a row at a time.

    ``lucid.bucketize`` answers this for one shared edge list, but it
    walks the whole list per element — a 256-entry table costs 255 passes
    over the weight — and it cannot give each row its own edges, which is
    the whole point of a per-channel palette. The edges are sorted and
    ``count`` is a power of two, so bisection answers in ``log2(count)``
    passes and the running index needs no bounds check: the steps sum to
    ``count - 1`` exactly.

    ``table`` is ``(groups, count)``: column zero is a placeholder that
    nothing probes (the first candidate is already ``count // 2``), and
    the rest are the midpoints between neighbouring palette entries. The
    convention matches ``bucketize`` — the entry is the number of edges
    strictly below the value — which is what the unit test compares
    against.
    """
    found = lucid.zeros_like(rows).to(lucid.int64)
    step = count // 2
    while step >= 1:
        candidate = found + step
        picked = lucid.gather(table, candidate, 1)
        found = lucid.where(picked < rows, candidate, found)
        step //= 2
    return found


def _edge_table(centres: Tensor) -> Tensor:
    """Midpoints between neighbouring palette entries, per row.

    Column zero is the placeholder ``_assign`` never reads, kept so the
    table indexes the same way the palette does.
    """
    groups = int(centres.shape[0])
    lower = centres[:, :-1]
    upper = centres[:, 1:]
    head = lucid.full((groups, 1), -math.inf)
    return lucid.concat([head, (lower + upper) / 2], dim=1)


#: A palette is fitted from at most this many values per group.  The
#: table has at most 256 entries, so past a certain sample the centroids
#: stop moving and every extra element is time spent for nothing.
_PALETTE_SAMPLE = 1 << 14

#: Lloyd passes.  Well past where the movement stops mattering at these
#: sizes, and a fixed count keeps export time predictable.
_PALETTE_PASSES = 10


def _palettes_for(rows: Tensor, count: int) -> Tensor:
    """One palette per row, by Lloyd's algorithm in one dimension.

    Every row is fitted at once rather than in a Python loop over
    channels: a bisection for the assignment and two ``scatter_add``
    passes for the means, all batched. A per-channel fit over a 2048
    channel weight is 2048 independent clusterings, and doing them one at
    a time is the difference between an export that takes seconds and one
    that takes an hour.

    Started from evenly spaced order statistics rather than from the
    range, so a row whose mass sits near zero does not spend most of its
    table on tails it barely uses.
    """
    groups = int(rows.shape[0])
    width = int(rows.shape[1])
    sample = (
        rows[:, :: (width // _PALETTE_SAMPLE + 1)] if width > _PALETTE_SAMPLE else rows
    )
    span = int(sample.shape[1])

    ordered = lucid.sort(sample, dim=-1)
    marks = [
        float(min(span - 1, int((index + 0.5) * span / count)))
        for index in range(count)
    ]
    centres = lucid.gather(
        ordered, (lucid.zeros(groups, 1) + lucid.tensor(marks)).to(lucid.int64), 1
    )

    ones = lucid.ones_like(sample)
    empty = lucid.zeros(groups, count)
    for _ in range(_PALETTE_PASSES):
        assigned = _assign(sample, _edge_table(centres), count)
        totals = lucid.scatter_add(empty, 1, assigned, sample)
        weights = lucid.scatter_add(empty, 1, assigned, ones)
        # An entry no value landed on keeps the centre it had; dividing by
        # its zero population would put a NaN in the table and take the
        # whole row's bisection with it.
        moved = lucid.where(
            weights > 0,
            totals / lucid.where(weights > 0, weights, ones[:, :1]),
            centres,
        )
        # Re-sorted because the bisection needs a monotone table, and a
        # row whose entries crossed would otherwise answer nonsense.
        centres = lucid.sort(moved, dim=-1)
    return centres


def _palette_groups(channels: int, count: int, elements: int) -> int:
    """How many palettes to fit along the output-channel axis.

    One per channel is what makes palettization usable — a convolution's
    channels differ in magnitude by orders of magnitude, and a single
    shared table spends nearly all of its resolution on the loudest ones.
    But the tables are stored, so at 256 entries and 2048 channels they
    would cost more than the keys they index. Groups are halved until the
    table is a small fraction of the payload it serves.
    """
    key_bytes = elements * (count.bit_length() - 1) // 8
    groups = channels
    while groups > 1 and groups * count * 2 > key_bytes // 4:
        groups //= 2
    # The group count has to divide the channel count for the operation
    # to lay the tables out over the axis.
    while groups > 1 and channels % groups:
        groups -= 1
    return max(groups, 1)


def _palettize_weight(
    tensor: Tensor, bits: int
) -> tuple[bytes, Tensor, list[int], int] | None:
    """A weight as packed keys into one palette per group of channels.

    ``None`` when the weight is too small to be worth the tables, or has
    no channel axis — the same two refusals the int8 path makes, for the
    same reasons: the tables cost something fixed, and a bias or a norm's
    parameters have no output channel to group along and are ruinous to
    approximate.

    Returns the packed keys, the tables, the shape the operation wants
    them in, and the width the keys were packed at — the last so the
    caller picks its dtype tags from the same number that did the
    packing rather than from the request.
    """
    if len(tensor.shape) < 2 or int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    count = 1 << bits
    channels = int(tensor.shape[0])
    elements = int(tensor.numel())
    groups = _palette_groups(channels, count, elements)
    rows = tensor.reshape(groups, -1)

    palettes = _palettes_for(rows, count)
    keys = _assign(rows, _edge_table(palettes), count)
    # ``[groups, 1, ..., 1, count, 1]``: one table per group along the
    # output-channel axis, then the entries, then the vector width — one,
    # because these are scalar palettes.
    lut_shape = [groups] + [1] * (len(tensor.shape) - 1) + [count, 1]
    return (
        _C_engine.coreml.pack_bits(_unwrap(keys.reshape(-1)), bits),
        palettes.reshape(-1),
        lut_shape,
        bits,
    )


def _sparsify_weight(tensor: Tensor, ratio: float) -> tuple[Tensor, bytes, int] | None:
    """A weight as its surviving values plus one bit per element.

    ``None`` when nothing would be dropped, or when the weight is small
    enough that the mask costs more than the zeros save.
    """
    if len(tensor.shape) < 2 or ratio <= 0.0:
        return None
    if int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    flat = tensor.reshape(-1)
    threshold = float(lucid.quantile(flat.abs(), ratio).item())
    keep = flat.abs() > threshold
    kept = int(keep.to(lucid.int64).sum().item())
    if kept == int(flat.numel()) or kept == 0:
        # Nothing to drop, or nothing to keep. The second happens for
        # real: a constant whose elements are all the same value has no
        # element above its own median, and a folded batch norm leaves
        # whole zero vectors behind. Such a constant would be a sparse
        # const with an empty payload, which is not something to write —
        # it goes out dense, at a cost of one small tensor.
        return None

    # Both of these are tensor work rather than Python loops: a weight
    # carries tens of millions of elements and iterating them here cost
    # minutes per layer.
    values = lucid.masked_select(flat, keep)
    return values, _C_engine.coreml.pack_bits(_unwrap(keep), 1), kept


def _operands(bindings: Bindings) -> list[tuple[str, list[str]]]:
    """Parameter bindings in the shape the engine takes.

    An emitter may bind a parameter to one operand or to several —
    ``concat`` is variadic — and the engine wants a list either way.

    Parameters
    ----------
    bindings : Bindings
        Pairs of parameter name and operand name, or names.

    Returns
    -------
    list[tuple[str, list[str]]]
        The same pairs, with every operand side a list.
    """
    return [
        (param, [value] if isinstance(value, str) else list(value))
        for param, value in bindings
    ]


class Builder:
    """Mints the constants MIL requires and appends operations.

    Every scalar operand in a MIL program is its own ``const`` operation —
    a convolution's ``groups`` is a value, not an attribute — so emitters
    ask the builder for names rather than writing literals.
    """

    def __init__(
        self,
        program: MilProgram,
        blob: BlobWriter,
        body_mil: int = 0,
        body_blob: int = 0,
        half: bool = False,
    ) -> None:
        self._program = program
        self._blob = blob
        self._body_mil = body_mil
        self._body_blob = body_blob
        self._half = half
        self._counter = 0
        # Value name -> shape, for emitters that need an operand's rank
        # rather than just its name (``layer_norm`` normalises over the
        # trailing axes its weight covers).
        self.shapes: dict[str, list[int]] = {}
        # Values that are constants.  Core ML rejects a computed operand
        # where it wants a constant, so emitters check before binding.
        self.consts: set[str] = set()
        # Values emitted one rank lower than the trace records, because
        # their leading axis is one and Core ML stops at rank five. Keyed
        # by MIL value name so an emitter can ask about its operand.
        self.unit_axis_dropped: set[str] = set()
        # The same set keyed by traced value id, for an emitter asking
        # about its own result rather than about an operand.
        self.thinned: set[int] = set()
        # Value name -> MIL element type. Core ML does not promote: an
        # elementwise operation whose two operands differ in dtype is
        # refused when the package is parsed, naming an internal value
        # and nothing else. Recorded so an emitter can insert the cast.
        self.dtypes: dict[str, int] = {}
        # Traced value id -> the axes a flexible export leaves open, so an
        # emitter that writes a shape into a constant can leave them open
        # too rather than baking the default.
        self.varying: dict[int, set[int]] = {}

    def result_shape(self, op: TracedOp, index: int = 0) -> list[int]:
        """One of an operation's result shapes, with varying axes as ``-1``.

        Emitters that carry a shape in a constant — ``reshape`` is the
        one — must ask for it this way, or a flexible export bakes the
        default and is wrong at every other shape.

        Parameters
        ----------
        op : TracedOp
            Operation whose result to describe.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        list[int]
            The shape, with ``-1`` where the export is flexible.
        """
        out = op.outputs[index]
        shape = [int(d) for d in out.shape]
        # A value the driver emits a rank lower loses its leading axis
        # here too, or an emitter that writes the shape into a constant
        # would ask for the rank Core ML refused.
        if int(out.id) in self.thinned:
            shape = shape[1:]
        return _flex(shape, self.varying.get(out.id))

    def result_mil_dtype(self, op: TracedOp, index: int = 0) -> int:
        """MIL element type the driver will declare for a result.

        An emitter that has to build a constant of the *result's* type —
        ``one_hot``'s two fill values must match it, and Core ML says so
        at parse time — needs the same answer the driver reaches, not a
        guess from the operation's name.

        Parameters
        ----------
        op : TracedOp
            Operation whose result to describe.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        int
            MIL element-type number.
        """
        out = op.outputs[index]
        return _spec.trace_dtype(str(out.dtype).split(".")[-1], self._body_mil)

    def result_thinned(self, op: TracedOp, index: int = 0) -> bool:
        """Whether a result is emitted without its leading axis.

        Parameters
        ----------
        op : TracedOp
            Operation whose result to ask about.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        bool
            True when the driver drops the leading axis of this result.
        """
        return int(op.outputs[index].id) in self.thinned

    def leading_axis_dropped(self, name: str) -> bool:
        """Whether this value is a rank lower than the trace recorded.

        An operation that names an axis has to move it when its operand
        lost the leading one. Everything else — arithmetic, and the shape
        operations that reshape to their recorded output — adapts by
        itself.

        Parameters
        ----------
        name : str
            Value to ask about.

        Returns
        -------
        bool
            True when the leading axis was dropped.
        """
        return name in self.unit_axis_dropped

    def dtype_of(self, name: str) -> int | None:
        """MIL element type of a value, when the builder recorded one.

        ``None`` for a value nothing declared — an emitter should leave
        such an operand alone rather than guess at a cast.

        Parameters
        ----------
        name : str
            Value to look up.

        Returns
        -------
        int or None
            MIL element-type number, or ``None`` if unrecorded.
        """
        return self.dtypes.get(name)

    def agree_on_dtype(self, names: list[str], target: int | None = None) -> list[str]:
        """The same values, cast so an elementwise operation accepts them.

        Core ML does not promote. ``mul`` of an int32 by a float32
        constant is refused when the package is parsed — with an
        internal value name and no indication of which operand is
        wrong — where Lucid promoted and moved on. CLIP builds its
        position indices that way.

        Pass the operation's declared result type as ``target``: an
        arithmetic operation must have operands of the type its output
        was declared as, and guessing a promotion instead is how an
        ``int64 * 3`` ends up with float operands feeding an int32
        result. Leave ``target`` out for a comparison, whose result is
        boolean and whose operands only have to match each other.

        Values whose type was never recorded are left alone: an unforced
        cast can be as wrong as a missing one.

        Parameters
        ----------
        names : list of str
            Operands of one elementwise operation.
        target : int or None, optional, default=None
            MIL element type to cast to. ``None`` promotes among the
            operands, with the float side winning.

        Returns
        -------
        list of str
            The operands, some possibly replaced by a cast of themselves.
        """
        kinds = {
            kind for kind in (self.dtype_of(name) for name in names) if kind is not None
        }
        if len(kinds) < 2 and (target is None or kinds <= {target}):
            return names
        if target is None:
            floats = {
                _C_engine.coreml.DTYPE_FLOAT16,
                _C_engine.coreml.DTYPE_FLOAT32,
            }
            settled = self._body_mil if kinds & floats else max(kinds)
        else:
            settled = target
        spelling = _MIL_CAST_NAMES.get(settled)
        if spelling is None:
            return names
        agreed = []
        for name in names:
            kind = self.dtype_of(name)
            if kind is None or kind == settled:
                agreed.append(name)
                continue
            agreed.append(
                self.emit(
                    "cast",
                    [("x", name), ("dtype", self.const_str(spelling))],
                    # A scalar constant has no recorded shape, and none
                    # is the right answer for it.
                    self.shapes.get(name, []),
                    dtype=settled,
                )
            )
        return agreed

    def result_cast_spelling(self, op: TracedOp, index: int = 0) -> str | None:
        """MIL's name for the type a result is *declared* as.

        Not the type the trace asked for. A float16 program declares
        every float intermediate as float16, so a model that casts to
        float32 in the middle — VQ-VAE does, around its codebook — has a
        ``cast`` whose requested type and whose declared output type
        disagree, and Core ML refuses the package with "Specified dtype
        of cast does not match that of output tensor".

        Parameters
        ----------
        op : TracedOp
            Operation whose result to name.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        str or None
            MIL's spelling, or ``None`` for a type ``cast`` cannot name.
        """
        return _MIL_CAST_NAMES.get(self.result_mil_dtype(op, index))

    def narrow_to_int32(self, name: str) -> str:
        """The same value as int32, for an operand that takes nothing wider.

        This opset's ``gather`` accepts int32 indices and smaller, and
        refuses int64 when the package is parsed rather than converting
        it. Casting int32 to int32 is a no-op the compiler drops, so the
        emitter does not have to know which case it is in.
        """
        return self.emit(
            "cast",
            [("x", name), ("dtype", self.const_str("int32"))],
            self.shape_of(name),
            dtype=_C_engine.coreml.DTYPE_INT32,
        )

    def result_is_float(self, op: TracedOp, index: int = 0) -> bool:
        """Whether a result is declared as a floating type."""
        return self.result_mil_dtype(op, index) in (
            _C_engine.coreml.DTYPE_FLOAT16,
            _C_engine.coreml.DTYPE_FLOAT32,
        )

    def shape_of(self, name: str) -> list[int]:
        shape = self.shapes.get(name)
        if shape is None:
            raise KeyError(
                f"lucid.coreml: no shape recorded for value {name!r} — an emitter "
                "asked about an operand the driver never registered"
            )
        return shape

    def _next(self, kind: str) -> str:
        self._counter += 1
        return f"_c{self._counter}_{kind}"

    def _const_name(self, kind: str) -> str:
        """Allocate a name and remember that it names a constant.

        Core ML refuses some operands that are not constants —
        ``linear``'s weight and bias, a convolution's weight — so an
        emitter has to be able to ask. Recorded rather than inferred from
        the name: ``_next`` also names the results of operations, and a
        prefix check would call those constants too.
        """
        name = self._next(kind)
        self.consts.add(name)
        return name

    def mark_const(self, name: str) -> None:
        """Record a constant the driver named itself, such as a weight."""
        self.consts.add(name)

    def is_const(self, name: str) -> bool:
        """Whether this value is a constant Core ML will accept as one."""
        return name in self.consts

    def const_ints(self, values: list[int]) -> str:
        name = self._const_name("ints")
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        self._program.add_int_const(name, [int(v) for v in values], False)
        return name

    def const_ints_shaped(self, values: list[int], shape: list[int]) -> str:
        """An integer constant of arbitrary shape, carried inline.

        ``const_from_tensor`` is the blob path and the blob holds float
        payloads only, so an integer tensor — a gather's indices, say —
        cannot go through it without changing dtype.

        Parameters
        ----------
        values : list[int]
            Flattened values, in row-major order.
        shape : list[int]
            Shape to declare.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("intsn")
        self._program.add_int_const_shaped(name, [int(v) for v in values], shape)
        self.shapes[name] = list(shape)
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        return name

    def const_int(self, value: int) -> str:
        name = self._const_name("int")
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        self._program.add_int_const(name, [int(value)], True)
        return name

    def const_float(self, value: float) -> str:
        """A float scalar, at the body's precision.

        In an fp16 program this has to be fp16 too — MIL rejects a
        ``batch_norm`` whose ``epsilon`` is float32 while its statistics
        are float16, and that rejection happens at load time, not at
        export. Core ML writes fp16 constants into the weight blob rather
        than inline (a reference fp16 package contains no fp16 immediate
        values at all), so this follows it there.
        """
        name = self._const_name("float")
        self.dtypes[name] = _C_engine.coreml.DTYPE_FLOAT32
        if not self._half:
            self._program.add_float_const(name, [float(value)], True)
            return name
        payload = lucid.tensor([float(value)]).half()
        offset = self._blob.append_tensor(_unwrap(payload), self._body_blob)
        # Declared scalar (rank 0) even though the payload is one element:
        # MIL types ``epsilon`` as a scalar and rejects tensor<fp16, [1]>.
        self._program.add_blob_const(name, (self._body_mil, []), offset)
        return name

    def const_float32_shaped(self, values: list[float], shape: list[int]) -> str:
        """A float32 constant of arbitrary shape, carried inline.

        Image preprocessing needs a per-channel bias shaped
        ``(1, C, 1, 1)``, which neither the scalar nor the rank-1 helper
        can express, and which must stay float32 because it is applied
        before the body's cast.

        Parameters
        ----------
        values : list[float]
            Flattened values, row-major.
        shape : list[int]
            Shape to declare.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("f32n")
        self._program.add_float_const_shaped(name, [float(v) for v in values], shape)
        self.shapes[name] = list(shape)
        self.dtypes[name] = _C_engine.coreml.DTYPE_FLOAT32
        return name

    def const_float32(self, value: float) -> str:
        """A float scalar that stays float32 whatever the body's precision.

        Not every float parameter is data. ``epsilon`` is added to the
        activations and must share their dtype — MIL rejects a float32
        epsilon beside float16 statistics. A resampling ``scale_factor``
        is the opposite: it configures the operator, and MIL accepts only
        int32 or float32 there, rejecting the float16 that following the
        body would produce.

        Parameters
        ----------
        value : float
            Scalar value.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("f32")
        self.dtypes[name] = _C_engine.coreml.DTYPE_FLOAT32
        self._program.add_float_const(name, [float(value)], True)
        return name

    def const_str(self, value: str) -> str:
        name = self._const_name("str")
        self._program.add_string_const(name, value)
        return name

    def const_from_tensor(self, tensor: Tensor) -> str:
        """A constant of arbitrary shape, carried in the weight blob.

        The inline constant helpers make rank-1 values, which is enough for
        the small integer lists MIL wants for axes and strides. An op that
        *produces* a tensor out of nothing — ``zeros``, ``full``,
        ``arange`` — needs its real shape, and the blob path already
        handles shape and dtype for weights.

        Parameters
        ----------
        tensor : Tensor
            Host tensor holding the constant's value.

        Returns
        -------
        str
            Name of the constant.
        """
        _refuse_if_empty(tensor)
        payload = tensor.half() if self._half else tensor
        name = self._const_name("blobconst")
        offset = self._blob.append_tensor(_unwrap(payload), self._body_blob)
        shape = [int(d) for d in tensor.shape]
        self._program.add_blob_const(name, (self._body_mil, shape), offset)
        self.shapes[name] = shape
        self.dtypes[name] = self._body_mil
        return name

    def emit(
        self,
        mil_type: str,
        bindings: Bindings,
        shape: list[int],
        dtype: int | None = None,
    ) -> str:
        """Append an intermediate operation and return its value name.

        Some Lucid ops are several MIL ops — scaled dot-product attention
        is a matmul, a scale, a softmax and another matmul — so an emitter
        needs to place the ones before the last itself. The driver names
        and types the final one from the trace.

        Parameters
        ----------
        mil_type : str
            MIL operator name.
        bindings : list
            ``(parameter, value name)`` pairs, as an emitter returns.
        shape : list[int]
            Result shape.
        dtype : int or None, optional, default=None
            MIL element type of the result. ``None`` is the program
            body's, which is right for arithmetic; a comparison has to
            say ``BOOL`` or Core ML rejects the operation for producing
            the wrong type.

        Returns
        -------
        str
            Name of the value this operation produces.
        """
        name = self._next(mil_type)
        normalised = _operands(bindings)
        self._program.add_op(
            mil_type,
            normalised,
            name,
            (self._body_mil if dtype is None else dtype, list(shape)),
        )
        self.shapes[name] = list(shape)
        self.dtypes[name] = self._body_mil if dtype is None else int(dtype)
        return name

    def emit_multi(
        self, mil_type: str, bindings: Bindings, shapes: list[list[int]]
    ) -> list[str]:
        """Append an intermediate operation with several results.

        ``split`` is the one that needs it — decomposing a ``roll`` into a
        split and a swapped concat keeps the translation on operations
        that are already verified, instead of reaching for a slice whose
        masks would be a second thing to get right.

        Parameters
        ----------
        mil_type : str
            MIL operator name.
        bindings : list
            ``(parameter, value name)`` pairs.
        shapes : list[list[int]]
            One shape per result.

        Returns
        -------
        list[str]
            Names of the values produced, in order.
        """
        normalised = _operands(bindings)
        names = [self._next(f"{mil_type}{i}") for i in range(len(shapes))]
        self._program.add_op_multi(
            mil_type,
            normalised,
            [(n, (self._body_mil, list(sh))) for n, sh in zip(names, shapes)],
        )
        for n, sh in zip(names, shapes):
            self.shapes[n] = list(sh)
        return names

    def const_bool(self, value: bool) -> str:
        name = self._const_name("bool")
        self._program.add_bool_const(name, bool(value))
        return name


class _WindowRewrite(NamedTuple):
    """A ``reshape → permute → reshape`` whose middle exceeds the rank cap.

    Keyed and cross-referenced by value id rather than by op identity:
    ``graph.ops`` hands out fresh Python wrappers on each access, so
    ``id(op)`` is stable only by accident and silently stops matching on
    a graph large enough to matter.

    ``absorbs`` are the two follow-on ops the rewrite consumes, so the
    emit loop knows to skip them; ``out_id`` is the value the rewritten
    chain has to bind, which is the final reshape's.
    """

    mid_shape: list[int]
    perm: list[int]
    out_shape: list[int]
    out_id: int
    absorbs: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _StuffRewrite:
    """A ``reshape`` / ``pad`` / ``reshape`` that stuffs zeros between elements.

    Upsampling before a convolution is often written by splitting each
    axis into ``(d, 1)``, padding the new axis out to ``(d, k)`` with
    zeros, and merging back to ``d * k`` — every element followed by
    ``k - 1`` zeros. Doing two axes at once needs rank 6, which Core ML
    does not have.
    """

    source_shape: list[int]
    factors: list[int]
    fill: float
    out_shape: list[int]
    out_id: int
    absorbs: tuple[int, ...]


def _stuff_rewrites(graph: Any) -> dict[int, _StuffRewrite]:
    """Find the zero-stuffing upsamples that can be staged under the cap.

    Matched on shape rather than on the model: the tall tensor has to be
    the source with singleton axes interleaved, the pad has to add only
    at the end of exactly those axes, and the result has to merge each
    pair back. Anything else keeps its rank and is refused as before.
    """
    producer: dict[int, Any] = {}
    readers: dict[int, list[Any]] = {}
    for op in graph.ops:
        for out in op.outputs:
            producer[out.id] = op
        for iid in op.inputs:
            readers.setdefault(iid, []).append(op)

    found: dict[int, _StuffRewrite] = {}
    for op in graph.ops:
        if op.name != "reshape" or len(op.outputs) != 1:
            continue
        tall = op.outputs[0]
        if len(tall.shape) <= _MAX_RANK:
            continue
        mid_readers = readers.get(tall.id, [])
        if len(mid_readers) != 1:
            continue
        pad_op = mid_readers[0]
        if pad_op.name != "pad" or len(pad_op.outputs) != 1:
            continue
        padded = pad_op.outputs[0]
        down_readers = readers.get(padded.id, [])
        if len(down_readers) != 1:
            continue
        down = down_readers[0]
        if down.name != "reshape" or len(down.outputs) != 1:
            continue
        if len(down.outputs[0].shape) > _MAX_RANK:
            continue

        tall_shape = [int(d) for d in tall.shape]
        pads = [int(v) for v in (pad_op.attrs.get("pads") or [])]
        if len(pads) != 2 * len(tall_shape):
            continue

        # Only singleton axes may grow, and only at their far end: that
        # is what makes the pad an interleave rather than a border.
        widths: dict[int, int] = {}
        rejected = False
        for axis, extent in enumerate(tall_shape):
            before, after = pads[2 * axis], pads[2 * axis + 1]
            if before != 0 or (after != 0 and extent != 1):
                rejected = True
                break
            if after:
                widths[axis] = after + 1
        if rejected or not widths:
            continue

        # The source is the tall shape without those singletons — read
        # off the shapes rather than from the producing operation, which
        # does not exist when the reshape reads a model input.
        source_shape = [d for i, d in enumerate(tall_shape) if i not in widths]
        factors = [1] * len(source_shape)
        walk = -1
        for axis, extent in enumerate(tall_shape):
            if axis in widths:
                if walk < 0:
                    rejected = True
                    break
                factors[walk] = widths[axis]
            else:
                walk += 1
        if rejected:
            continue

        out_shape = [int(d) for d in down.outputs[0].shape]
        if [d * f for d, f in zip(source_shape, factors)] != out_shape:
            continue

        found[tall.id] = _StuffRewrite(
            source_shape=source_shape,
            factors=factors,
            fill=float(pad_op.attrs.get("constant", 0.0)),
            out_shape=out_shape,
            out_id=down.outputs[0].id,
            absorbs=(padded.id, down.outputs[0].id),
        )
    return found


def _stage_zero_stuff(builder: Any, source: str, plan: _StuffRewrite) -> str:
    """Stuff one axis at a time, so no stage exceeds the rank cap.

    Splitting every axis at once is what needed rank 6. Splitting one
    axis, padding it and merging it back leaves the rank where it
    started, so a rank-4 tensor never passes rank 5 however many axes
    are upsampled.
    """
    current = list(plan.source_shape)
    value = source
    for axis, factor in enumerate(plan.factors):
        if factor <= 1:
            continue
        split = current[:axis] + [current[axis], 1] + current[axis + 1 :]
        value = builder.emit(
            "reshape",
            [("x", value), ("shape", builder.const_ints(split))],
            split,
        )
        pads = [0] * (2 * len(split))
        pads[2 * (axis + 1) + 1] = factor - 1
        widened = list(split)
        widened[axis + 1] = factor
        value = builder.emit(
            "pad",
            [
                ("x", value),
                ("pad", builder.const_ints(pads)),
                ("mode", builder.const_str("constant")),
                ("constant_val", builder.const_float(plan.fill)),
            ],
            widened,
        )
        current[axis] *= factor
        value = builder.emit(
            "reshape",
            [("x", value), ("shape", builder.const_ints(current))],
            list(current),
        )
    return value


#: Operations that a leading axis of one can be taken away from without
#: changing what they compute. Elementwise arithmetic follows its
#: operands; the shape operations here are emitted as a reshape to their
#: recorded output, so they adapt on their own; ``split_at`` needs its
#: axis moved and knows to ask.
_UNIT_AXIS_SAFE = frozenset(
    {
        "add",
        "div",
        "mul",
        "sub",
        "maximum",
        "minimum",
        "pow",
        "reshape",
        "squeeze",
        "unsqueeze",
        "split_at",
    }
)


#: Operations that make a tensor out of nothing. A traced node with no
#: recorded inputs is one of these, or it is an operation whose inputs
#: the tracer failed to record — and treating the second kind as a
#: constant is how a package comes to answer every input with the first
#: one's result. Only names that genuinely take no tensor are here:
#: ``tril`` and ``meshgrid`` do take one, whatever a node missing its
#: edges suggests.
_TRUE_FACTORIES = frozenset(
    {"arange", "empty", "eye", "full", "linspace", "logspace", "ones", "zeros"}
)


def _foldable(
    graph: Any,
    feeds: dict[int, Any],
    input_ids: set[int],
    traced_values: dict[int, Any],
) -> tuple[set[int], set[int]]:
    """Values a package can carry instead of computing, and who makes them.

    A windowed transformer spends a third of its graph on arithmetic that
    has nothing to do with its input: Swin builds the index into its
    relative-position table out of ``arange``, and rebuilds it on every
    prediction. 503 of its 1473 operations are that, and they run on the
    CPU, between operations that run on the Neural Engine.

    Constant means constant *here*: reachable only from the model's
    weights and from operations that make a tensor out of nothing.
    Anything one step from the input is not, and an operation whose
    inputs the tracer never recorded is treated as unknown rather than
    as a factory — that mistake is what freezes a model's answer.

    Returns
    -------
    tuple[set[int], set[int]]
        The values to emit as constants — only those something outside
        the constant region reads, since the rest have no consumer left
        — and the ids of every operation head inside the region, which
        the driver skips.
    """
    constant: set[int] = {tid for tid in feeds if tid not in input_ids}
    inside: set[int] = set()
    for op in graph.ops:
        if not op.outputs:
            continue
        if op.inputs:
            if not all(i in constant for i in op.inputs):
                continue
        elif op.name not in _TRUE_FACTORIES:
            continue
        if not all(int(o.id) in traced_values for o in op.outputs):
            # Nothing to write in its place.
            continue
        inside.add(int(op.outputs[0].id))
        constant.update(int(o.id) for o in op.outputs)

    if not inside:
        return set(), set()

    # Only the values something outside the region reads have to be
    # written; the rest were steps on the way to them.
    wanted: set[int] = set()
    for op in graph.ops:
        if op.outputs and int(op.outputs[0].id) in inside:
            continue
        wanted.update(i for i in op.inputs if i in constant and i not in feeds)
    return wanted, inside


def _unit_axis_rewrites(
    graph: Any, handled: set[int], folded: set[int] | None = None
) -> set[int]:
    """Rank-6 values that can be emitted without their leading axis.

    The two staging rewrites erase a rank-6 value that is only reshaped
    and permuted. Deformable attention does something they cannot touch:
    it *computes* at rank 6, over (batch, queries, heads, levels, points,
    xy), with ordinary arithmetic between the reshapes. Mask2Former is
    built out of it.

    Every one of those values carries a batch axis of one, and dropping
    an axis of one from every operand of an elementwise operation leaves
    the result unchanged — broadcasting sees the same shapes with the
    same trailing alignment. So the whole connected set is emitted a
    rank lower, and the operations that leave it reshape to their
    recorded output anyway, which restores the axis without being asked.

    Refuses unless the whole set qualifies: a single value that does not
    have the leading one, or a single operation outside the safe list,
    and none of it is rewritten. A partial rewrite would put a rank-5
    value where the rest of the graph expects rank 6.
    """
    tall: set[int] = set()
    for op in graph.ops:
        for out in op.outputs:
            if len(out.shape) > _MAX_RANK and out.id not in handled:
                if int(out.shape[0]) != 1:
                    return set()
                tall.add(out.id)
    if not tall:
        return set()

    gone = folded or set()
    for op in graph.ops:
        if op.outputs and int(op.outputs[0].id) in gone:
            # Not emitted at all; its result arrives as a constant, which
            # the driver writes at whatever rank this pass settles on.
            continue
        touches = any(o.id in tall for o in op.outputs) or any(
            i in tall for i in op.inputs
        )
        if touches and op.name not in _UNIT_AXIS_SAFE:
            return set()
    return tall


def _window_rewrites(graph: Any) -> dict[int, _WindowRewrite]:
    """Find the high-rank triples that can be staged under the cap.

    A reshape that lifts a tensor past rank 5 is representable only if
    the rank comes straight back down, which in practice means a permute
    and a reshape follow it and nothing else reads the tall value. That
    is the window partition (and its inverse, the merge); the pattern is
    matched rather than the model, so a grid partition or any other
    factorised permutation is the same code path.
    """
    producer: dict[int, Any] = {}
    readers: dict[int, list[Any]] = {}
    for op in graph.ops:
        for out in op.outputs:
            producer[out.id] = op
        for iid in op.inputs:
            readers.setdefault(iid, []).append(op)

    found: dict[int, _WindowRewrite] = {}
    for op in graph.ops:
        if op.name != "reshape" or len(op.outputs) != 1:
            continue
        tall = op.outputs[0]
        if len(tall.shape) <= _MAX_RANK:
            continue
        # Exactly one reader, and it has to be the permute — a second
        # reader would need the tall tensor to exist for real.
        mid_readers = readers.get(tall.id, [])
        if len(mid_readers) != 1:
            continue
        perm_op = mid_readers[0]
        if perm_op.name != "permute" or len(perm_op.outputs) != 1:
            continue
        turned = perm_op.outputs[0]
        down_readers = readers.get(turned.id, [])
        if len(down_readers) != 1:
            continue
        down = down_readers[0]
        if down.name != "reshape" or len(down.outputs) != 1:
            continue
        if len(down.outputs[0].shape) > _MAX_RANK:
            continue
        raw = perm_op.attrs["permutation"]
        perm = [int(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw])]
        if sorted(perm) != list(range(len(tall.shape))):
            continue
        found[tall.id] = _WindowRewrite(
            mid_shape=[int(d) for d in tall.shape],
            perm=perm,
            out_shape=[int(d) for d in down.outputs[0].shape],
            out_id=down.outputs[0].id,
            absorbs=(turned.id, down.outputs[0].id),
        )
    return found


def _stage_permutation(builder: Any, source: str, plan: _WindowRewrite) -> str:
    """Realise ``plan`` without ever materialising the tall tensor.

    A reshape is a reinterpretation of a contiguous buffer, so any run of
    adjacent axes may be grouped into one. Each stage therefore views the
    value as ``(left, a, b, right)`` — rank 4, whatever the logical rank
    is — swaps the middle pair, and leaves the result contiguous in the
    new order for the next stage to regroup. Sorting the axis order into
    the wanted one by adjacent swaps then costs one stage per inversion,
    and the tall shape stays a bookkeeping detail.
    """
    shape = list(plan.mid_shape)
    order = list(range(len(shape)))
    name = source

    for slot, axis in enumerate(plan.perm):
        at = order.index(axis)
        while at > slot:
            i = at - 1
            left = math.prod(shape[:i]) if i else 1
            right = math.prod(shape[i + 2 :]) if i + 2 < len(shape) else 1
            grouped = [left, shape[i], shape[i + 1], right]
            viewed = builder.emit(
                "reshape",
                [("x", name), ("shape", builder.const_ints(grouped))],
                grouped,
            )
            swapped = [left, shape[i + 1], shape[i], right]
            name = builder.emit(
                "transpose",
                [("x", viewed), ("perm", builder.const_ints([0, 2, 1, 3]))],
                swapped,
            )
            shape[i], shape[i + 1] = shape[i + 1], shape[i]
            order[i], order[i + 1] = order[i + 1], order[i]
            at = i

    staged: str = builder.emit(
        "reshape",
        [("x", name), ("shape", builder.const_ints(plan.out_shape))],
        plan.out_shape,
    )
    return staged


def _reachable_ops(graph: Any, output_ids: list[int]) -> list[Any]:
    """The traced ops the output actually depends on, in trace order.

    A trace can carry operations nothing consumes. Lucid's ``norm``
    computes a guard against a zero scale eagerly, so the comparison
    collapses to a constant feed and the ``bitwise_and`` that would have
    combined it is left behind with no operands and no consumer. Emitting
    such an op means asking an emitter for something the graph never
    computed; refusing to emit it is both correct and smaller.

    Parameters
    ----------
    graph : trace graph
        Graph to walk.
    output_ids : list[int]
        Values the package returns.

    Returns
    -------
    list
        Ops in their original order, minus the unreachable ones.
    """
    producer = {out.id: op for op in graph.ops for out in op.outputs}
    wanted = set(output_ids)
    pending = list(output_ids)
    while pending:
        op = producer.get(pending.pop())
        if op is None:
            continue
        for value in op.inputs:
            if value not in wanted:
                wanted.add(value)
                pending.append(value)
    return [op for op in graph.ops if any(out.id in wanted for out in op.outputs)]


def _apply_image_normalisation(tensor: Tensor, spec: ImageInput) -> Tensor:
    """``pixel * scale + bias``, the way the exported package applies it.

    Used to put the eager model on the same footing when verifying an
    image export: the package normalises internally, so comparing it
    against a model fed raw pixels would compare two different inputs.

    Parameters
    ----------
    tensor : Tensor
        Pixel values, shaped ``(1, C, H, W)``.
    spec : ImageInput
        The normalisation the package carries.

    Returns
    -------
    Tensor
        Normalised input.
    """
    out = tensor * spec.scale if spec.scale != 1.0 else tensor
    if spec.bias:
        offsets = lucid.tensor([[[[b]] for b in spec.bias]])
        out = out + offsets
    return out


class ShapeNotFlexible(NotImplementedError):
    """An operation's configuration was derived from the input's size.

    Named rather than generic because the alternative is a package that
    accepts several shapes and is only correct at one of them. An adaptive
    pool is the usual cause: the tracer records it as an average pool
    whose kernel came from the input, so the same model traced at two
    resolutions produces two different kernels.
    """

    def __init__(self, op_name: str, detail: str) -> None:
        super().__init__(
            f"lucid.coreml: operation {op_name!r} cannot take a flexible shape — "
            f"{detail}. Trace it at one shape, or replace the layer with one whose "
            "configuration does not depend on the input size."
        )
        self.op_name = op_name


class StatefulModel(NotImplementedError):
    """The model changed its own buffers while being traced.

    An exported package is a pure function of its inputs: whatever the
    buffers held is written into it as constants. A model that updates a
    buffer as it runs — a counter, a cache, a running statistic — keeps
    accumulating in eager and stops the moment it is exported.

    The first call still agrees, because the constants were read after
    the trace, which is what makes this worth refusing rather than
    warning about: ``verify`` runs one prediction and passes.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__(
            f"lucid.coreml: tracing changed {', '.join(names)}, so this model is "
            "not a pure function of its input and an exported package would stop "
            "accumulating after the first call — while still agreeing on that "
            "call. Express the state as an input the model reads and an output "
            "it returns, and name the pair with state=..., or export a model "
            "that does not write to its own buffers."
        )
        self.names = names


def _buffer_marks(model: Module) -> dict[str, float]:
    """A cheap fingerprint of every buffer, to notice one changing.

    A sum rather than a copy: buffers are read once before the trace and
    once after, and copying them all would double the memory an export
    already spends on weights. A mutation that leaves the sum unchanged
    would escape, which is a trade this check states rather than hides.

    Parameters
    ----------
    model : nn.Module
        Model whose buffers to fingerprint.

    Returns
    -------
    dict[str, float]
        Buffer name to the sum of its values.
    """
    marks: dict[str, float] = {}
    for name, buffer in model.named_buffers():
        if buffer is None:
            continue
        marks[name] = float(buffer.sum().item())
    return marks


def _varying_axes(
    model: Module,
    example: Tensor,
    shapes: list[tuple[int, ...]],
    output_field: str | None,
) -> dict[int, set[int]]:
    """Which axes of which traced values change with the input shape.

    Found by tracing the model once per shape and comparing, rather than
    propagating symbols through the graph. The tracer already knows every
    value's shape; asking it twice is cheaper than teaching it algebra,
    and it cannot disagree with itself about how an operation behaves.

    Parameters
    ----------
    model : nn.Module
        Model to trace, in ``eval()`` mode.
    example : Tensor
        Supplies the dtype the probes are built with.
    shapes : list[tuple[int, ...]]
        Shapes to compare, the default first.
    output_field : str or None
        Passed through to :func:`trace`.

    Returns
    -------
    dict[int, set[int]]
        Value id to the axes that varied.

    Raises
    ------
    ShapeNotFlexible
        The graph or an operation's attributes changed with the shape.
    """
    recorded: list[list[tuple[str, dict[str, object], list[tuple[int, ...]]]]] = []
    ids: list[list[int]] = []
    for shape in shapes:
        probe = lucid.zeros(*shape).to(example.dtype)
        graph, _feeds, _inputs, _outputs, _values = trace(
            model, probe, output_field=output_field
        )
        recorded.append(
            [
                (
                    op.name,
                    dict(op.attrs),
                    [tuple(int(d) for d in out.shape) for out in op.outputs],
                )
                for op in graph.ops
            ]
        )
        ids.append([out.id for op in graph.ops for out in op.outputs])

    base = recorded[0]
    for other, other_ids in zip(recorded[1:], ids[1:]):
        if len(other) != len(base) or [n for n, _a, _s in other] != [
            n for n, _a, _s in base
        ]:
            raise ShapeNotFlexible(
                "<graph>", "the traced operations themselves differ between shapes"
            )
        if other_ids != ids[0]:
            raise ShapeNotFlexible("<graph>", "the traced values differ between shapes")
        for (name, attrs, _shape), (_n, other_attrs, _s) in zip(base, other):
            if attrs != other_attrs:
                changed = sorted(
                    key
                    for key in set(attrs) | set(other_attrs)
                    if attrs.get(key) != other_attrs.get(key)
                )
                raise ShapeNotFlexible(
                    name, f"its {', '.join(changed)} changed with the input size"
                )

    varying: dict[int, set[int]] = {}
    position = 0
    for index, (_name, _attrs, base_shapes) in enumerate(base):
        for output, base_shape in enumerate(base_shapes):
            value_id = ids[0][position]
            position += 1
            axes = set()
            for other in recorded[1:]:
                other_shape = other[index][2][output]
                if len(other_shape) != len(base_shape):
                    raise ShapeNotFlexible(
                        base[index][0], "its result changes rank with the input size"
                    )
                axes |= {
                    axis
                    for axis, (a, b) in enumerate(zip(base_shape, other_shape))
                    if a != b
                }
            if axes:
                varying[value_id] = axes
    return varying


def _flex(shape: list[int], axes: set[int] | None) -> list[int]:
    """A shape with its varying axes left for Core ML to fill in."""
    if not axes:
        return list(shape)
    return [-1 if axis in axes else int(dim) for axis, dim in enumerate(shape)]


def _declare_image(
    program: Any, builder: Builder, name: str, shape: list[int], spec: ImageInput
) -> str:
    """Mark an input as an image and prepend its normalisation.

    Core ML puts the image *type* in the model description and the
    normalisation in the program: ``pixel * scale + bias`` is an ordinary
    ``mul`` and ``add`` at the head, before any cast to the body's
    precision, which is where a reference package puts them too.

    Parameters
    ----------
    program : MilProgram
        Program being written.
    builder : Builder
        Builder minting the constants.
    name : str
        Input feature name.
    shape : list[int]
        Input shape, which must be ``(1, C, H, W)``.
    spec : ImageInput
        Colour layout and normalisation.

    Returns
    -------
    str
        Value name the rest of the program should read.
    """
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(
            f"lucid.coreml: an image input must be (1, C, H, W), and this one is "
            f"{tuple(shape)}"
        )
    channels = shape[1]
    wanted = 1 if spec.color is ColorSpace.GRAYSCALE else 3
    if channels != wanted:
        raise ValueError(
            f"lucid.coreml: {spec.color.value} needs {wanted} channel(s) and the "
            f"input has {channels}"
        )
    if spec.bias and len(spec.bias) != channels:
        raise ValueError(
            f"lucid.coreml: bias has {len(spec.bias)} entries for {channels} channels"
        )

    program.set_image_input(name, shape[3], shape[2], _spec.color_space(spec.color))

    source = name
    interface = (_spec.FLOAT32, shape)
    if spec.scale != 1.0:
        scaled = "_image_scaled"
        program.add_op(
            "mul",
            _operands([("x", source), ("y", builder.const_float32(spec.scale))]),
            scaled,
            interface,
        )
        builder.shapes[scaled] = shape
        source = scaled
    if spec.bias:
        biased = "_image_biased"
        offsets = builder.const_float32_shaped(list(spec.bias), [1, channels, 1, 1])
        program.add_op(
            "add", _operands([("x", source), ("y", offsets)]), biased, interface
        )
        builder.shapes[biased] = shape
        source = biased
    return source


@dataclasses.dataclass
class _Shared:
    """A package and blob several functions are being written into.

    ``offsets`` maps a parameter's identity to where its bytes already
    are. Two entry points over the same module hold the same tensors, so
    keying on identity is what makes one copy serve both — which is the
    only reason to put them in one package rather than two.
    """

    paths: Any
    blob: Any
    offsets: dict[int, int] = dataclasses.field(default_factory=dict)


def build_package(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
    weights: WeightPrecision | _spec.Palettize | _spec.Sparsify = WeightPrecision.FLOAT,
    shapes: list[tuple[int, ...]] | None = None,
    shape_range: dict[int, tuple[int, int]] | None = None,
    state: list[State] | None = None,
    image_input: ImageInput | None = None,
    classifier: Classifier | None = None,
    metadata: Metadata | None = None,
    output_field: str | None = None,
    minimum_deployment_target: _spec.DeploymentTarget | None = None,
    into: _Shared | None = None,
) -> dict[str, object]:
    """Trace ``model`` and write a complete ``.mlpackage`` at ``path``.

    Parameters
    ----------
    model : nn.Module
        Model to export, in ``eval()`` mode.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Supplies each input's shape and dtype; values are irrelevant.
        A tuple is passed positionally, a mapping by keyword.
    path : str
        Destination package. Replaced if it already exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Body precision. ``FLOAT16`` is what the Neural Engine runs;
        inputs and outputs stay float32 either way, bracketed by casts.
    weights : WeightPrecision, optional, keyword-only, default=FLOAT
        How weights are stored. ``INT8`` keeps eight bits per weight plus
        a per-channel scale, halving the package against float16; the
        body still computes at ``precision``.
    image_input : ImageInput or None, optional, keyword-only, default=None
        Present the sole input as an image, with the normalisation it
        expects. Refused when the model takes more than one input, since
        which of them is the image would be a guess.
    shapes : list of tuple of int or None, optional, keyword-only, default=None
        Every input shape the package should accept, the example's own
        among them. Found by tracing at each and comparing, so an
        operation whose configuration came from the input size is
        refused by name rather than silently fixed to one of them.
    shape_range : dict of int to tuple of int or None, optional, keyword-only, default=None
        Axis to ``(lowest, highest)``, for an input whose size is not a
        short list — a variable sequence length, a camera's resolution.
        Axes left out keep the example's size. Mutually exclusive with
        ``shapes``, which admits only what it lists.
    state : list of State or None, optional, keyword-only, default=None
        Input/output pairs the package should carry between predictions
        instead of exchanging with the caller.
    classifier : Classifier or None, optional, keyword-only, default=None
        Declare the model a classifier over these labels. Needs a single
        output shaped ``(1, len(labels))``.
    metadata : Metadata or None, optional, keyword-only, default=None
        What the package says about itself.
    minimum_deployment_target : DeploymentTarget or None, optional, keyword-only, default=None
        Oldest system the package must run on. State, palettization and
        several entry points each raise that floor to ``IOS18``; naming a
        lower one refuses the export rather than producing a package that
        loads nowhere the caller intended. ``None`` accepts whatever the
        features require, and the result is reported on the model.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to export from an output dataclass. ``None``
        takes every tensor field it declares.
    into : _Shared or None, optional, keyword-only, default=None
        A package and weight blob already open, for building one function
        of several. The blob is neither finalised nor written out here;
        the caller does that once every function is in it, which is what
        lets them share the weights.

    Returns
    -------
    dict
        ``inputs`` and ``outputs`` — each a list of ``(feature name,
        shape)`` in declared order — plus ``ops``, ``precision`` and
        ``path``: what a caller needs to drive predictions without
        re-reading the package.

    Raises
    ------
    UnsupportedOp
        The trace contains an op with no MIL translation.
    """
    # Settled before anything is written: a floor the package cannot
    # meet is the caller's mistake to hear about now, not a device's to
    # report later.
    target = _settle_target(
        minimum_deployment_target,
        state=state,
        weights=weights,
        functions=into is not None,
    )
    graph, feeds, inputs, outputs, traced_values = trace(
        model, example, output_field=output_field
    )

    varying: dict[int, set[int]] = {}
    ordered: list[tuple[int, ...]] = []
    bounds: list[tuple[int, int]] = []
    if shapes is not None and shape_range is not None:
        raise ValueError(
            "lucid.coreml: shapes and shape_range say different things about the "
            "same input — a list of allowed shapes, or a range that admits "
            "everything between. Give one"
        )
    if shape_range is not None:
        if len(inputs) != 1:
            raise ValueError(
                f"lucid.coreml: a shape range needs a single-input model, and this "
                f"one takes {len(inputs)}"
            )
        default = tuple(int(d) for d in inputs[0][2].shape)
        for axis, (low, high) in sorted(shape_range.items()):
            if not 0 <= axis < len(default):
                raise ValueError(
                    f"lucid.coreml: axis {axis} is outside the input's rank "
                    f"{len(default)}"
                )
            if low > high:
                raise ValueError(
                    f"lucid.coreml: axis {axis} has a range of ({low}, {high})"
                )
            if not low <= default[axis] <= high:
                raise ValueError(
                    f"lucid.coreml: the example's axis {axis} is {default[axis]}, "
                    f"outside the range ({low}, {high}) it is meant to sit in"
                )
        bounds = [
            shape_range.get(axis, (size, size)) for axis, size in enumerate(default)
        ]
        # Trace at both ends as well as the default: within a range the
        # graph has to be the same at every size, and the ends are where a
        # dependence on the size shows up.
        corners = [
            tuple(low for low, _high in bounds),
            tuple(high for _low, high in bounds),
        ]
        probes = [default] + [c for c in corners if c != default]
        if len(probes) < 2:
            raise ValueError(
                "lucid.coreml: the range admits only the example's own shape "
                f"{default}"
            )
        varying = _varying_axes(model, inputs[0][2], probes, output_field)
        ranged_axes = {axis for axis, (low, high) in enumerate(bounds) if low != high}
        if ranged_axes:
            varying[inputs[0][1]] = ranged_axes
    if shapes is not None:
        if len(inputs) != 1:
            raise ValueError(
                f"lucid.coreml: flexible shapes need a single-input model, and this "
                f"one takes {len(inputs)}"
            )
        default = tuple(int(d) for d in inputs[0][2].shape)
        ordered = [default] + [tuple(s) for s in shapes if tuple(s) != default]
        if len(ordered) < 2:
            raise ValueError(
                "lucid.coreml: shapes must name at least one shape besides the "
                f"example's own {default}"
            )
        varying = _varying_axes(model, inputs[0][2], ordered, output_field)
        # The input is an external feed, not an operation's result, so it
        # is not in the map the trace comparison builds; its varying axes
        # are simply the ones the enumerated shapes disagree on.
        input_axes = {
            axis
            for axis in range(len(default))
            for alternative in ordered[1:]
            if alternative[axis] != default[axis]
        }
        if input_axes:
            varying[inputs[0][1]] = input_axes

    carried = {spec.input: spec for spec in (state or [])}
    if carried and precision is not Precision.FLOAT16:
        raise ValueError(
            "lucid.coreml: a carried state is stored as float16 — Core ML accepts "
            "no other type for one — so the body has to be float16 too, or the "
            "value written back would not be the value read. Pass "
            "precision=Precision.FLOAT16"
        )
    if carried:
        by_input = {name for name, _tid, _t in inputs}
        by_output = {field for field, _tid, _t in outputs}
        for spec in state or []:
            if spec.input not in by_input:
                raise ValueError(
                    f"lucid.coreml: {spec.input!r} is not an input of this model "
                    f"{sorted(by_input)}"
                )
            if spec.output not in by_output:
                raise ValueError(
                    f"lucid.coreml: {spec.output!r} is not an output of this model "
                    f"{sorted(by_output)}"
                )
            held = next(t for n, _i, t in inputs if n == spec.input)
            written = next(t for f, _i, t in outputs if f == spec.output)
            if tuple(held.shape) != tuple(written.shape):
                raise ValueError(
                    f"lucid.coreml: state {spec.input!r} is shaped "
                    f"{tuple(int(d) for d in held.shape)} and {spec.output!r} is "
                    f"{tuple(int(d) for d in written.shape)}; what is written back "
                    "has to be what was read"
                )

    plain_inputs = [entry for entry in inputs if entry[0] not in carried]
    if not plain_inputs:
        raise ValueError(
            "lucid.coreml: every input was declared state, leaving nothing for the "
            "caller to pass"
        )

    cm = _C_engine.coreml
    paths = into.paths if into is not None else cm.prepare_package(path)

    program = cm.MilProgram(
        [
            (
                name,
                (
                    _spec.mil_dtype(tensor.dtype),
                    _flex([int(d) for d in tensor.shape], varying.get(tid)),
                ),
            )
            for name, tid, tensor in plain_inputs
        ]
    )
    if shapes is not None:
        program.set_enumerated_shapes(inputs[0][0], [list(s) for s in ordered])
        program.set_default_shape(inputs[0][0], list(ordered[0]))
    if shape_range is not None:
        program.set_shape_range(inputs[0][0], bounds)
        program.set_default_shape(inputs[0][0], [int(d) for d in inputs[0][2].shape])
    names: dict[int, str] = {tid: name for name, tid, _t in inputs}
    builder_shapes_state: dict[str, list[int]] = {}
    input_ids = {tid for _n, tid, _t in inputs}

    # The body's precision, which is not necessarily the interface's.  The
    # Neural Engine only runs float16, so an fp32 program silently lands
    # on CPU or GPU no matter what compute units are requested; fp16 is
    # what actually reaches it.  Inputs and outputs stay float32 either
    # way, with casts bracketing the body, so callers are not asked to
    # hand over half precision.
    body_mil, body_blob = _spec.body_dtypes(precision)
    half = precision is Precision.FLOAT16
    for name, tid, tensor in inputs:
        if name not in carried:
            continue
        # Declared, then read at the head so the graph that follows sees an
        # ordinary value; the write-back is appended once the value it
        # stores exists.
        # The state lives at the body's precision, which Core ML requires
        # to be float16; reading it therefore needs no cast, unlike an
        # ordinary float input.
        carried_type = (body_mil, [int(d) for d in tensor.shape])
        program.add_state(name, carried_type)
        held = f"_state_{name}"
        program.read_state(name, held, carried_type)
        names[tid] = held
        builder_shapes_state[held] = [int(d) for d in tensor.shape]

    # Parameters and buffers become blob-backed constants.  The blob has
    # to be finalized before the protobuf that carries offsets into it is
    # written, which is why it is opened and closed here rather than by
    # the caller.
    blob = into.blob if into is not None else cm.BlobWriter(paths.weight_bin)
    weight_shapes: dict[str, list[int]] = {}
    quantized_count = 0
    for tid, impl in feeds.items():
        if tid in input_ids:
            continue
        tensor = _wrap(impl)
        _refuse_if_empty(tensor)
        is_float = tensor.dtype in (lucid.float32, lucid.float16)
        if half and is_float:
            tensor = tensor.half()
        # Float payloads go straight from the tensor's host storage — no
        # numpy anywhere in this package (H4).
        shape = [int(d) for d in tensor.shape]
        name = f"_w{tid}"
        palettized = (
            _palettize_weight(tensor, weights.bits)
            if is_float and isinstance(weights, _spec.Palettize)
            else None
        )
        sparse = (
            _sparsify_weight(tensor, weights.ratio)
            if is_float and isinstance(weights, _spec.Sparsify)
            else None
        )
        quantized = (
            _quantize_weight(tensor, body_mil)
            if is_float and weights is WeightPrecision.INT8
            else None
        )
        if palettized is not None:
            keys, palettes, lut_shape, key_bits = palettized
            offset = blob.append_bytes(
                keys, _C_engine.coreml.BLOB_SUBBYTE[str(key_bits)]
            )
            lut_offset = blob.append_tensor(
                _unwrap(palettes.half() if half else palettes), body_blob
            )
            program.add_grouped_lut_const(
                name,
                (body_mil, shape),
                offset,
                _C_engine.coreml.MIL_SUBBYTE[str(key_bits)],
                lut_offset,
                body_mil,
                lut_shape,
            )
            quantized_count += 1
        elif sparse is not None:
            values, mask, kept = sparse
            if half:
                values = values.half()
            value_offset = blob.append_tensor(_unwrap(values), body_blob)
            mask_offset = blob.append_bytes(mask, _spec.BLOB_UINT8)
            program.add_sparse_const(
                name, (body_mil, shape), value_offset, kept, mask_offset, len(mask)
            )
            quantized_count += 1
        elif quantized is not None:
            codes, scale_bytes, zero_bytes, channels = quantized
            offset = blob.append_tensor(_unwrap(codes), _spec.BLOB_INT8)
            program.add_quantized_const(
                name,
                (body_mil, shape),
                offset,
                scale_bytes,
                body_mil,
                zero_bytes,
                channels,
                0,
            )
            quantized_count += 1
        elif is_float:
            # Keyed by the parameter's identity, so a weight two
            # functions share is written once and pointed at twice.
            key = id(impl)
            offset = into.offsets.get(key) if into is not None else None
            if offset is None:
                offset = blob.append_tensor(_unwrap(tensor), body_blob)
                if into is not None:
                    into.offsets[key] = offset
            program.add_blob_const(name, (body_mil, shape), offset)
        elif tensor.dtype == lucid.bool_:
            # A boolean buffer is a mask, not a count that happens to be 0
            # or 1, and MIL types the two apart — an int32 constant would
            # be rejected wherever a condition is wanted.
            program.add_bool_const_shaped(name, _flatten_bools(tensor), shape)
        else:
            # Integer buffers (position ids, token-type ids) go inline: the
            # blob carries float payloads only, and MIL has an integer
            # tensor value already, so nothing has to be guessed.
            program.add_int_const_shaped(name, _flatten_ints(tensor), shape)
        names[tid] = name
        weight_shapes[name] = shape

    # Core ML's program dialect caps tensors at rank 5.  One shape reaches
    # past it for a living: the window partition every windowed-attention
    # transformer performs, which splits two spatial axes and permutes the
    # halves — ``(B, H, W, C)`` becomes ``(B, H/w, w, W/w, w, C)``, rank 6,
    # for exactly two operations before collapsing back.  Swin and MaxViT
    # are built out of it.
    #
    # That triple is rewritten rather than refused, so the rank-6 tensor is
    # never asked for; see ``_window_rewrites``.  Anything else that
    # exceeds the cap is still refused by name, because letting it through
    # produces an opaque parse failure from the compiler several steps away
    # from the operation that built the tensor.
    rewrites = _window_rewrites(graph)
    stuffing = _stuff_rewrites(graph)
    absorbed = {oid for r in rewrites.values() for oid in r.absorbs}
    absorbed |= {oid for r in stuffing.values() for oid in r.absorbs}
    # A third shape past the cap: values that are computed on at rank 6
    # rather than passed through it. Emitted a rank lower — see
    # ``_unit_axis_rewrites`` — so they never exist at rank 6 either.
    # Folding is decided first: it removes operations, and whether the
    # rank-6 component can be emitted a rank lower depends on which
    # operations are left to check.
    precomputed, folded = _foldable(graph, feeds, set(input_ids), traced_values)
    thinned = _unit_axis_rewrites(
        graph, set(rewrites) | set(stuffing) | absorbed, folded
    )
    _pending_thinned = thinned
    for op in graph.ops:
        head = int(op.outputs[0].id) if op.outputs else -1
        if head in rewrites or head in stuffing or head in absorbed:
            continue
        for out in op.outputs:
            if len(out.shape) > _MAX_RANK and out.id not in thinned:
                raise UnsupportedRank(op.name, tuple(int(d) for d in out.shape))

    builder = Builder(program, blob, body_mil, body_blob, half)
    builder.varying = varying
    builder.thinned = set(_pending_thinned)
    builder.shapes.update(weight_shapes)
    builder.shapes.update(builder_shapes_state)
    # Weights are constants whichever way they were stored — plain, int8,
    # palettized or sparse. Emitters that must bind a constant ask here.
    for weight_name in weight_shapes:
        builder.mark_const(weight_name)
        builder.dtypes[weight_name] = body_mil
    if image_input is not None and len(inputs) != 1:
        raise ValueError(
            f"lucid.coreml: image_input needs a single-input model, and this one "
            f"takes {len(inputs)} — which of them is the image would be a guess"
        )
    for name, tid, tensor in plain_inputs:
        shape = [int(d) for d in tensor.shape]
        builder.shapes[name] = shape
        builder.dtypes[name] = _spec.mil_dtype(tensor.dtype)
        source = name
        if image_input is not None:
            source = _declare_image(program, builder, name, shape, image_input)
            names[tid] = source
        # Only a float interface needs bracketing.  An integer input —
        # token ids — must reach its lookup as an integer; casting it to
        # half would turn indices into approximations of themselves.
        if half and tensor.dtype in (lucid.float32, lucid.float16):
            cast_name = f"_cast_in_{name}"
            mil_type, raw = emit_cast(builder, source, "fp16")
            program.add_op(
                mil_type,
                _operands(raw),
                cast_name,
                (body_mil, _flex(shape, varying.get(tid))),
            )
            builder.shapes[cast_name] = shape
            names[tid] = cast_name
    # An output's value is named after the model's own field, so a caller
    # reading the package knows which head it is looking at.  Done when the
    # producing op is emitted rather than by appending an identity to
    # rename it — an extra operation would land on the CPU and show up in
    # every compute plan.  Under fp16 the cast that follows takes the name
    # instead, since it is what the interface actually returns.
    wanted_name: dict[int, str] = (
        {} if half else {tid: field for field, tid, _t in outputs}
    )
    # A value the graph recomputes on every prediction although nothing
    # about it depends on the input is written down instead. See
    # ``_foldable``: this is a third of a windowed transformer, and all
    # of it lands on the CPU.
    for tid in precomputed:
        folded_value = _wrap(traced_values[tid])
        if tid in thinned:
            # Written a rank lower, like everything else in that
            # component; the leading axis is one by construction.
            folded_value = folded_value.reshape(
                *[int(d) for d in folded_value.shape][1:]
            )
        if folded_value.dtype in (lucid.float32, lucid.float16):
            names[tid] = builder.const_from_tensor(
                folded_value.half() if half else folded_value
            )
        elif folded_value.dtype in (lucid.int32, lucid.int64):
            names[tid] = builder.const_ints_shaped(
                [int(v) for v in _flatten_ints(folded_value)],
                [int(d) for d in folded_value.shape],
            )
        else:
            # Nothing to write it as; leave the operations in place.
            folded = set()
            precomputed = set()
            break

    for op in _reachable_ops(graph, [tid for _n, tid, _t in outputs]):
        if op.outputs and int(op.outputs[0].id) in folded:
            # Its result is a constant now, and so is everything it fed.
            continue
        head = int(op.outputs[0].id) if op.outputs else -1
        if head in absorbed:
            # Emitted already, as part of the window rewrite below.
            continue
        plan = rewrites.get(head)
        if plan is not None:
            names[plan.out_id] = _stage_permutation(builder, names[op.inputs[0]], plan)
            continue
        stuff = stuffing.get(head)
        if stuff is not None:
            names[stuff.out_id] = _stage_zero_stuff(builder, names[op.inputs[0]], stuff)
            continue
        emitter = EMITTERS.get(op.name)
        if emitter is None:
            raise UnsupportedOp(op.name)
        operands = [names[i] for i in op.inputs]
        result = emitter(builder, op, operands)
        if isinstance(result, Constant):
            # The value is the constant; there is nothing to append.
            names[op.outputs[0].id] = result.name
            continue
        if isinstance(result, Bound):
            # The emitter already appended whatever it needed; these are
            # the values its results are.
            for out, bound in zip(op.outputs, result.names):
                names[out.id] = bound
            continue
        multi = isinstance(result, MultiOutput)
        if isinstance(result, MultiOutput):
            mil_type, raw_bindings = result.mil_type, result.bindings
        else:
            mil_type, raw_bindings = result
        # Emitters may bind a parameter to one name or several (``concat``);
        # the engine wants a list either way.
        bindings = _operands(raw_bindings)
        produced = [
            (
                wanted_name.get(o.id, f"_v{o.id}_{op.name}"),
                (
                    _spec.trace_dtype(str(o.dtype).split(".")[-1], body_mil),
                    _flex(
                        (
                            [int(d) for d in o.shape][1:]
                            if o.id in thinned
                            else [int(d) for d in o.shape]
                        ),
                        varying.get(o.id),
                    ),
                ),
            )
            for o in (op.outputs if multi else op.outputs[:1])
        ]
        if multi:
            program.add_op_multi(mil_type, bindings, produced)
        else:
            program.add_op(mil_type, bindings, produced[0][0], produced[0][1])
        for (out_name, (out_dtype, shape)), o in zip(produced, op.outputs):
            names[o.id] = out_name
            builder.shapes[out_name] = shape
            builder.dtypes[out_name] = out_dtype
            if o.id in thinned:
                builder.unit_axis_dropped.add(out_name)

    if into is None:
        blob.finalize()

    if classifier is not None:
        if len(outputs) != 1:
            raise ValueError(
                f"lucid.coreml: a classifier needs one output and this model "
                f"returns {len(outputs)} ({[name for name, _t, _x in outputs]})"
            )
        scores = outputs[0][2]
        score_shape = tuple(int(d) for d in scores.shape)
        if len(score_shape) != 2 or score_shape[0] != 1:
            raise ValueError(
                f"lucid.coreml: a classifier's output must be (1, classes) and this "
                f"one is {score_shape}"
            )
        if score_shape[1] != len(classifier.labels):
            raise ValueError(
                f"lucid.coreml: {len(classifier.labels)} labels for "
                f"{score_shape[1]} scores"
            )

    written = {spec.output: spec.input for spec in (state or [])}
    declared: list[tuple[str, tuple[int, ...]]] = []
    for field, tid, tensor in outputs:
        if field in written:
            # The caller does not receive it; Core ML keeps it.
            program.write_state(written[field], names[tid])
            continue
        value = names[tid]
        flexible_type = (
            _spec.mil_dtype(tensor.dtype),
            _flex([int(d) for d in tensor.shape], varying.get(tid)),
        )
        # Only a float output needs bracketing back to fp32, and the
        # guard has to be here as well as on the input side: casting an
        # integer output — VQ-VAE returns its codebook indices — asks
        # Core ML for a cast whose named type is not the output's, and
        # the package will not parse.
        if half and tensor.dtype in (lucid.float32, lucid.float16):
            mil_type, raw = emit_cast(builder, value, "fp32")
            program.add_op(mil_type, _operands(raw), field, flexible_type)
            value = field
        if value != field:
            # Reachable only if the producing op was shared between two
            # declared outputs, which the naming pass cannot satisfy twice.
            program.add_op("identity", [("x", [value])], field, flexible_type)
        if classifier is not None:
            # The scores stop being the model's output; the label and the
            # probability map take their place.
            program.set_classifier(
                value,
                list(classifier.labels),
                classifier.label_name,
                classifier.probabilities_name,
            )
            declared.append((classifier.label_name, ()))
            declared.append((classifier.probabilities_name, ()))
            continue
        program.add_output(field, flexible_type)
        declared.append((field, tuple(int(d) for d in tensor.shape)))
    if metadata is not None:
        program.set_metadata(
            metadata.description, metadata.author, metadata.license, metadata.version
        )
    if into is None:
        cm.finish_package(paths, program.serialize())

    return {
        "inputs": [
            (name, tuple(int(d) for d in tensor.shape))
            for name, _tid, tensor in plain_inputs
        ],
        "outputs": declared,
        "ops": int(program.op_count),
        "precision": precision.value,
        # A name for the summary: the enum has one, and the two
        # parameterised forms are described by the parameter that makes
        # them different.
        "weights": (
            weights.value
            if isinstance(weights, WeightPrecision)
            else (
                f"PALETTIZED_{weights.bits}BIT"
                if isinstance(weights, _spec.Palettize)
                else f"SPARSE_{weights.ratio:g}"
            )
        ),
        "classifier": classifier is not None,
        "quantized_weights": quantized_count,
        "flexible": shapes is not None or shape_range is not None,
        "state": [(spec.input, spec.output) for spec in (state or [])],
        "deployment_target": target,
        "program": program,
        "path": paths.root,
    }


# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call.
from lucid._tensor.tensor import Tensor  # noqa: E402,F811
from lucid.nn.module import Module  # noqa: E402,F811
