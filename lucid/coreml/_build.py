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

from typing import TYPE_CHECKING, Any

import dataclasses

import lucid
import lucid.compile as _compile
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.coreml import _spec
from lucid.coreml._emit import (
    EMITTERS,
    Bindings,
    MultiOutput,
    _as_int,
    emit_cast,
)
from lucid.coreml._spec import Precision

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
    """A traced op has no MIL translation.

    Named rather than generic because the alternative failure — emitting
    a package that quietly lacks a layer — produces a model that loads
    and returns plausible numbers.
    """

    def __init__(self, op_name: str) -> None:
        super().__init__(
            f"lucid.coreml: no Core ML translation for Lucid op {op_name!r}. "
            f"Mapped ops: {', '.join(sorted(EMITTERS))}. Add an emitter in "
            f"lucid/coreml/_emit.py."
        )
        self.op_name = op_name


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
    with _compile._tracing() as tracer:
        if by_keyword:
            result = model(**dict(examples))
        else:
            result = model(*(tensor for _, tensor in examples))
        selected = _select_outputs(result, output_field)

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
                f"lucid.coreml: output {name!r} is not part of the trace"
            )
        outputs.append((name, value_id, tensor))
    return tracer.graph, tracer.external_feeds, inputs, outputs


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

    def const_ints(self, values: list[int]) -> str:
        name = self._next("ints")
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
        name = self._next("intsn")
        self._program.add_int_const_shaped(name, [int(v) for v in values], shape)
        self.shapes[name] = list(shape)
        return name

    def const_int(self, value: int) -> str:
        name = self._next("int")
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
        name = self._next("float")
        if not self._half:
            self._program.add_float_const(name, [float(value)], True)
            return name
        payload = lucid.tensor([float(value)]).half()
        offset = self._blob.append_tensor(_unwrap(payload), self._body_blob)
        # Declared scalar (rank 0) even though the payload is one element:
        # MIL types ``epsilon`` as a scalar and rejects tensor<fp16, [1]>.
        self._program.add_blob_const(name, (self._body_mil, []), offset)
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
        name = self._next("f32")
        self._program.add_float_const(name, [float(value)], True)
        return name

    def const_str(self, value: str) -> str:
        name = self._next("str")
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
        payload = tensor.half() if self._half else tensor
        name = self._next("blobconst")
        offset = self._blob.append_tensor(_unwrap(payload), self._body_blob)
        shape = [int(d) for d in tensor.shape]
        self._program.add_blob_const(name, (self._body_mil, shape), offset)
        self.shapes[name] = shape
        return name

    def emit(self, mil_type: str, bindings: Bindings, shape: list[int]) -> str:
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
            Result shape. The dtype is the program body's.

        Returns
        -------
        str
            Name of the value this operation produces.
        """
        name = self._next(mil_type)
        normalised = _operands(bindings)
        self._program.add_op(mil_type, normalised, name, (self._body_mil, list(shape)))
        self.shapes[name] = list(shape)
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
        name = self._next("bool")
        self._program.add_bool_const(name, bool(value))
        return name


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


def build_package(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
    output_field: str | None = None,
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
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to export from an output dataclass. ``None``
        takes every tensor field it declares.

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
    graph, feeds, inputs, outputs = trace(
        model, example, output_field=output_field
    )

    cm = _C_engine.coreml
    paths = cm.prepare_package(path)

    program = cm.MilProgram(
        [(name, _spec.type_spec(tensor)) for name, _tid, tensor in inputs]
    )
    names: dict[int, str] = {tid: name for name, tid, _t in inputs}
    input_ids = {tid for _n, tid, _t in inputs}

    # The body's precision, which is not necessarily the interface's.  The
    # Neural Engine only runs float16, so an fp32 program silently lands
    # on CPU or GPU no matter what compute units are requested; fp16 is
    # what actually reaches it.  Inputs and outputs stay float32 either
    # way, with casts bracketing the body, so callers are not asked to
    # hand over half precision.
    body_mil, body_blob = _spec.body_dtypes(precision)
    half = precision is Precision.FLOAT16

    # Parameters and buffers become blob-backed constants.  The blob has
    # to be finalized before the protobuf that carries offsets into it is
    # written, which is why it is opened and closed here rather than by
    # the caller.
    blob = cm.BlobWriter(paths.weight_bin)
    weight_shapes: dict[str, list[int]] = {}
    for tid, impl in feeds.items():
        if tid in input_ids:
            continue
        tensor = _wrap(impl)
        is_float = tensor.dtype in (lucid.float32, lucid.float16)
        if half and is_float:
            tensor = tensor.half()
        # Float payloads go straight from the tensor's host storage — no
        # numpy anywhere in this package (H4).
        shape = [int(d) for d in tensor.shape]
        name = f"_w{tid}"
        if is_float:
            offset = blob.append_tensor(_unwrap(tensor), body_blob)
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

    # Core ML's program dialect caps tensors at rank 5.  Catching it here
    # names the constraint and the offending shape; letting it through
    # produces an opaque parse failure from the compiler instead, several
    # steps away from the operation that built the tensor.
    for op in graph.ops:
        for out in op.outputs:
            if len(out.shape) > _MAX_RANK:
                raise UnsupportedRank(op.name, tuple(int(d) for d in out.shape))

    builder = Builder(program, blob, body_mil, body_blob, half)
    builder.shapes.update(weight_shapes)
    for name, tid, tensor in inputs:
        shape = [int(d) for d in tensor.shape]
        builder.shapes[name] = shape
        # Only a float interface needs bracketing.  An integer input —
        # token ids — must reach its lookup as an integer; casting it to
        # half would turn indices into approximations of themselves.
        if half and tensor.dtype in (lucid.float32, lucid.float16):
            cast_name = f"_cast_in_{name}"
            mil_type, raw = emit_cast(builder, name, "fp16")
            program.add_op(mil_type, _operands(raw), cast_name, (body_mil, shape))
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
    for op in _reachable_ops(graph, [tid for _n, tid, _t in outputs]):
        emitter = EMITTERS.get(op.name)
        if emitter is None:
            raise UnsupportedOp(op.name)
        operands = [names[i] for i in op.inputs]
        result = emitter(builder, op, operands)
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
                    [int(d) for d in o.shape],
                ),
            )
            for o in (op.outputs if multi else op.outputs[:1])
        ]
        if multi:
            program.add_op_multi(mil_type, bindings, produced)
        else:
            program.add_op(mil_type, bindings, produced[0][0], produced[0][1])
        for (out_name, (_dt, shape)), o in zip(produced, op.outputs):
            names[o.id] = out_name
            builder.shapes[out_name] = shape

    blob.finalize()

    declared: list[tuple[str, tuple[int, ...]]] = []
    for field, tid, tensor in outputs:
        value = names[tid]
        if half:
            mil_type, raw = emit_cast(builder, value, "fp32")
            program.add_op(mil_type, _operands(raw), field, _spec.type_spec(tensor))
            value = field
        if value != field:
            # Reachable only if the producing op was shared between two
            # declared outputs, which the naming pass cannot satisfy twice.
            program.add_op(
                "identity", [("x", [value])], field, _spec.type_spec(tensor)
            )
        program.add_output(field, _spec.type_spec(tensor))
        declared.append((field, tuple(int(d) for d in tensor.shape)))
    cm.finish_package(paths, program.serialize())

    return {
        "inputs": [
            (name, tuple(int(d) for d in tensor.shape)) for name, _tid, tensor in inputs
        ],
        "outputs": declared,
        "ops": int(program.op_count),
        "precision": precision.value,
        "path": paths.root,
    }
