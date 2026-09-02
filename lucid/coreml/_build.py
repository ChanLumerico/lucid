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

import lucid
import lucid.compile as _compile
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.coreml import _spec
from lucid.coreml._emit import EMITTERS, MultiOutput, emit_cast
from lucid.coreml._spec import Precision

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["Builder", "UnsupportedOp", "build_package", "trace"]


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


def _select_output(output: object, field: str | None) -> "Tensor":
    """Reduce a model's return value to the single tensor to export."""
    if isinstance(output, lucid.Tensor):
        return output
    name = field if field is not None else "logits"
    picked = getattr(output, name, None)
    if isinstance(picked, lucid.Tensor):
        return picked
    if field is not None:
        raise TypeError(
            f"lucid.coreml: {type(output).__name__}.{field} is "
            f"{type(picked).__name__}, not a Tensor"
        )
    raise TypeError(
        f"lucid.coreml: the model returned {type(output).__name__}, which has no "
        "``logits``. Name the field with output_field=..., or return a Tensor."
    )


def trace(
    model: "Module", example: "Tensor", *, output_field: str | None = None
) -> Any:
    """Run one traced forward pass.

    Parameters
    ----------
    model : nn.Module
        Model to trace. Should already be in ``eval()`` mode.
    example : Tensor
        Input to trace with; its identity is how the driver tells the
        argument apart from the parameters, which are external feeds too.
    output_field : str or None, optional, keyword-only, default=None
        Attribute to take when the model returns an output dataclass.
        ``None`` takes ``logits``.

    Returns
    -------
    tuple
        ``(graph, external_feeds, input_id, output_id, output_tensor)``.

    Raises
    ------
    ValueError
        The input never reached an op, or the output is not in the trace.
    TypeError
        The return value is neither a Tensor nor carries the named field.
    """
    with _compile._tracing() as tracer:
        result = _select_output(model(example), output_field)
    input_id = tracer.lookup_id(_unwrap(example))
    output_id = tracer.lookup_id(_unwrap(result))
    if input_id is None:
        raise ValueError(
            "lucid.coreml: the example input never reached an op — the model "
            "ignored it, or copied it before use"
        )
    if output_id is None:
        raise ValueError("lucid.coreml: the model's output is not part of the trace")
    return tracer.graph, tracer.external_feeds, input_id, output_id, result


def _flatten_ints(tensor: "Tensor") -> list[int]:
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
            flat.append(int(value))  # type: ignore[arg-type]

    walk(tensor.tolist())
    return flat


class Builder:
    """Mints the constants MIL requires and appends operations.

    Every scalar operand in a MIL program is its own ``const`` operation —
    a convolution's ``groups`` is a value, not an attribute — so emitters
    ask the builder for names rather than writing literals.
    """

    def __init__(
        self,
        program: object,
        blob: object = None,
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

    def const_str(self, value: str) -> str:
        name = self._next("str")
        self._program.add_string_const(name, value)
        return name

    def const_from_tensor(self, tensor: "Tensor") -> str:
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

    def emit(self, mil_type: str, bindings: list, shape: list[int]) -> str:
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
        normalised = [
            (param, [value] if isinstance(value, str) else list(value))
            for param, value in bindings
        ]
        self._program.add_op(mil_type, normalised, name, (self._body_mil, list(shape)))
        self.shapes[name] = list(shape)
        return name

    def const_bool(self, value: bool) -> str:
        name = self._next("bool")
        self._program.add_bool_const(name, bool(value))
        return name


def build_package(
    model: "Module",
    example: "Tensor",
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
    example : Tensor
        Supplies the input shape and dtype; values are irrelevant.
    path : str
        Destination package. Replaced if it already exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Body precision. ``FLOAT16`` is what the Neural Engine runs;
        inputs and outputs stay float32 either way, bracketed by casts.
    output_field : str or None, optional, keyword-only, default=None
        Attribute to export from an output dataclass. ``None`` takes
        ``logits``.

    Returns
    -------
    dict
        ``input`` / ``output`` feature names, ``input_shape`` /
        ``output_shape``, ``ops``, ``precision`` and ``path`` — what a
        caller needs to drive predictions without re-reading the package.

    Raises
    ------
    UnsupportedOp
        The trace contains an op with no MIL translation.
    """
    graph, feeds, input_id, output_id, output_tensor = trace(
        model, example, output_field=output_field
    )

    cm = _C_engine.coreml
    paths = cm.prepare_package(path)

    input_name = "input"
    program = cm.MilProgram(input_name, _spec.type_spec(example))
    names: dict[int, str] = {input_id: input_name}

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
        if tid == input_id:
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
        else:
            # Integer buffers (position ids, token-type ids) go inline: the
            # blob carries float payloads only, and MIL has an integer
            # tensor value already, so nothing has to be guessed.
            program.add_int_const_shaped(name, _flatten_ints(tensor), shape)
        names[tid] = name
        weight_shapes[name] = shape

    builder = Builder(program, blob, body_mil, body_blob, half)
    builder.shapes.update(weight_shapes)
    builder.shapes[input_name] = [int(d) for d in example.shape]
    # Only a float interface needs bracketing.  An integer input — token
    # ids — must reach its lookup as an integer; casting it to half would
    # turn indices into approximations of themselves.
    float_input = example.dtype in (lucid.float32, lucid.float16)
    if half and float_input:
        cast_name = "_cast_in"
        mil_type, raw = emit_cast(builder, input_name, "fp16")
        bindings = [(p_, [v]) for p_, v in raw]
        program.add_op(
            mil_type, bindings, cast_name, (body_mil, [int(d) for d in example.shape])
        )
        names[input_id] = cast_name
    for op in graph.ops:
        emitter = EMITTERS.get(op.name)
        if emitter is None:
            raise UnsupportedOp(op.name)
        operands = [names[i] for i in op.inputs]
        result = emitter(builder, op, operands)
        multi = isinstance(result, MultiOutput)
        mil_type, raw_bindings = (result.mil_type, result.bindings) if multi else result
        # Emitters may bind a parameter to one name or several (``concat``);
        # the engine wants a list either way.
        bindings = [
            (param, [value] if isinstance(value, str) else list(value))
            for param, value in raw_bindings
        ]
        outputs = [
            (
                f"_v{o.id}_{op.name}",
                (
                    _spec.trace_dtype(str(o.dtype).split(".")[-1], body_mil),
                    [int(d) for d in o.shape],
                ),
            )
            for o in (op.outputs if multi else op.outputs[:1])
        ]
        if multi:
            program.add_op_multi(mil_type, bindings, outputs)
        else:
            program.add_op(mil_type, bindings, outputs[0][0], outputs[0][1])
        for (out_name, (_dt, shape)), o in zip(outputs, op.outputs):
            names[o.id] = out_name
            builder.shapes[out_name] = shape

    blob.finalize()

    output_name = names[output_id]
    if half:
        cast_out = "_cast_out"
        mil_type, raw = emit_cast(builder, output_name, "fp32")
        bindings = [(p_, [v]) for p_, v in raw]
        program.add_op(mil_type, bindings, cast_out, _spec.type_spec(output_tensor))
        output_name = cast_out
    program.set_output(output_name, _spec.type_spec(output_tensor))
    cm.finish_package(paths, program.serialize())

    return {
        "input": input_name,
        "output": output_name,
        "input_shape": tuple(int(d) for d in example.shape),
        "output_shape": tuple(int(d) for d in output_tensor.shape),
        "ops": int(program.op_count),
        "precision": precision.value,
        "path": paths.root,
    }
