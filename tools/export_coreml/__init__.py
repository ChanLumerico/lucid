"""Export a Lucid model to Core ML (``.mlpackage``).

Why this exists: Lucid runs on Accelerate and MLX, and neither of those
touches the Neural Engine.  Core ML is the only public route to the ANE,
and it is also how a model trained here gets into an iOS or macOS app at
all.  Everything else Lucid can already export is weights
(``save_pretrained`` / safetensors) or a graph that only Lucid's own
runtime can execute.

Where it lives: ``tools/`` rather than ``lucid/``, exactly like
``tools/convert_weights``.  ``coremltools`` is a developer dependency,
and putting the exporter outside the package keeps H4's no-external-imports
rule untouched instead of negotiating a seventh bridge boundary for it.
Install with ``pip install lucid-dl[coreml]``.

How it works: the compile tracer already produces a complete graph —
``lucid.compile`` lowers that same graph into MPSGraph — so this is a
second backend for an existing IR, not a new front end.  The model is
traced once, its parameters become MIL constants, and each op is
translated by ``_emit.EMITTERS``.

Coverage is deliberately narrow and loud.  The ten ops mapped today are
the ones four real classifiers actually emit (ResNet, MobileNetV2, VGG,
AlexNet — measured, not guessed); anything else raises with the op's name
rather than producing a model that is quietly missing a layer.

Usage
-----
::

    from tools.export_coreml import export

    model = lucid.models.create_model("resnet_18_cls").eval()
    export(model, lucid.zeros(1, 3, 224, 224), "/tmp/resnet18.mlpackage")

or from the command line::

    python -m tools.export_coreml resnet_18_cls --out /tmp/resnet18.mlpackage
"""

from __future__ import annotations  # tooling only — outside lucid/ (H1 OK)

from typing import Any

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb

import lucid
import lucid.compile as lc
from lucid._dispatch import _unwrap, _wrap

from tools.export_coreml._emit import EMITTERS

__all__ = ["export", "trace_graph", "UnsupportedOp"]


class UnsupportedOp(NotImplementedError):
    """A traced op has no MIL translation.

    Carries the op name so the message names the gap instead of failing
    somewhere downstream with a shape error.
    """

    def __init__(self, op_name: str) -> None:
        super().__init__(
            f"export_coreml: no Core ML translation for Lucid op {op_name!r}. "
            f"Mapped ops: {', '.join(sorted(EMITTERS))}. Add an emitter in "
            f"tools/export_coreml/_emit.py."
        )
        self.op_name = op_name


def _select_output(output: Any, field: str | None) -> Any:
    """Reduce a model's return value to the one tensor to export.

    Zoo models return an output dataclass rather than a bare tensor —
    ``ImageClassificationOutput`` and friends — so exporting one would
    otherwise mean asking every caller to wrap the head. The convention
    across those classes is that the payload lives on ``logits``; a model
    with a different shape of answer names its field explicitly.
    """
    if isinstance(output, lucid.Tensor):
        return output
    if field is not None:
        picked = getattr(output, field, None)
        if not isinstance(picked, lucid.Tensor):
            raise TypeError(
                f"export_coreml: {type(output).__name__}.{field} is "
                f"{type(picked).__name__}, not a Tensor"
            )
        return picked
    picked = getattr(output, "logits", None)
    if isinstance(picked, lucid.Tensor):
        return picked
    raise TypeError(
        f"export_coreml: the model returned {type(output).__name__}, which has "
        "no ``logits``. Name the field to export with output_field=..., or "
        "wrap the head so it returns a single Tensor."
    )


def trace_graph(
    model: Any, example_input: Any, *, output_field: str | None = None
) -> tuple[Any, Any, int, int]:
    """Trace ``model`` once and locate its input and output in the graph.

    Returns
    -------
    tuple
        ``(graph, external_feeds, input_id, output_id)``.
    """
    with lc._tracing() as tracer:
        output = _select_output(model(example_input), output_field)
    input_id = tracer.lookup_id(_unwrap(example_input))
    output_id = tracer.lookup_id(_unwrap(output))
    if input_id is None:
        raise ValueError(
            "export_coreml: the example input never reached an op — the "
            "model ignored it, or it was copied before use"
        )
    if output_id is None:
        raise ValueError("export_coreml: the model's output is not part of the trace")
    return tracer.graph, tracer.external_feeds, input_id, output_id


def export(
    model: Any,
    example_input: Any,
    path: str,
    *,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    compute_precision: ct.precision = ct.precision.FLOAT32,
    minimum_deployment_target: ct.target = ct.target.macOS14,
    output_field: str | None = None,
    author: str = "Lucid",
) -> Any:
    """Convert ``model`` to a Core ML package at ``path``.

    Parameters
    ----------
    model : nn.Module
        Must be in ``eval()`` mode — an exported graph is an inference
        graph, and a training-mode dropout is refused rather than
        silently dropped.
    example_input : Tensor
        Shape and dtype come from this; its values do not matter. The
        exported model's input shape is fixed to it.
    path : str
        Destination ``.mlpackage``.
    compute_units : ct.ComputeUnit, optional, keyword-only, default=ALL
        Which processors Core ML may use. ``ct.ComputeUnit.CPU_AND_NE``
        pins execution to the Neural Engine plus CPU — the reason this
        exporter exists.
    compute_precision : ct.precision, optional, keyword-only, default=FLOAT32
        Weight and activation precision in the exported graph. The
        default keeps the export faithful to the Lucid model it came from
        (agreement is then ~1e-6 rather than ~1e-3). Core ML's own
        default is FLOAT16, which is also what the Neural Engine wants —
        pass ``ct.precision.FLOAT16`` when targeting it, and expect the
        looser agreement that comes with it.
    minimum_deployment_target : ct.target, optional, keyword-only, default=macOS14
        Oldest OS the package must load on.
    output_field : str or None, optional, keyword-only, default=None
        Attribute to export when the model returns an output dataclass.
        ``None`` takes ``logits``, which is what the zoo's output classes
        carry.
    author : str, optional, keyword-only, default="Lucid"
        Written into the package metadata.

    Returns
    -------
    ct.models.MLModel
        The converted model, already saved to ``path``.

    Raises
    ------
    UnsupportedOp
        The trace contains an op with no MIL translation.
    """
    if getattr(model, "training", False):
        raise ValueError(
            "export_coreml: model is in training mode; call model.eval() first"
        )

    graph, feeds, input_id, output_id = trace_graph(
        model, example_input, output_field=output_field
    )

    # Every external feed except the input is a parameter or a buffer, so
    # it becomes a constant in the exported graph.  Reading them here
    # (rather than from ``state_dict``) keeps the values keyed by the
    # trace ids the ops actually reference.
    consts: dict[int, np.ndarray] = {}
    for tid, impl in feeds.items():
        if tid == input_id:
            continue
        consts[tid] = _wrap(impl).numpy()

    in_shape = tuple(int(d) for d in example_input.shape)

    @mb.program(input_specs=[mb.TensorSpec(shape=in_shape)])  # type: ignore[misc]
    def program(x: Any) -> Any:
        env: dict[int, Any] = {input_id: x}
        for tid, value in consts.items():
            env[tid] = mb.const(val=value, name=f"const_{tid}")
        for op in graph.ops:
            emitter = EMITTERS.get(op.name)
            if emitter is None:
                raise UnsupportedOp(op.name)
            operands = [env[i] for i in op.inputs]
            result = emitter(op, operands)
            env[op.outputs[0].id] = result
        return env[output_id]

    mlmodel = ct.convert(
        program,
        compute_units=compute_units,
        compute_precision=compute_precision,
        minimum_deployment_target=minimum_deployment_target,
    )
    mlmodel.author = author
    mlmodel.short_description = f"{type(model).__name__} exported from Lucid"
    mlmodel.save(path)
    return mlmodel
