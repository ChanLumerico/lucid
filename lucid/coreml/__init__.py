"""``lucid.coreml`` — export Lucid models to Core ML, and run them on the ANE.

Lucid computes on Accelerate (CPU) and MLX (Metal).  Neither targets the
Neural Engine, so a whole processor on every machine Lucid supports was
unreachable; the same gap meant a model trained here could not ship inside
an iOS or macOS app.  Core ML is the only public route to either.

**No third-party dependency.**  The package format is written by
:mod:`lucid._C.coreml` — the MIL protobuf, the weight blob, the bundle —
and executed through Apple's own CoreML.framework, which stands beside
Accelerate and Metal rather than beside a pip package.  Nothing under
``lucid/`` imports anything external (H4).

**Two things worth knowing before trusting an export.**

*The Neural Engine only runs float16.*  A float32 program asked for with
``ComputeUnits.CPU_AND_NE`` does not warn, does not error, and runs at CPU
speed — measured: zero of its operations are scheduled on the ANE.  Pass
``Precision.FLOAT16`` to actually reach it, and expect the ~1e-4 that
half precision costs.  :meth:`CoreMLModel.compute_plan` reports where the
operations really landed, so this is checkable rather than inferred from a
stopwatch.

*Coverage is narrow and loud.*  The mapped operations are the ones real
models were measured to emit; anything else raises
:class:`UnsupportedOp` naming the gap, because a package quietly missing a
layer still loads and still returns plausible numbers.

Examples
--------
::

    import lucid, lucid.coreml as cml

    model = lucid.models.create_model("resnet_18_cls").eval()
    x = lucid.zeros(1, 3, 224, 224)

    cm = cml.export(model, x, "/tmp/r18.mlpackage",
                    precision=cml.Precision.FLOAT16,
                    compute_units=cml.ComputeUnits.CPU_AND_NE)
    cm.verify(model, x)        # max|coreml - eager|
    cm.compute_plan()          # PlacementSummary(ANE=69, CPU=2, ane=97%)
    y = cm.predict(x)
"""

from typing import TYPE_CHECKING

from lucid.coreml._build import UnsupportedOp, UnsupportedRank, build_package
from lucid.coreml._model import CoreMLModel, PlacementSummary
from lucid.coreml._spec import (
    Classifier,
    ColorSpace,
    ComputeUnits,
    ImageInput,
    Metadata,
    Precision,
    WeightPrecision,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = [
    "Classifier",
    "ColorSpace",
    "ComputeUnits",
    "ImageInput",
    "Metadata",
    "WeightPrecision",
    "CoreMLModel",
    "PlacementSummary",
    "Precision",
    "UnsupportedOp",
    "UnsupportedRank",
    "export",
    "load",
]


def _features(value: object) -> list[tuple[str, tuple[int, ...]]]:
    """The feature names and shapes as ``build_package`` reports them."""
    if not isinstance(value, list):
        raise TypeError(
            f"lucid.coreml: expected a feature list, got {type(value).__name__}"
        )
    return [(str(name), tuple(int(d) for d in shape)) for name, shape in value]


def export(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
    weights: WeightPrecision = WeightPrecision.FLOAT,
    image_input: ImageInput | None = None,
    classifier: Classifier | None = None,
    metadata: Metadata | None = None,
    compute_units: ComputeUnits = ComputeUnits.ALL,
    output_field: str | None = None,
) -> CoreMLModel:
    """Trace ``model``, write a ``.mlpackage`` at ``path``, and load it.

    Parameters
    ----------
    model : nn.Module
        Must be in ``eval()`` mode: an exported graph is an inference
        graph, and a training-mode dropout is refused rather than
        silently turned into an identity.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Supplies each input's shape and dtype; the values are irrelevant.
        A tuple is passed to the model positionally, a mapping by
        keyword. The exported model's input shapes are fixed to these.
    path : str
        Destination ``.mlpackage``. Replaced if it exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Precision of the program body. ``FLOAT32`` keeps the export
        faithful to the model it came from; ``FLOAT16`` is what the
        Neural Engine runs. Inputs and outputs stay float32 either way.
    weights : WeightPrecision, optional, keyword-only, default=FLOAT
        How weights are stored. ``INT8`` keeps eight bits per weight plus
        one scale per output channel and lets Core ML dequantize on the
        way in — the package halves against float16 and the accelerator
        moves half as much memory, at a real cost in agreement that
        :meth:`~CoreMLModel.verify` will quantify.
    image_input : ImageInput or None, optional, keyword-only, default=None
        Present the sole input as an image, with the normalisation it
        expects, so an app can hand over a pixel buffer instead of
        converting pixels itself.
    classifier : Classifier or None, optional, keyword-only, default=None
        Declare the model a classifier over these labels, so Core ML —
        and Vision through it — returns the winning label and a
        label-to-probability map instead of a score array. Read with
        :meth:`~CoreMLModel.classify`.
    metadata : Metadata or None, optional, keyword-only, default=None
        Description, author, licence and version to record in the package.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to export when the model returns an output
        dataclass. ``None`` exports every tensor field it declares —
        a detector's boxes and objectness as well as its class scores.

    Returns
    -------
    CoreMLModel
        Loaded and ready to :meth:`~CoreMLModel.predict`.

    Raises
    ------
    ValueError
        The model is in training mode.
    UnsupportedOp
        The trace contains an operation with no MIL translation.
    """
    if getattr(model, "training", False):
        raise ValueError(
            "lucid.coreml: model is in training mode; call model.eval() first"
        )

    info = build_package(
        model,
        example,
        path,
        precision=precision,
        weights=weights,
        image_input=image_input,
        classifier=classifier,
        metadata=metadata,
        output_field=output_field,
    )
    outputs = _features(info["outputs"])
    return CoreMLModel(
        str(info["path"]),
        [name for name, _shape in _features(info["inputs"])],
        [name for name, _shape in outputs],
        compute_units=compute_units,
        precision=precision.value,
        output_shapes=dict(outputs),
        image_input=image_input,
        classifier=classifier,
    )


def load(
    path: str,
    *,
    compute_units: ComputeUnits = ComputeUnits.ALL,
) -> CoreMLModel:
    """Load an existing ``.mlpackage``.

    Feature names are read from the package, so this works on packages
    Lucid did not write.

    Parameters
    ----------
    path : str
        Package to load.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on. Reaching the Neural
        Engine also needs the package to be float16 — check with
        :meth:`CoreMLModel.compute_plan`.

    Returns
    -------
    CoreMLModel
        Loaded and ready to predict.

    Raises
    ------
    RuntimeError
        Core ML could not compile or load the package; its own message
        names the offending layer.
    """
    from lucid._C import engine as _C_engine

    from lucid.coreml._model import _UNITS

    handle = _C_engine.coreml.load_model(path, _UNITS[compute_units])
    input_names, output_names = handle.input_names, handle.output_names
    handle.close()
    return CoreMLModel(
        path, list(input_names), list(output_names), compute_units=compute_units
    )
