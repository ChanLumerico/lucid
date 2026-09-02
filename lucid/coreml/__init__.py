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
from lucid.coreml._spec import ComputeUnits, Precision

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = [
    "ComputeUnits",
    "CoreMLModel",
    "PlacementSummary",
    "Precision",
    "UnsupportedOp",
    "UnsupportedRank",
    "export",
    "load",
]


def _shape_of(value: object) -> tuple[int, ...]:
    """The traced output shape as ``build_package`` reports it."""
    if not isinstance(value, tuple):
        raise TypeError(f"lucid.coreml: expected a shape, got {type(value).__name__}")
    return tuple(int(d) for d in value)


def export(
    model: Module,
    example: Tensor,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
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
    example : Tensor
        Supplies the input shape and dtype; its values are irrelevant.
        The exported model's input shape is fixed to it.
    path : str
        Destination ``.mlpackage``. Replaced if it exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Precision of the program body. ``FLOAT32`` keeps the export
        faithful to the model it came from; ``FLOAT16`` is what the
        Neural Engine runs. Inputs and outputs stay float32 either way.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on.
    output_field : str or None, optional, keyword-only, default=None
        Attribute to export when the model returns an output dataclass.
        ``None`` takes ``logits``, which the zoo's output types carry.

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
        model, example, path, precision=precision, output_field=output_field
    )
    return CoreMLModel(
        str(info["path"]),
        str(info["input"]),
        str(info["output"]),
        compute_units=compute_units,
        precision=precision.value,
        output_shape=_shape_of(info["output_shape"]),
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
    input_name, output_name = handle.input_name, handle.output_name
    handle.close()
    return CoreMLModel(path, input_name, output_name, compute_units=compute_units)
