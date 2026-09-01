"""Lucid trace op → Core ML MIL op.

One function per Lucid op name.  Each takes the already-emitted MIL vars
for the op's inputs, plus the op node itself (for attributes and for the
output shape, which some Lucid ops carry only on their result rather than
as an attribute), and returns the MIL var for the output.

Operand order is Lucid's, and it is not always the obvious one — the
table was read off a trace rather than assumed:

    conv2d           (x, weight, bias)
    linear           (x, weight, bias)
    batch_norm_eval  (x, running_mean, running_var, weight, bias)

That third line is the one worth pausing on: the statistics come *before*
the affine parameters, so an emitter written from the usual
``(gamma, beta, mean, var)`` habit would produce a model that runs,
matches shapes, and is wrong everywhere.
"""

from __future__ import annotations  # tooling only — outside lucid/ (H1 OK)

from typing import Any, Callable

from coremltools.converters.mil import Builder as mb

# Registry: Lucid op name -> emitter.
EMITTERS: dict[str, Callable[..., Any]] = {}


def _emitter(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        EMITTERS[name] = fn
        return fn

    return register


def _pad4(padding: Any) -> list[int]:
    """Lucid's ``[pad_h, pad_w]`` in MIL's ``[top, bottom, left, right]``."""
    ph, pw = int(padding[0]), int(padding[1])
    return [ph, ph, pw, pw]


def _pair(value: Any) -> list[int]:
    return [int(value[0]), int(value[1])]


def _flag(value: Any) -> bool:
    # Lucid encodes these as single-element int lists in the trace.
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


@_emitter("conv2d")
def _conv2d(op: Any, ins: list[Any]) -> Any:
    x, weight = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    attrs = op.attrs
    return mb.conv(
        x=x,
        weight=weight,
        bias=bias,
        strides=_pair(attrs["stride"]),
        pad_type="custom",
        pad=_pad4(attrs["padding"]),
        dilations=_pair(attrs["dilation"]),
        groups=int(attrs["groups"]),
    )


@_emitter("linear")
def _linear(op: Any, ins: list[Any]) -> Any:
    x, weight = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    return mb.linear(x=x, weight=weight, bias=bias)


@_emitter("batch_norm_eval")
def _batch_norm_eval(op: Any, ins: list[Any]) -> Any:
    x, mean, variance, gamma, beta = ins[0], ins[1], ins[2], ins[3], ins[4]
    return mb.batch_norm(
        x=x,
        mean=mean,
        variance=variance,
        gamma=gamma,
        beta=beta,
        epsilon=float(op.attrs["eps"]),
    )


@_emitter("relu")
def _relu(op: Any, ins: list[Any]) -> Any:
    return mb.relu(x=ins[0])


@_emitter("relu6")
def _relu6(op: Any, ins: list[Any]) -> Any:
    return mb.relu6(x=ins[0])


@_emitter("add")
def _add(op: Any, ins: list[Any]) -> Any:
    return mb.add(x=ins[0], y=ins[1])


@_emitter("mul")
def _mul(op: Any, ins: list[Any]) -> Any:
    return mb.mul(x=ins[0], y=ins[1])


@_emitter("reshape")
def _reshape(op: Any, ins: list[Any]) -> Any:
    # Lucid keeps the target shape on the result, not in the attributes.
    return mb.reshape(x=ins[0], shape=[int(d) for d in op.outputs[0].shape])


@_emitter("max_pool2d")
def _max_pool2d(op: Any, ins: list[Any]) -> Any:
    attrs = op.attrs
    return mb.max_pool(
        x=ins[0],
        kernel_sizes=_pair(attrs["kernel_size"]),
        strides=_pair(attrs["stride"]),
        pad_type="custom",
        pad=_pad4(attrs["padding"]),
        ceil_mode=_flag(attrs.get("ceil_mode", [0])),
    )


@_emitter("avg_pool2d")
def _avg_pool2d(op: Any, ins: list[Any]) -> Any:
    attrs = op.attrs
    return mb.avg_pool(
        x=ins[0],
        kernel_sizes=_pair(attrs["kernel_size"]),
        strides=_pair(attrs["stride"]),
        pad_type="custom",
        pad=_pad4(attrs["padding"]),
        exclude_padding_from_average=not _flag(attrs.get("count_include_pad", [1])),
        ceil_mode=_flag(attrs.get("ceil_mode", [0])),
    )


@_emitter("dropout")
def _dropout(op: Any, ins: list[Any]) -> Any:
    # An exported model is an inference model.  Lucid still records the op
    # under ``eval()``; refusing a training-mode one keeps the export from
    # silently dropping the noise a caller expected to keep.
    if op.attrs.get("training"):
        raise NotImplementedError(
            "export_coreml: dropout was traced in training mode — call "
            "model.eval() before exporting, or the exported graph would "
            "differ from the traced one"
        )
    return mb.identity(x=ins[0])


@_emitter("identity")
def _identity(op: Any, ins: list[Any]) -> Any:
    return mb.identity(x=ins[0])
