"""Custom ``Function`` calls a compiled backward cannot run.

A compiled training step traces its model under ``no_grad``, where
``Function.apply`` runs ``forward`` and registers no node: the recording holds
``forward``'s own ops, and the compiled backward differentiates those.  The
gradient ``backward`` defines never runs.  For a straight-through estimator, a
reversed or scaled gradient, or a clipped one, that is a different gradient —
``fake_quantize`` on a QAT weight differentiated its ``round`` and handed the
weight a compiled gradient of exactly zero.

Whether a ``backward`` is the derivative of its ``forward`` cannot be told, so
none is differentiated through.  ``apply`` reports every call made while a
tracer is installed (:func:`note_applied`); each entry point that
differentiates a recording calls :func:`refuse_custom_backward` first, and a
gradient that would pass through such a call makes the step run eagerly.  A
forward-only compile and a Core ML export differentiate nothing and never ask.
"""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

if TYPE_CHECKING:
    from lucid._C.engine.compile import TraceGraph, Tracer

__all__ = [
    "GRADIENT_DTYPES",
    "CustomBackwardInTrace",
    "refuse_custom_backward",
    "note_applied",
]

#: The dtypes a gradient can flow through — eager never tracks an integer or
#: bool value.  The backward walker's ``carries_gradient`` is the same rule.
GRADIENT_DTYPES = frozenset(
    {
        _C_engine.Dtype.F16,
        _C_engine.Dtype.BF16,
        _C_engine.Dtype.F32,
        _C_engine.Dtype.F64,
        _C_engine.Dtype.C64,
        _C_engine.Dtype.C128,
    }
)


class CustomBackwardInTrace(NotImplementedError):
    """A gradient a lowering would compute passes through a custom ``Function``.

    A ``NotImplementedError``, and so a ``RuntimeError``: an entry point
    that runs eagerly when its lowering fails catches it with the rest,
    and ``compiled_step``, which has no eager step, lets it through.
    """


@final
@dataclass(frozen=True, slots=True)
class _Applied:
    """One ``Function.apply`` made while a recording was being made."""

    name: str
    # The tensors ``forward`` was given and the floating ones it returned,
    # with the ids the tracer gave them right then.  The impls are looked up
    # again at the end as well: a tensor no op has consumed yet — an input
    # handed back unchanged — has no id until one does.
    inputs: tuple[_C_engine.TensorImpl, ...]
    outputs: tuple[_C_engine.TensorImpl, ...]
    input_ids: frozenset[int]
    output_ids: frozenset[int]
    # An input that requires grad under the trace's ``no_grad`` is a leaf —
    # a parameter, most often — which the trace may never record at all
    # (``forward`` ignored it) and ``backward`` may still answer for.
    takes_leaf: bool


#: Every call noted, per tracer — gone with the tracer.
_APPLIED: weakref.WeakKeyDictionary[Tracer, list[_Applied]] = (
    weakref.WeakKeyDictionary()
)


def _impls(
    values: tuple[object, ...], *, floating: bool
) -> tuple[_C_engine.TensorImpl, ...]:
    """The impls of the tensors among ``values`` — only floating ones if asked."""
    return tuple(
        v._impl
        for v in values
        if isinstance(v, Tensor) and (not floating or v._impl.dtype in GRADIENT_DTYPES)
    )


def _ids(tracer: Tracer, impls: tuple[_C_engine.TensorImpl, ...]) -> frozenset[int]:
    """The trace ids ``tracer`` has for ``impls`` now; one it has none for is left out."""
    return frozenset(
        int(tid) for tid in map(tracer.lookup_id, impls) if tid is not None
    )


def note_applied(
    tracer: Tracer, function: type, args: tuple[object, ...], output: object
) -> None:
    """Record that ``function``'s ``forward`` ran while ``tracer`` recorded.

    Called by ``Function.apply`` as soon as ``forward`` returns, whatever the
    grad mode: under a compiled step's ``no_grad`` it is the only trace the
    call leaves.

    Parameters
    ----------
    tracer : Tracer
        The installed tracer.
    function : type
        The ``Function`` subclass, named in the fallback reason.
    args : tuple of object
        ``forward``'s positional arguments, the ones ``backward`` answers for.
    output : object
        What ``forward`` returned — a tensor, or a tuple that may hold
        non-tensor entries.
    """
    outputs = _impls(output if isinstance(output, tuple) else (output,), floating=True)
    if not outputs:
        return
    inputs = _impls(args, floating=False)
    _APPLIED.setdefault(tracer, []).append(
        _Applied(
            function.__name__,
            inputs,
            outputs,
            _ids(tracer, inputs),
            _ids(tracer, outputs),
            any(isinstance(a, Tensor) and a.requires_grad for a in args),
        )
    )


# Both walks follow every recorded op, the ones no gradient crosses (``detach``,
# a comparison) as well: which ops a gradient stops at is the differentiating
# backend's to decide, and a path kept here that no gradient takes only sends
# to eager a step that could have compiled.


def _ancestors(graph: TraceGraph, sink_id: int) -> set[int]:
    """Every trace id ``sink_id`` was computed from, itself included."""
    reached = {sink_id}
    for node in reversed(graph.ops):
        if any(int(meta.id) in reached for meta in node.outputs):
            reached.update(int(i) for i in node.inputs if int(i) >= 0)
    return reached


def _descendants(graph: TraceGraph, source_ids: list[int]) -> set[int]:
    """Every trace id computed from one of ``source_ids``, those included."""
    reached = set(source_ids)
    for node in graph.ops:
        if any(int(i) in reached for i in node.inputs):
            reached.update(int(meta.id) for meta in node.outputs)
    return reached


def refuse_custom_backward(tracer: Tracer, sink_id: int, source_ids: list[int]) -> None:
    """Refuse a lowering whose gradient would pass through a noted call.

    It would when one of the call's outputs leads to ``sink_id``, and one of
    its inputs or outputs comes from a source, or one of its inputs is a leaf
    that requires grad.  Whichever backward ran — the user's, eagerly, or the
    compiled derivative of ``forward`` — the gradient of a source would then
    go through it.

    Parameters
    ----------
    tracer : Tracer
        A finished recording.
    sink_id : int
        Trace id of the value the lowering differentiates — the loss.
    source_ids : list of int
        Trace ids of what it differentiates with respect to — the parameters,
        and a segment's input.

    Raises
    ------
    CustomBackwardInTrace
        Naming the first such ``Function``.
    """
    applied = _APPLIED.get(tracer)
    if not applied:
        return
    ancestors = _ancestors(tracer.graph, sink_id)
    descendants = _descendants(tracer.graph, source_ids)
    for call in applied:
        outputs = call.output_ids | _ids(tracer, call.outputs)
        touched = outputs | call.input_ids | _ids(tracer, call.inputs)
        if not ancestors.isdisjoint(outputs) and (
            call.takes_leaf or not descendants.isdisjoint(touched)
        ):
            raise CustomBackwardInTrace(
                f"autograd.Function {call.name} defines its own backward, which a "
                f"compiled backward cannot run: it would differentiate "
                f"{call.name}.forward instead"
            )
