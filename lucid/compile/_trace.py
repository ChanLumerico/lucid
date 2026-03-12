"""
lucid.compile._trace
--------------------
Forward-pass graph tracing for Lucid's compilation system.

Usage
~~~~~
Wrap a forward call with :func:`tracing_mode` to capture every operation
dispatched through :func:`lucid._backend.core._py_func_op` into a live
:class:`TraceContext`.  The tracer uses Python ``id()`` as a temporary key to
map tensor objects to their :class:`~lucid.compile._ir.IRNode` counterparts.

The hook in ``_backend/core.py`` calls :func:`get_trace_context` on every
operation dispatch.  When ``None`` is returned the overhead is a single
attribute lookup – negligible in eager mode.

Thread safety
~~~~~~~~~~~~~
The active context is stored in a :class:`threading.local` so that multiple
threads can trace independent graphs simultaneously.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Generator, Sequence, TYPE_CHECKING

from lucid.compile._ir import IRGraph, IRNode

if TYPE_CHECKING:
    from lucid.types import _TensorLike

__all__ = [
    "TraceContext",
    "get_trace_context",
    "is_tracing",
    "tracing_mode",
]

_trace_local = threading.local()


def get_trace_context() -> "TraceContext | None":
    """Return the active :class:`TraceContext`, or ``None`` when not tracing."""
    return getattr(_trace_local, "context", None)


def is_tracing() -> bool:
    """Return ``True`` if a trace is currently active on this thread."""
    return get_trace_context() is not None


class TraceContext:
    """Mutable state accumulated while tracing a single forward pass.

    Parameters
    ----------
    input_tensors:
        The live tensors supplied by the caller.  Each will be registered as
        an ``"input"`` node in the graph.

    Attributes
    ----------
    graph:
        The :class:`~lucid.compile._ir.IRGraph` being built.
    """

    def __init__(self, input_tensors: "Sequence[_TensorLike]" = ()) -> None:
        self.graph: IRGraph = IRGraph()
        self._tensor_to_node_id: dict[int, int] = {}
        self._constant_ids: set[int] = set()

        for t in input_tensors:
            self.register_input(t)

    def register_input(self, tensor: "_TensorLike") -> IRNode:
        """Register *tensor* as a live input node."""
        node = self.graph.add_input(
            shape=tuple(tensor.shape), dtype=tensor.dtype
        )
        self._tensor_to_node_id[id(tensor)] = node.id
        return node

    def register_constant(self, tensor: "_TensorLike") -> IRNode:
        """Register *tensor* as a constant (weight / buffer) node.

        The raw array is copied into the node so the graph is self-contained.
        """
        data = tensor.data
        if hasattr(data, "copy"):
            data = data.copy()
        node = self.graph.add_constant(
            data=data,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        self._tensor_to_node_id[id(tensor)] = node.id
        self._constant_ids.add(node.id)
        return node

    def record_op(
        self,
        op: Any,
        input_tensors: "tuple[_TensorLike, ...]",
        output_tensors: "tuple[_TensorLike, ...] | _TensorLike",
    ) -> None:
        """Record a single dispatched operation.

        Called automatically from the ``_py_func_op`` wrapper whenever a trace
        is active.  Unknown input tensors (parameters seen for the first time)
        are auto-registered as constants.

        Parameters
        ----------
        op:
            The :class:`~lucid.._backend.core.Operation` instance that was
            executed.
        input_tensors:
            The tensor operands consumed by the operation.
        output_tensors:
            The tensor(s) produced by the operation.
        """
        if not isinstance(output_tensors, tuple):
            output_tensors = (output_tensors,)

        op_type = type(op).__name__

        input_node_ids: list[int] = []
        for t in input_tensors:
            tid = id(t)
            if tid not in self._tensor_to_node_id:
                self.register_constant(t)
            input_node_ids.append(self._tensor_to_node_id[tid])

        for out_tensor in output_tensors:
            node = self.graph.add_op(
                op_type=op_type,
                input_node_ids=input_node_ids,
                output_shape=tuple(out_tensor.shape),
                output_dtype=out_tensor.dtype,
            )
            self._tensor_to_node_id[id(out_tensor)] = node.id

    def set_outputs(
        self, output_tensors: "Sequence[_TensorLike] | _TensorLike"
    ) -> None:
        """Mark which tensor(s) are the callable's return values."""
        if not isinstance(output_tensors, (list, tuple)):
            output_tensors = (output_tensors,)

        out_ids: list[int] = []
        for t in output_tensors:
            nid = self._tensor_to_node_id.get(id(t))
            if nid is not None:
                out_ids.append(nid)

        self.graph.set_outputs(out_ids)

    def get_node_id_for(self, tensor: "_TensorLike") -> int | None:
        """Return the graph node ID for *tensor*, or ``None`` if untracked."""
        return self._tensor_to_node_id.get(id(tensor))


@contextmanager
def tracing_mode(
    input_tensors: "Sequence[_TensorLike]" = (),
) -> Generator[TraceContext, None, None]:
    """Context manager that activates graph tracing for the current thread.

    All operations dispatched through :func:`lucid._backend.core._py_func_op`
    inside this block are recorded in the returned :class:`TraceContext`.

    Parameters
    ----------
    input_tensors:
        Live tensors to pre-register as ``"input"`` nodes before the forward
        pass runs.

    Yields
    ------
    TraceContext
        The context object accumulating the graph.

    Example
    -------
    ::

        x = lucid.ones((1, 3, 224, 224))
        with tracing_mode([x]) as ctx:
            y = model(x)
            ctx.set_outputs([y])

        graph = ctx.graph
    """
    ctx = TraceContext(input_tensors)
    _trace_local.context = ctx
    try:
        yield ctx
    finally:
        _trace_local.context = None
