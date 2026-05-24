"""
JSON serialisation for :class:`TraceGraph` (3.5 Phase 1.1 Weeks 2–4).

The trace IR (``TensorMeta`` / ``OpNode`` / ``TraceGraph``) is a plain
record graph; round-tripping it through JSON lets diagnostics, parity
tests, and the Phase 1.2 builder pick up captured traces without
holding a live Tracer.

Two surfaces are exposed:

- :func:`trace_to_dict` / :func:`trace_to_json` — convert a recorded
  graph to a JSON-safe Python dict (or formatted JSON string).
- :func:`dump_to_path_if_debug_enabled` — invoked from the
  ``_tracing()`` context manager's ``finally`` block; writes the trace
  to a per-process file when ``LUCID_COMPILE_DEBUG=1`` is set in the
  environment.  Returns the chosen path or ``None`` when debug dumping
  is off.

Notes
-----
H4-compliant: only :mod:`json` / :mod:`os` / :mod:`pathlib` /
:mod:`tempfile` from the standard library — no external dependencies.
The ``Device`` / ``Dtype`` enums serialise as their ``.name`` strings
so JSON consumers don't need the engine's enum mapping.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._C.engine.compile import OpNode, TensorMeta, TraceGraph

__all__ = [
    "trace_to_dict",
    "trace_to_json",
    "dump_to_path_if_debug_enabled",
]


# Module-level monotonic counter for trace file names so two traces in
# the same process don't collide.  Reset is intentionally not exposed —
# callers care about uniqueness, not the absolute value.
_DUMP_SEQ: int = 0


def _tensor_meta_to_dict(meta: TensorMeta) -> dict[str, object]:
    """Convert a single :class:`TensorMeta` record into a JSON-safe dict."""
    return {
        "id": int(meta.id),
        "shape": list(meta.shape),
        "dtype": meta.dtype.name,
        "device": meta.device.name,
    }


def _op_node_to_dict(node: OpNode) -> dict[str, object]:
    """Convert one :class:`OpNode` (including its outputs + attrs) into a JSON dict."""
    # ``node.attrs`` is a pybind11-converted dict; values arrive as
    # Python primitives (int / list[int] / float / bool / str) thanks to
    # the variant caster, so direct ``dict(...)`` is JSON-safe.
    return {
        "name": node.name,
        "inputs": [int(x) for x in node.inputs],
        "outputs": [_tensor_meta_to_dict(m) for m in node.outputs],
        "attrs": dict(node.attrs),
    }


def trace_to_dict(graph: TraceGraph) -> dict[str, object]:
    """Convert a :class:`TraceGraph` into a JSON-serialisable dict.

    Parameters
    ----------
    graph : TraceGraph
        The recorded op DAG returned by ``tracer.graph``.

    Returns
    -------
    dict
        ``{"format": ..., "next_id": ..., "ops": [...]}`` — round-trips
        through :func:`json.dumps` / :func:`json.loads` without loss.
    """
    return {
        "format": "lucid.compile.trace/v1",
        "next_id": int(graph.next_id),
        "ops": [_op_node_to_dict(node) for node in graph.ops],
    }


def trace_to_json(graph: TraceGraph, *, indent: int = 2) -> str:
    """Serialise a :class:`TraceGraph` to a formatted JSON string.

    Parameters
    ----------
    graph : TraceGraph
        The recorded op DAG.
    indent : int
        Number of spaces per nesting level (passed to :func:`json.dumps`).

    Returns
    -------
    str
        The pretty-printed JSON text.  Stable ordering: ``"format"``,
        ``"next_id"``, ``"ops"`` keys come out in insertion order
        thanks to Python's dict guarantee.
    """
    return json.dumps(trace_to_dict(graph), indent=indent)


def _next_dump_path() -> Path:
    """Pick a fresh, unique temp-dir path for the next debug dump."""
    global _DUMP_SEQ
    _DUMP_SEQ += 1
    tmp = Path(tempfile.gettempdir())
    return tmp / f"lucid_trace_{os.getpid()}_{_DUMP_SEQ:04d}.json"


def dump_to_path_if_debug_enabled(graph: TraceGraph) -> Path | None:
    """Dump *graph* to a temp file when ``LUCID_COMPILE_DEBUG=1``.

    Called from the :func:`lucid.compile._tracing` context manager's
    ``finally`` block.  When the environment flag is unset, returns
    ``None`` immediately without touching the filesystem.

    Parameters
    ----------
    graph : TraceGraph
        The recorded op DAG to serialise.

    Returns
    -------
    Path | None
        The path the trace was written to, or ``None`` when debug
        dumping is disabled.

    Notes
    -----
    The chosen path is :func:`tempfile.gettempdir` /
    ``lucid_trace_<pid>_<seq>.json``.  The sequence counter is
    process-local; calling this concurrently from two interpreters
    yields distinct paths because the PID differs.
    """
    if os.environ.get("LUCID_COMPILE_DEBUG", "") != "1":
        return None
    path = _next_dump_path()
    payload = trace_to_json(graph)
    path.write_text(payload, encoding="utf-8")
    return path
