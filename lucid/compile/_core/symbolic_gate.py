"""Pre-compile gate: is a traced graph safe to compile with a symbolic batch axis?

A symbolic-batch compile aborts (uncatchably) or silently mis-shapes when the
graph bakes the batch dimension into a constant shape MPSGraph cannot infer:

* explicit ``broadcast_to`` / ``expand`` / ``repeat`` / ``tile`` (the target
  shape carries the batch);
* a ``concatenate`` / ``stack`` on the **batch axis** (dim 0) — the joined dim
  is ``2*B``, which the dim-0 symbolic machinery can't represent;
* a **batch-shaped factory** (``zeros`` / ``ones`` / ``full`` / ``arange`` /
  ``eye`` sized like the input, e.g. ``zeros_like(x)`` or an RNN's zero hidden
  init) — a constant with the batch in its shape.

This gate scans the trace once.  A ``False`` result routes the model to robust
per-shape static caching (correct, never crashes) instead of attempting
symbolic.  It is the safety net that lets ``dynamic=True`` default to *attempting*
symbolic: only graphs that pass the gate ride the single-executable path.

The *view* ops (reshape / flatten / squeeze / contiguous / reduce-squeeze) are
deliberately NOT gated here — they fail gracefully in the emitter (returning nil
when the batch isn't provably preserved at dim 0), so the compile simply retries
static.  Only the ops above, which abort uncatchably or corrupt silently, need
the up-front gate.
"""

# Ops whose target shape carries the batch dimension verbatim.
_BROADCAST_OPS: frozenset[str] = frozenset(
    {"broadcast_to", "expand", "repeat", "tile"}
)
# Concatenate / stack: unsafe only when the join axis IS the batch axis (dim 0).
_JOIN_OPS: frozenset[str] = frozenset({"concatenate", "concat", "cat", "stack"})
# Factory ops: unsafe only when the produced shape's leading dim is the batch.
_FACTORY_OPS: frozenset[str] = frozenset(
    {"zeros", "ones", "full", "arange", "eye", "linspace"}
)

__all__ = ["graph_symbolic_safe"]


def _out_shape(op: object) -> tuple[int, ...]:
    """Leading-output shape of ``op`` as a tuple, or ``()`` if unavailable."""
    outs = getattr(op, "outputs", None)
    if not outs:
        return ()
    shp = getattr(outs[0], "shape", None)
    return tuple(shp) if shp else ()


def graph_symbolic_safe(graph: object, trace_batch: int) -> bool:
    """Return ``True`` if ``graph`` can be compiled with a symbolic batch axis.

    Parameters
    ----------
    graph : TraceGraph
        The recorded trace (its ``ops`` are scanned by name + attrs + shape).
    trace_batch : int
        The leading (batch) size of the user input at trace time — used to tell a
        batch-shaped factory (``zeros_like(x)`` → leading dim == ``trace_batch``)
        from a concrete-shaped one (a fixed positional-encoding table, say).

    Returns
    -------
    bool
        ``False`` if any op would bake the batch into a constant MPSGraph can't
        symbolicise; ``True`` otherwise.
    """
    for op in getattr(graph, "ops", []):
        name = getattr(op, "name", "")
        if name in _BROADCAST_OPS:
            return False
        if name in _JOIN_OPS:
            attrs = getattr(op, "attrs", None) or {}
            axis = attrs.get("dim", attrs.get("axis", 0))
            rank = len(_out_shape(op))
            if isinstance(axis, int):
                if axis < 0:
                    axis += rank
                if axis == 0:  # joining along the batch axis
                    return False
            else:
                return False  # unknown axis → conservatively unsafe
        if name in _FACTORY_OPS:
            shape = _out_shape(op)
            if shape and shape[0] == trace_batch:
                return False
    return True
