"""
lucid.compile._passes.constant_folding
---------------------------------------
Pre-evaluate IR sub-graphs whose every input is a constant (weight / buffer).

Motivation
~~~~~~~~~~
Many networks carry *fixed* tensors that are mathematically combined at the
start of every forward pass – e.g. ``weight / sqrt(running_var + eps)`` in a
fused BatchNorm, or the key/value projection matrices pre-multiplied together.
By evaluating these sub-graphs *once* at compile time and embedding the result
as a new ``"constant"`` node we eliminate redundant computation at runtime.

Algorithm
~~~~~~~~~
1. Mark every node that is transitively computable from constants-only as
   *foldable*.
2. For each foldable non-constant op node, look up the live Operation class by
   name and execute it eagerly with ``lucid.no_grad()`` to obtain the folded
   tensor value.
3. Replace the op node with a ``"constant"`` node holding the result; rewire
   downstream consumers accordingly.
4. Run a lightweight DCE pass to remove nodes that are now orphaned.

Limitations
~~~~~~~~~~~
* Only CPU nodes are folded (MLX constants require a device flush which is
  deferred; GPU paths are left to the MLX compiler).
* Operations that cannot be looked up by name (custom ops without a registered
  ``Operation`` subclass) are skipped safely.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from lucid.compile._ir import IRGraph, IRNode
from lucid.compile._passes.dead_code_elimination import dead_code_elimination

__all__ = ["constant_folding"]


def constant_folding(graph: IRGraph) -> IRGraph:
    """Return a graph with constant sub-expressions pre-evaluated.

    Parameters
    ----------
    graph:
        Source graph (not mutated).

    Returns
    -------
    IRGraph
        Optimised graph with foldable op nodes replaced by constant nodes.
    """
    foldable = _find_foldable_nodes(graph)
    if not foldable:
        return graph

    return _fold_and_rebuild(graph, foldable)


def _find_foldable_nodes(graph: IRGraph) -> set[int]:
    """Return IDs of non-input nodes that depend only on constants."""
    const_ids: set[int] = set()

    for node in graph.nodes():
        if node.is_input:
            continue
        if node.is_constant:
            const_ids.add(node.id)
            continue
        if all(inp_id in const_ids for inp_id in node.input_node_ids):
            const_ids.add(node.id)

    return const_ids - {n.id for n in graph.nodes() if n.is_constant}


def _resolve_op_class(op_type: str) -> type | None:
    """Look up the Lucid Operation class for *op_type*, or return ``None``."""
    try:
        import lucid._func as _func_module
        import lucid._func.bfunc as _bfunc
        import lucid._func.ufunc as _ufunc
        import lucid._func.gfunc as _gfunc

        for mod in (_func_module, _bfunc, _ufunc, _gfunc):
            cls = vars(mod).get(op_type)
            if cls is not None and isinstance(cls, type):
                return cls
    except Exception:
        pass
    return None


def _execute_constant_op(
    op_type: str,
    input_data: list[Any],
) -> Any | None:
    """Eagerly run an operation on raw NumPy arrays and return the result.

    Returns ``None`` if the operation cannot be executed (unknown op or
    runtime error).
    """
    import lucid

    op_cls = _resolve_op_class(op_type)
    if op_cls is None:
        return None

    try:
        tensors = [lucid.tensor(d) for d in input_data]
        with lucid.no_grad():
            op_instance = op_cls()
            result = op_instance(*tensors)
        if isinstance(result, tuple):
            result = result[0]
        return result.data
    except Exception:
        return None


def _fold_and_rebuild(graph: IRGraph, foldable: set[int]) -> IRGraph:
    """Build a new graph with foldable nodes replaced by constant nodes."""
    new_graph = IRGraph()
    old_to_new: dict[int, int] = {}

    for node in graph.nodes():
        new_input_ids = [old_to_new[i] for i in node.input_node_ids]

        if node.is_input:
            new_node = new_graph.add_input(node.output_shape, node.output_dtype)

        elif node.is_constant:
            new_node = new_graph.add_constant(
                node.constant_data, node.output_shape, node.output_dtype
            )

        elif node.id in foldable:
            input_data = [
                new_graph.get_node(old_to_new[i]).constant_data
                for i in node.input_node_ids
            ]
            folded = _execute_constant_op(node.op_type, input_data)

            if folded is not None:
                new_node = new_graph.add_constant(
                    data=folded,
                    shape=tuple(folded.shape) if hasattr(folded, "shape") else (),
                    dtype=folded.dtype if hasattr(folded, "dtype") else type(folded),
                )
            else:
                new_node = new_graph.add_op(
                    op_type=node.op_type,
                    input_node_ids=new_input_ids,
                    output_shape=node.output_shape,
                    output_dtype=node.output_dtype,
                    attrs=dict(node.attrs),
                )

        else:
            new_node = new_graph.add_op(
                op_type=node.op_type,
                input_node_ids=new_input_ids,
                output_shape=node.output_shape,
                output_dtype=node.output_dtype,
                attrs=dict(node.attrs),
            )

        old_to_new[node.id] = new_node.id

    new_output_ids = [
        old_to_new[oid] for oid in graph.output_node_ids if oid in old_to_new
    ]
    new_graph.set_outputs(new_output_ids)

    return dead_code_elimination(new_graph)
