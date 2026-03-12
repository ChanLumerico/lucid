"""
lucid.compile._passes.dead_code_elimination
--------------------------------------------
Remove IR nodes whose outputs are never used on any path from the graph
inputs to the declared outputs.

Algorithm
~~~~~~~~~
A backward BFS from the output nodes collects every *live* node ID.  Any
node not in that set is unreachable and is dropped.  Node IDs and
``input_node_ids`` references are then renumbered so that the resulting
graph is dense and consistent.
"""

from __future__ import annotations

from lucid.compile._ir import IRGraph, IRNode

__all__ = ["dead_code_elimination"]


def dead_code_elimination(graph: IRGraph) -> IRGraph:
    """Return a new :class:`~lucid.compile._ir.IRGraph` with dead nodes removed.

    Parameters
    ----------
    graph:
        The source graph (not mutated).

    Returns
    -------
    IRGraph
        A compact graph containing only nodes reachable from the outputs.
    """
    live_ids = _find_live_nodes(graph)

    if len(live_ids) == len(graph):
        return graph

    return _rebuild(graph, live_ids)


def _find_live_nodes(graph: IRGraph) -> set[int]:
    """BFS backward from output nodes to collect all live node IDs."""
    live: set[int] = set()
    queue = list(graph.output_node_ids)

    while queue:
        nid = queue.pop()
        if nid in live:
            continue
        live.add(nid)
        node = graph.get_node(nid)
        for inp_id in node.input_node_ids:
            if inp_id not in live:
                queue.append(inp_id)

    return live


def _rebuild(graph: IRGraph, live_ids: set[int]) -> IRGraph:
    """Construct a fresh graph containing only the *live* nodes."""
    new_graph = IRGraph()

    old_to_new: dict[int, int] = {}

    for node in graph.nodes():
        if node.id not in live_ids:
            continue

        new_inputs = [old_to_new[i] for i in node.input_node_ids]

        if node.is_input:
            new_node = new_graph.add_input(node.output_shape, node.output_dtype)
        elif node.is_constant:
            new_node = new_graph.add_constant(
                node.constant_data, node.output_shape, node.output_dtype
            )
        else:
            new_node = new_graph.add_op(
                op_type=node.op_type,
                input_node_ids=new_inputs,
                output_shape=node.output_shape,
                output_dtype=node.output_dtype,
                attrs=dict(node.attrs),
            )

        old_to_new[node.id] = new_node.id

    new_output_ids = [
        old_to_new[oid] for oid in graph.output_node_ids if oid in old_to_new
    ]
    new_graph.set_outputs(new_output_ids)

    return new_graph
