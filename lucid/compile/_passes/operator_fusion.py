"""
lucid.compile._passes.operator_fusion
--------------------------------------
Fuse pairs (or chains) of compatible forward-pass operations into a single
``"fused:<A>+<B>"`` node, reducing memory traffic and Python-dispatch overhead.

Design
~~~~~~
A *fusion rule* is a pair ``(producer_op, consumer_op)`` that can be merged.
Rules are registered in :data:`FUSION_RULES` and describe:

* which op-type strings can fuse,
* whether the consumer must be the *only* user of the producer's output
  (``single_consumer_only``),
* a human-readable ``fused_name`` used as the merged node's ``op_type``.

The pass runs a single sweep over the topological order, greedily merging
adjacent pairs that match a rule.  After a merge the resulting fused node is
eligible for further merging in the same sweep.

Built-in rules
~~~~~~~~~~~~~~
==================  ========================  ==================================
Producer            Consumer                  Fused name
==================  ========================  ==================================
``matmul``          ``add``                   ``fused:linear``
``linear``          ``relu``                  ``fused:linear_relu``
``linear``          ``gelu``                  ``fused:linear_gelu``
``linear``          ``silu``                  ``fused:linear_silu``
``conv2d``          ``add``                   ``fused:conv_bias``
``add``             ``relu``                  ``fused:add_relu``
``mul``             ``add``                   ``fused:mul_add``
``exp``             ``sum``                   ``fused:logsumexp_inner``
==================  ========================  ==================================

Custom rules can be added via :func:`register_fusion_rule`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from lucid.compile._ir import IRGraph, IRNode

__all__ = ["operator_fusion", "register_fusion_rule", "FUSION_RULES"]


class FusionRule(NamedTuple):
    """Description of a valid producer→consumer fusion."""

    producer_op: str
    consumer_op: str
    fused_name: str
    single_consumer_only: bool = True


FUSION_RULES: list[FusionRule] = [
    FusionRule("matmul", "add",  "fused:linear",         single_consumer_only=True),
    FusionRule("linear", "relu", "fused:linear_relu",    single_consumer_only=True),
    FusionRule("linear", "gelu", "fused:linear_gelu",    single_consumer_only=True),
    FusionRule("linear", "silu", "fused:linear_silu",    single_consumer_only=True),
    FusionRule("fused:linear",      "relu", "fused:linear_relu", single_consumer_only=True),
    FusionRule("fused:linear",      "gelu", "fused:linear_gelu", single_consumer_only=True),
    FusionRule("fused:linear",      "silu", "fused:linear_silu", single_consumer_only=True),
    FusionRule("conv2d", "add",  "fused:conv_bias",      single_consumer_only=True),
    FusionRule("add",    "relu", "fused:add_relu",       single_consumer_only=True),
    FusionRule("mul",    "add",  "fused:mul_add",        single_consumer_only=True),
    FusionRule("exp",    "sum",  "fused:logsumexp_inner",single_consumer_only=True),
]


def register_fusion_rule(rule: FusionRule) -> None:
    """Append *rule* to the global :data:`FUSION_RULES` list."""
    FUSION_RULES.append(rule)


def operator_fusion(graph: IRGraph) -> IRGraph:
    """Return a new graph with eligible op pairs merged.

    Parameters
    ----------
    graph:
        Source graph (not mutated).

    Returns
    -------
    IRGraph
        Graph where fused nodes carry ``op_type`` strings like
        ``"fused:linear_relu"``.
    """
    _build_lookup()

    topo = graph.topological_order()
    consumer_count = _count_consumers(topo)
    merged: set[int] = set()

    new_graph = IRGraph()
    old_to_new: dict[int, int] = {}

    for node in topo:
        if node.id in merged:
            continue

        new_input_ids = [old_to_new[i] for i in node.input_node_ids]

        if node.is_input:
            new_node = new_graph.add_input(node.output_shape, node.output_dtype)
        elif node.is_constant:
            new_node = new_graph.add_constant(
                node.constant_data, node.output_shape, node.output_dtype
            )
        else:
            fused_node, skip_ids = _try_fuse_chain(
                node, graph, consumer_count, old_to_new
            )
            if fused_node is not None:
                new_node = new_graph.add_op(
                    op_type=fused_node.fused_name,
                    input_node_ids=fused_node.merged_inputs,
                    output_shape=fused_node.output_shape,
                    output_dtype=fused_node.output_dtype,
                    attrs=fused_node.attrs,
                )
                for sid in skip_ids:
                    merged.add(sid)
                for sid in skip_ids:
                    old_to_new[sid] = new_node.id
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
    return new_graph


@dataclass
class _FusedCandidate:
    fused_name: str
    merged_inputs: list[int]
    output_shape: tuple[int, ...]
    output_dtype: object
    attrs: dict


_rule_lookup: dict[tuple[str, str], FusionRule] = {}


def _build_lookup() -> None:
    _rule_lookup.clear()
    for rule in FUSION_RULES:
        _rule_lookup[(rule.producer_op, rule.consumer_op)] = rule


def _count_consumers(topo: list[IRNode]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in topo:
        for inp_id in node.input_node_ids:
            counts[inp_id] = counts.get(inp_id, 0) + 1
    return counts


def _try_fuse_chain(
    producer: IRNode,
    graph: IRGraph,
    consumer_count: dict[int, int],
    old_to_new: dict[int, int],
) -> tuple["_FusedCandidate | None", list[int]]:
    """Attempt to greedily extend *producer* into a fused chain.

    Returns
    -------
    (_FusedCandidate, [fused_node_ids_to_skip]) or (None, [])
    """
    current = producer
    skipped: list[int] = []
    merged_inputs = [old_to_new[i] for i in current.input_node_ids]
    current_op = current.op_type
    output_shape = current.output_shape
    output_dtype = current.output_dtype
    merged_attrs: dict = dict(current.attrs)

    fused_name: str | None = None

    while True:
        consumer = _find_single_consumer(current, graph, consumer_count)
        if consumer is None:
            break

        rule = _rule_lookup.get((current_op, consumer.op_type))
        if rule is None:
            break

        if rule.single_consumer_only and consumer_count.get(current.id, 0) > 1:
            break

        missing_extra = [
            i for i in consumer.input_node_ids
            if i != current.id and i not in old_to_new
        ]
        if missing_extra:
            break

        extra_inputs = [
            old_to_new[i]
            for i in consumer.input_node_ids
            if i != current.id
        ]

        fused_name = rule.fused_name
        skipped.append(consumer.id)
        output_shape = consumer.output_shape
        output_dtype = consumer.output_dtype
        merged_attrs.update(consumer.attrs)
        merged_inputs = merged_inputs + extra_inputs

        current_op = fused_name
        current = consumer

    if fused_name is None:
        return None, []

    return (
        _FusedCandidate(
            fused_name=fused_name,
            merged_inputs=merged_inputs,
            output_shape=output_shape,
            output_dtype=output_dtype,
            attrs=merged_attrs,
        ),
        skipped,
    )


def _find_single_consumer(
    node: IRNode,
    graph: IRGraph,
    consumer_count: dict[int, int],
) -> "IRNode | None":
    """Return the unique consumer of *node*'s output if exactly one exists."""
    for candidate in graph.nodes():
        if node.id in candidate.input_node_ids:
            return candidate
    return None
