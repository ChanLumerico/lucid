"""
lucid.compile._ir
-----------------
Intermediate Representation (IR) for Lucid's graph compilation system.

Each forward-pass computation is captured as a directed acyclic graph (DAG) of
``IRNode`` objects.  The graph is device-agnostic and is used by the optimisation
passes before being lowered back to an executable plan.

Node kinds
~~~~~~~~~~
* ``"input"``    – a live tensor fed by the caller at runtime
* ``"constant"`` – a frozen weight / buffer whose data is embedded in the graph
* ``op-type``    – any ``Operation`` subclass name (e.g. ``"matmul"``, ``"add"``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


__all__ = ["IRNode", "IRGraph"]


@dataclass
class IRNode:
    """A single vertex in an :class:`IRGraph`.

    Parameters
    ----------
    id:
        Sequential integer assigned at insertion time.  Doubles as the index
        into :py:attr:`IRGraph._nodes`.
    op_type:
        String name of the operation, e.g. ``"matmul"`` or ``"relu"``.
        Special values: ``"input"`` and ``"constant"``.
    input_node_ids:
        Ordered list of predecessor node IDs whose outputs feed this node.
    output_shape:
        Shape of the tensor produced by this node.
    output_dtype:
        NumPy / MLX dtype of the tensor produced by this node.
    attrs:
        Non-tensor keyword arguments forwarded to the operation
        (e.g. ``{"axis": 1}``).
    is_input:
        ``True`` when the node represents a live caller-supplied tensor.
    is_constant:
        ``True`` when the node is a frozen parameter or buffer.
    constant_data:
        The raw array data when ``is_constant=True``.
    """

    id: int
    op_type: str
    input_node_ids: list[int]
    output_shape: tuple[int, ...]
    output_dtype: Any
    attrs: dict[str, Any] = field(default_factory=dict)
    is_input: bool = False
    is_constant: bool = False
    constant_data: Any = None

    def __repr__(self) -> str:
        kind = (
            "input"
            if self.is_input
            else "const"
            if self.is_constant
            else self.op_type
        )
        return (
            f"IRNode(id={self.id}, kind={kind!r}, "
            f"inputs={self.input_node_ids}, shape={self.output_shape})"
        )


class IRGraph:
    """Directed acyclic graph of :class:`IRNode` objects.

    Nodes are stored in insertion order; because nodes may only reference
    predecessors, the list is always a valid topological sort prefix.

    Attributes
    ----------
    input_node_ids:
        IDs of nodes that represent caller-supplied tensors.
    output_node_ids:
        IDs of nodes whose tensors form the callable's return value.
    """

    def __init__(self) -> None:
        self._nodes: list[IRNode] = []
        self._id_counter: int = 0
        self.input_node_ids: list[int] = []
        self.output_node_ids: list[int] = []

    def _next_id(self) -> int:
        nid = self._id_counter
        self._id_counter += 1
        return nid

    def add_input(self, shape: tuple[int, ...], dtype: Any) -> IRNode:
        """Register a live input tensor and return its node."""
        nid = self._next_id()
        node = IRNode(
            id=nid,
            op_type="input",
            input_node_ids=[],
            output_shape=shape,
            output_dtype=dtype,
            is_input=True,
        )
        self._nodes.append(node)
        self.input_node_ids.append(nid)
        return node

    def add_constant(self, data: Any, shape: tuple[int, ...], dtype: Any) -> IRNode:
        """Embed a frozen constant (weight / buffer) into the graph."""
        nid = self._next_id()
        node = IRNode(
            id=nid,
            op_type="constant",
            input_node_ids=[],
            output_shape=shape,
            output_dtype=dtype,
            is_constant=True,
            constant_data=data,
        )
        self._nodes.append(node)
        return node

    def add_op(
        self,
        op_type: str,
        input_node_ids: list[int],
        output_shape: tuple[int, ...],
        output_dtype: Any,
        attrs: dict[str, Any] | None = None,
    ) -> IRNode:
        """Append a computed operation node."""
        nid = self._next_id()
        node = IRNode(
            id=nid,
            op_type=op_type,
            input_node_ids=list(input_node_ids),
            output_shape=output_shape,
            output_dtype=output_dtype,
            attrs=attrs or {},
        )
        self._nodes.append(node)
        return node

    def set_outputs(self, node_ids: list[int]) -> None:
        """Declare which nodes produce the callable's outputs."""
        self.output_node_ids = list(node_ids)

    def get_node(self, node_id: int) -> IRNode:
        """Return the node with the given ID (O(1))."""
        return self._nodes[node_id]

    def nodes(self) -> Iterator[IRNode]:
        """Iterate over all nodes in insertion order."""
        return iter(self._nodes)

    def topological_order(self) -> list[IRNode]:
        """Return nodes reachable from the outputs in dependency order."""
        visited: set[int] = set()
        order: list[IRNode] = []

        def _dfs(node: IRNode) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for inp_id in node.input_node_ids:
                _dfs(self._nodes[inp_id])
            order.append(node)

        for out_id in self.output_node_ids:
            _dfs(self._nodes[out_id])

        return order

    def op_count(self) -> int:
        """Return the number of non-input, non-constant nodes."""
        return sum(
            1
            for n in self._nodes
            if not n.is_input and not n.is_constant
        )

    def summary(self) -> str:
        """Return a human-readable summary of the graph."""
        lines = [repr(self)]
        for node in self._nodes:
            lines.append(f"  {node}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"IRGraph("
            f"{len(self._nodes)} nodes, "
            f"{len(self.input_node_ids)} inputs, "
            f"{len(self.output_node_ids)} outputs, "
            f"{self.op_count()} ops)"
        )
