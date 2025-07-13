from typing import Union
import networkx as nx
import matplotlib.pyplot as plt

import lucid.nn as nn
from lucid._tensor import Tensor


__all__ = ["draw_tensor_graph"]


def draw_tensor_graph(
    tensor: Tensor,
    horizontal: bool = False,
    title: Union[str, None] = None,
    start_id: Union[int, None] = None,
) -> plt.Figure:
    G: nx.DiGraph = nx.DiGraph()
    result_id: int = id(tensor)

    visited: set[int] = set()
    nodes_to_draw: list[Tensor] = []

    def dfs(t: Tensor) -> None:
        if id(t) in visited:
            return
        visited.add(id(t))
        for p in t._prev:
            dfs(p)
        nodes_to_draw.append(t)

    dfs(tensor)

    for t in nodes_to_draw:
        if not t.is_leaf and t._op is not None:
            op_id: int = id(t._op)
            op_label: str = type(t._op).__name__
            G.add_node(op_id, label=op_label, shape="circle", color="lightgreen")
            G.add_edge(op_id, id(t))
            for inp in t._prev:
                G.add_edge(id(inp), op_id)

        shape_label: str = str(t.shape) if t.ndim > 0 else str(t.item())
        if isinstance(t, nn.Parameter):
            color: str = "plum"
        else:
            color = (
                "lightcoral"
                if id(t) == result_id
                else "lightgray" if not t.requires_grad else "lightblue"
            )
        if start_id is not None and id(t) == start_id:
            color = "gold"

        G.add_node(id(t), label=shape_label, shape="rectangle", color=color)

    def grid_layout(
        G: nx.DiGraph, horizontal: bool = False
    ) -> tuple[dict, tuple, float, int]:
        levels: dict[int, int] = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1

        level_nodes: dict[int, list[int]] = {}
        for node, level in levels.items():
            level_nodes.setdefault(level, []).append(node)

        def autoscale(
            level_nodes: dict[int, list[int]],
            horizontal: bool = False,
            base_size: float = 0.5,
            base_nodesize: int = 500,
        ) -> tuple[tuple[float, float], float, int]:
            num_levels: int = len(level_nodes)
            max_width: int = max(len(nodes) for nodes in level_nodes.values())
            node_count: int = sum(len(nodes) for nodes in level_nodes.values())

            if horizontal:
                fig_w: float = min(32, max(4.0, base_size * num_levels))
                fig_h: float = min(32, max(4.0, base_size * max_width))
            else:
                fig_w = min(32, max(4.0, base_size * max_width))
                fig_h = min(32, max(4.0, base_size * num_levels))

            nodesize: float = (
                base_nodesize
                if node_count <= 100
                else base_nodesize * (100 / node_count)
            )
            fontsize: int = max(5, min(8, int(80 / node_count)))
            return (fig_w, fig_h), nodesize, fontsize

        figsize, nodesize, fontsize = autoscale(level_nodes, horizontal)
        pos: dict[int, tuple[float, float]] = {}
        for level, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                pos[node] = (
                    (level * 2.5, -i * 2.0) if horizontal else (i * 2.5, -level * 2.0)
                )
        return pos, figsize, nodesize, fontsize

    labels: dict[int, str] = nx.get_node_attributes(G, "label")
    colors: dict[int, str] = nx.get_node_attributes(G, "color")
    shapes: dict[int, str] = nx.get_node_attributes(G, "shape")
    pos, figsize, nodesize, fontsize = grid_layout(G, horizontal)

    fig, ax = plt.subplots(figsize=figsize)

    rect_nodes: list[int] = [n for n in G.nodes() if shapes.get(n) == "rectangle"]
    circ_nodes: list[int] = [n for n in G.nodes() if shapes.get(n) == "circle"]
    rect_colors: list[str] = [colors[n] for n in rect_nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=rect_nodes,
        node_color=rect_colors,
        node_size=nodesize,
        node_shape="s",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=circ_nodes,
        node_color="lightgreen",
        node_size=nodesize,
        node_shape="o",
        ax=ax,
    )
    nx.draw_networkx_edges(G, pos, width=0.5, arrows=True, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=fontsize, ax=ax)

    ax.axis("off")
    ax.set_title(title if title is not None else "")

    return fig
