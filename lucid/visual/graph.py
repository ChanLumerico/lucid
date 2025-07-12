import networkx as nx
import matplotlib.pyplot as plt

import lucid.nn as nn
from lucid._tensor import Tensor


__all__ = ["draw_tensor_graph"]


def draw_tensor_graph(
    tensor: Tensor, horizontal: bool = False, title: str | None = None
) -> plt.Figure:
    G = nx.DiGraph()
    visited = set()
    result_id = id(tensor)

    def build(t: Tensor) -> None:
        if id(t) in visited:
            return
        visited.add(id(t))

        if not t.is_leaf:
            op = t._op
            if op is not None:
                op_id = id(op)
                op_label = type(op).__name__
                G.add_node(op_id, label=op_label, shape="circle", color="lightgreen")
                G.add_edge(op_id, id(t))
                for inp in t._prev:
                    build(inp)
                    G.add_edge(id(inp), op_id)

        shape_label = str(t.shape) if t.ndim > 0 else str(t.item())
        if isinstance(t, nn.Parameter):
            color = "pink"
        else:
            color = (
                "orange"
                if id(t) == result_id
                else "lightgray" if not t.requires_grad else "lightblue"
            )

        G.add_node(id(t), label=shape_label, shape="rectangle", color=color)

    def grid_layout(
        G: nx.DiGraph, horizontal: bool = False
    ) -> tuple[dict, tuple, float, int]:
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1

        level_nodes = {}
        for node, level in levels.items():
            level_nodes.setdefault(level, []).append(node)

        def autoscale(
            level_nodes: dict[int, list[int]],
            horizontal: bool = False,
            base_size: float = 0.5,
            base_nodesize: int = 500,
        ) -> tuple[tuple[float, float], float, int]:
            num_levels = len(level_nodes)
            max_width = max(len(nodes) for nodes in level_nodes.values())
            node_count = sum(len(nodes) for nodes in level_nodes.values())

            if horizontal:
                fig_w = min(32, max(4.0, base_size * num_levels))
                fig_h = min(32, max(4.0, base_size * max_width))
            else:
                fig_w = min(32, max(4.0, base_size * max_width))
                fig_h = min(32, max(4.0, base_size * num_levels))

            nodesize = (
                base_nodesize if node_count <= 40 else base_nodesize * (40 / node_count)
            )
            fontsize = max(5, min(8, int(80 / node_count)))
            return (fig_w, fig_h), nodesize, fontsize

        figsize, nodesize, fontsize = autoscale(level_nodes, horizontal)
        pos = {}
        for level, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                if horizontal:
                    pos[node] = (level * 2.5, -i * 2.0)
                else:
                    pos[node] = (i * 2.5, -level * 2.0)

        return pos, figsize, nodesize, fontsize

    build(tensor)

    labels = nx.get_node_attributes(G, "label")
    colors = nx.get_node_attributes(G, "color")
    shapes = nx.get_node_attributes(G, "shape")
    pos, figsize, nodesize, fontsize = grid_layout(G, horizontal=horizontal)

    fig, ax = plt.subplots(figsize=figsize)

    rect_nodes = [n for n in G.nodes() if shapes.get(n) == "rectangle"]
    circ_nodes = [n for n in G.nodes() if shapes.get(n) == "circle"]
    rect_colors = [colors[n] for n in rect_nodes]

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
    plt.tight_layout()

    return fig
