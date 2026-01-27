from dataclasses import dataclass
from typing import Iterable, Literal

import lucid
import lucid.nn as nn
from lucid._tensor import Tensor
from lucid.types import _ShapeLike


__all__ = ["build_mermaid_chart"]


@dataclass
class _ModuleNode:
    module: nn.Module
    name: str
    depth: int
    children: list["_ModuleNode"]


def _flatten_tensors(obj: object) -> list[Tensor]:
    tensors: list[Tensor] = []

    if isinstance(obj, Tensor):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_flatten_tensors(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            tensors.extend(_flatten_tensors(item))

    return tensors


def _build_tree(
    module: nn.Module, depth: int, max_depth: int, name: str = ""
) -> _ModuleNode:
    children: list[_ModuleNode] = []
    if depth < max_depth:
        for child_name, child in module._modules.items():
            path = f"{name}.{child_name}" if name else child_name
            children.append(_build_tree(child, depth + 1, max_depth, path))

    return _ModuleNode(module=module, name=name, depth=depth, children=children)


def _module_label(module: nn.Module, show_params: bool) -> str:
    class_name = type(module).__name__
    if show_params:
        return f"{class_name} ({module.parameter_size:,} params)"
    return class_name


def build_mermaid_chart(
    module: nn.Module,
    input_shape: _ShapeLike | list[_ShapeLike] | None = None,
    inputs: Iterable[Tensor] | Tensor | None = None,
    depth: int = 1,
    direction: str = "LR",
    include_io: bool = True,
    show_params: bool = False,
    return_lines: bool = False,
    copy_to_clipboard: bool = False,
    compact: bool = False,
    use_class_defs: bool = False,
    end_semicolons: bool = True,
    edge_mode: Literal["dataflow", "execution"] = "execution",
    **forward_kwargs,
) -> str | list[str]:
    if inputs is None and input_shape is None:
        raise ValueError("Either inputs or input_shape must be provided.")
    if depth < 0:
        raise ValueError("depth must be >= 0")

    tree = _build_tree(module, depth=0, max_depth=depth)

    nodes: list[_ModuleNode] = []

    def _collect(n: _ModuleNode) -> None:
        nodes.append(n)
        for c in n.children:
            _collect(c)

    _collect(tree)

    module_to_id: dict[nn.Module, str] = {}
    for idx, n in enumerate(nodes):
        module_to_id[n.module] = f"m{idx}"

    def _build_parent_map(root: nn.Module) -> dict[nn.Module, nn.Module]:
        parent: dict[nn.Module, nn.Module] = {}

        def _walk(mod: nn.Module) -> None:
            for child in mod._modules.values():
                parent[child] = mod
                _walk(child)

        _walk(root)
        return parent

    parent_map = _build_parent_map(module)

    def _map_to_included(mod: nn.Module) -> nn.Module | None:
        cur = mod
        while cur not in module_to_id and cur in parent_map:
            cur = parent_map[cur]
        return cur if cur in module_to_id else None

    hooks = []
    edges: set[tuple[str, str]] = set()
    tensor_producer: dict[int, nn.Module] = {}
    input_node_id = "input"
    output_node_id = "output"
    root_module = module
    exec_order: list[nn.Module] = []

    def _hook(
        mod: nn.Module, input_arg: tuple, output: Tensor | tuple[Tensor, ...]
    ) -> None:
        mapped_mod = _map_to_included(mod)
        if mapped_mod is None:
            return

        if edge_mode == "dataflow":
            input_tensors = _flatten_tensors(input_arg)
            for t in input_tensors:
                producer = tensor_producer.get(id(t))
                if producer is None:
                    if include_io and mapped_mod is not root_module:
                        edges.add((input_node_id, module_to_id[mapped_mod]))
                else:
                    if producer is not mapped_mod:
                        edges.add((module_to_id[producer], module_to_id[mapped_mod]))

        output_tensors = _flatten_tensors(output)
        for t in output_tensors:
            tensor_producer[id(t)] = mapped_mod

        exec_order.append(mapped_mod)

    for mod in module_to_id:
        hooks.append(mod.register_forward_hook(_hook))

    try:
        if inputs is None:
            if isinstance(input_shape, list):
                input_tensors = [
                    lucid.random.rand(shape, device=module.device)
                    for shape in input_shape
                ]
            else:
                input_tensors = [lucid.random.rand(input_shape, device=module.device)]
        else:
            if isinstance(inputs, Tensor):
                input_tensors = [inputs]
            else:
                input_tensors = list(inputs)

        outputs = module(*input_tensors, **forward_kwargs)
    finally:
        for remove in hooks:
            remove()

    if edge_mode == "execution":
        seq = []
        for m in exec_order:
            if not seq or seq[-1] is not m:
                seq.append(m)
        for prev, cur in zip(seq, seq[1:]):
            edges.add((module_to_id[prev], module_to_id[cur]))
        if include_io and seq:
            if seq[0] is not root_module:
                edges.add((input_node_id, module_to_id[seq[0]]))
            if seq[-1] is not root_module:
                edges.add((module_to_id[seq[-1]], output_node_id))
    else:
        if include_io:
            output_tensors = _flatten_tensors(outputs)
            for t in output_tensors:
                producer = tensor_producer.get(id(t))
                if producer is not None and producer is not root_module:
                    edges.add((module_to_id[producer], output_node_id))

    node_ids = set(module_to_id.values())
    nodes_with_edges: set[str] = set()
    for src, dst in edges:
        if src in node_ids:
            nodes_with_edges.add(src)
        if dst in node_ids:
            nodes_with_edges.add(dst)

    lines: list[str] = []
    lines.append(f"flowchart {direction}")
    if use_class_defs:
        lines.append("  classDef module fill:#f9f9f9,stroke:#333,stroke-width:1px;")
        lines.append("  classDef io fill:#fff3cd,stroke:#a67c00,stroke-width:1px;")
        lines.append(
            "  classDef anchor fill:transparent,stroke:transparent,color:transparent;"
        )

    def _render(n: _ModuleNode, indent: str = "  ") -> None:
        node_id = module_to_id[n.module]
        label = _module_label(n.module, show_params)
        if n.children:
            lines.append(f'{indent}subgraph sg_{node_id}["{label}"]')
            if use_class_defs:
                lines.append(f'{indent}  {node_id}[""]:::anchor')
            elif node_id in nodes_with_edges:
                lines.append(f'{indent}  {node_id}["{label}"]')
            for c in n.children:
                _render(c, indent + "  ")
            lines.append(f"{indent}end")
        else:
            if use_class_defs:
                lines.append(f'{indent}{node_id}["{label}"]:::module')
            else:
                lines.append(f'{indent}{node_id}["{label}"]')

    _render(tree)

    if include_io:
        if use_class_defs:
            lines.append(f'  {input_node_id}["Input"]:::io')
            lines.append(f'  {output_node_id}["Output"]:::io')
        else:
            lines.append(f'  {input_node_id}["Input"]')
            lines.append(f'  {output_node_id}["Output"]')

    for src, dst in sorted(edges):
        lines.append(f"  {src} --> {dst}")

    def _finalize_lines(src_lines: list[str]) -> list[str]:
        if not end_semicolons:
            return src_lines
        out: list[str] = []
        for line in src_lines:
            stripped = line.rstrip()
            if stripped.endswith(";"):
                out.append(line)
            else:
                out.append(f"{line};")
        return out

    final_lines = _finalize_lines(lines)

    if compact:
        text = " ".join(final_lines)
    else:
        text = "\n".join(final_lines)
    if copy_to_clipboard:
        _copy_to_clipboard(text)

    if return_lines:
        return final_lines
    return text


def _copy_to_clipboard(text: str) -> None:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
    except Exception:
        pass
