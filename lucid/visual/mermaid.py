from dataclasses import dataclass
import json
from typing import Iterable, Literal

import lucid
import lucid.nn as nn

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


__all__ = ["build_mermaid_chart"]


_NN_MODULES_PREFIX = "lucid.nn.modules."


# fmt: off
_BUILTIN_SUBPACKAGE_STYLE: dict[str, tuple[str, str]] = {
    "conv":         ("#ffe8e8", "#c53030"),
    "norm":         ("#e6fffa", "#2c7a7b"),
    "activation":   ("#faf5ff", "#6b46c1"),
    "linear":       ("#ebf8ff", "#2b6cb0"),
    "pool":         ("#fefcbf", "#b7791f"),
    "drop":         ("#edf2f7", "#4a5568"),
    "transformer":  ("#e2e8f0", "#334155"),
    "attention":    ("#f0fff4", "#2f855a"),
    "vision":       ("#fdf2f8", "#b83280"),
    "rnn":          ("#f0f9ff", "#0284c7"),
    "sparse":       ("#f1f5f9", "#475569"),
    "loss":         ("#fffbeb", "#d97706"),
    "einops":       ("#ecfccb", "#65a30d"),
}
# fmt: on


@dataclass
class _ModuleNode:
    module: nn.Module
    name: str
    depth: int
    children: list["_ModuleNode"]
    group: list[nn.Module] | None = None

    @property
    def count(self) -> int:
        return 1 if self.group is None else len(self.group)

    def iter_modules(self) -> Iterable[nn.Module]:
        if self.group is None:
            yield self.module
            return
        yield from self.group


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
    module: nn.Module,
    depth: int,
    max_depth: int,
    name: str = "",
    *,
    collapse_repeats: bool = False,
    repeat_min: int = 3,
    hide_subpackages: set[str] | None = None,
    hide_module_names: set[str] | None = None,
) -> _ModuleNode:
    children: list[_ModuleNode] = []
    if depth < max_depth:
        for child_name, child in module._modules.items():
            path = f"{name}.{child_name}" if name else child_name

            node = _build_tree(
                child,
                depth + 1,
                max_depth,
                path,
                collapse_repeats=collapse_repeats,
                repeat_min=repeat_min,
                hide_subpackages=hide_subpackages,
                hide_module_names=hide_module_names,
            )

            child_cls_name = type(child).__name__
            child_mod_path = type(child).__module__
            child_subpkg = None

            if child_mod_path.startswith(_NN_MODULES_PREFIX):
                rest = child_mod_path[len(_NN_MODULES_PREFIX) :]
                child_subpkg = rest.split(".", 1)[0] if rest else None

            excluded = False
            if hide_module_names and child_cls_name in hide_module_names:
                excluded = True
            if hide_subpackages and child_subpkg and child_subpkg in hide_subpackages:
                excluded = True

            if excluded:
                children.extend(node.children)
            else:
                children.append(node)

    if collapse_repeats and children:
        children = _collapse_repeated_children(children, repeat_min=repeat_min)

    return _ModuleNode(module=module, name=name, depth=depth, children=children)


def _module_label(module: nn.Module, show_params: bool) -> str:
    class_name = getattr(module, "_alt_name", "") or type(module).__name__
    if show_params:
        return f"{class_name} ({module.parameter_size:,} params)"

    return class_name


def _builtin_subpackage_key(module: nn.Module) -> str | None:
    mod_path = type(module).__module__
    if not mod_path.startswith(_NN_MODULES_PREFIX):
        return None

    rest = mod_path[len(_NN_MODULES_PREFIX) :]
    return rest.split(".", 1)[0] if rest else None


def _shape_text_color(module: nn.Module) -> str | None:
    subpkg = _builtin_subpackage_key(module)
    if subpkg is None:
        return None

    style = _BUILTIN_SUBPACKAGE_STYLE.get(subpkg)
    if style is None:
        return None

    _, stroke = style
    return stroke


def _parse_rgba(value: str) -> tuple[str, float] | None:
    v = value.strip().lower()
    if not v.startswith("rgba(") or not v.endswith(")"):
        return None

    inner = v[5:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 4:
        return None

    try:
        r, g, b = (int(float(x)) for x in parts[:3])
        a = float(parts[3])
    except Exception:
        return None

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0.0, min(1.0, a))

    return (f"#{r:02x}{g:02x}{b:02x}", a)


def _container_attr_label(node: _ModuleNode) -> str | None:
    if not isinstance(node.module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
        return None
    if not node.name:
        return None

    leaf = node.name.rsplit(".", 1)[-1]
    if leaf.isdigit():
        return None

    return leaf


def _escape_label(text: str) -> str:
    return text.replace('"', "&quot;")


def _shape_str(shape: object) -> str:
    try:
        if isinstance(shape, tuple):
            return "(" + ",".join(str(x) for x in shape) + ")"
    except Exception:
        pass
    return str(shape)


def _flatten_shapes(obj: object) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for t in _flatten_tensors(obj):
        try:
            shapes.append(tuple(int(x) for x in t.shape))
        except Exception:
            continue
    return shapes


def _shapes_brief(shapes: list[tuple[int, ...]]) -> str:
    if not shapes:
        return "?"
    if len(shapes) == 1:
        return _shape_str(shapes[0])
    return f"{_shape_str(shapes[0])}x{len(shapes)}"


def _node_signature(node: _ModuleNode) -> tuple:
    return (type(node.module), tuple(_node_signature(c) for c in node.children))


def _collapse_repeated_children(
    children: list[_ModuleNode], *, repeat_min: int
) -> list[_ModuleNode]:
    if repeat_min <= 1:
        return children

    out: list[_ModuleNode] = []
    i = 0
    while i < len(children):
        base = children[i]
        base_sig = _node_signature(base)
        j = i + 1
        while j < len(children) and _node_signature(children[j]) == base_sig:
            j += 1

        run = children[i:j]
        if len(run) >= repeat_min:
            out.append(
                _ModuleNode(
                    module=base.module,
                    name=base.name,
                    depth=base.depth,
                    children=base.children,
                    group=[n.module for n in run],
                )
            )
        else:
            out.extend(run)

        i = j
    return out


def build_mermaid_chart(
    module: nn.Module,
    input_shape: _ShapeLike | list[_ShapeLike] | None = None,
    inputs: Iterable[Tensor] | Tensor | None = None,
    depth: int = 2,
    direction: str = "LR",
    include_io: bool = True,
    show_params: bool = False,
    return_lines: bool = False,
    copy_to_clipboard: bool = False,
    compact: bool = False,
    use_class_defs: bool = False,
    end_semicolons: bool = True,
    edge_mode: Literal["dataflow", "execution"] = "execution",
    collapse_repeats: bool = True,
    repeat_min: int = 2,
    color_by_subpackage: bool = True,
    container_name_from_attr: bool = True,
    edge_stroke_width: float = 2.0,
    emphasize_model_title: bool = True,
    model_title_font_px: int = 20,
    show_shapes: bool = False,
    hide_subpackages: Iterable[str] = (),
    hide_module_names: Iterable[str] = (),
    dash_multi_input_edges: bool = True,
    subgraph_fill: str = "#000000",
    subgraph_fill_opacity: float = 0.05,
    subgraph_stroke: str = "#000000",
    subgraph_stroke_opacity: float = 0.75,
    force_text_color: str | None = None,
    edge_curve: Literal["basis", "linear", "step"] = "step",
    node_spacing: int = 50,
    rank_spacing: int = 50,
    **forward_kwargs,
) -> str | list[str]:
    if inputs is None and input_shape is None:
        raise ValueError("Either inputs or input_shape must be provided.")
    if depth < 0:
        raise ValueError("depth must be >= 0")

    tree = _build_tree(
        module,
        depth=0,
        max_depth=depth,
        collapse_repeats=collapse_repeats,
        repeat_min=repeat_min,
        hide_subpackages=set(hide_subpackages),
        hide_module_names=set(hide_module_names),
    )

    nodes: list[_ModuleNode] = []

    def _collect(n: _ModuleNode) -> None:
        nodes.append(n)
        for c in n.children:
            _collect(c)

    _collect(tree)

    module_to_node: dict[nn.Module, _ModuleNode] = {n.module: n for n in nodes}

    module_to_id: dict[nn.Module, str] = {}
    for idx, n in enumerate(nodes):
        node_id = f"m{idx}"
        for mod in n.iter_modules():
            module_to_id[mod] = node_id

    def _build_parent_map(root: nn.Module) -> dict[nn.Module, nn.Module]:
        parent: dict[nn.Module, nn.Module] = {}

        def _walk(mod: nn.Module) -> None:
            for child in mod._modules.values():
                parent[child] = mod
                _walk(child)

        _walk(root)
        return parent

    parent_map = _build_parent_map(module)
    has_non_root_included = any(mod is not module for mod in module_to_id)

    def _map_to_included(mod: nn.Module) -> nn.Module | None:
        cur = mod
        while cur not in module_to_id and cur in parent_map:
            cur = parent_map[cur]
        return cur if cur in module_to_id else None

    hooks = []
    edges: set[tuple[str, str]] = set()
    tensor_producer: dict[int, nn.Module] = {}
    module_in_shapes: dict[nn.Module, list[tuple[int, ...]]] = {}
    module_out_shapes: dict[nn.Module, list[tuple[int, ...]]] = {}
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

        if show_shapes:
            module_in_shapes[mapped_mod] = _flatten_shapes(input_arg)
            module_out_shapes[mapped_mod] = _flatten_shapes(output)

        if edge_mode == "dataflow":
            input_tensors = _flatten_tensors(input_arg)
            for t in input_tensors:
                producer = tensor_producer.get(id(t))
                if producer is None:
                    if include_io and (
                        mapped_mod is not root_module or not has_non_root_included
                    ):
                        edges.add((input_node_id, module_to_id[mapped_mod]))

                else:
                    if producer is not mapped_mod:
                        edges.add((module_to_id[producer], module_to_id[mapped_mod]))

        output_tensors = _flatten_tensors(output)
        for t in output_tensors:
            key = id(t)
            if mapped_mod is root_module and key in tensor_producer:
                continue
            tensor_producer[key] = mapped_mod

        exec_order.append(mapped_mod)

    for mod in module.modules():
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

    model_input_shapes = _flatten_shapes(input_tensors)
    model_output_shapes = _flatten_shapes(outputs)

    if edge_mode == "execution":
        seq = []
        for m in exec_order:
            if not seq or seq[-1] is not m:
                seq.append(m)
        seq_no_root = [m for m in seq if m is not root_module]
        if not seq_no_root:
            seq_no_root = [root_module]

        def _is_included_container(mod: nn.Module) -> bool:
            node = module_to_node.get(mod)
            return bool(node and node.children)

        seq_effective = [m for m in seq_no_root if not _is_included_container(m)]
        if not seq_effective:
            seq_effective = seq_no_root

        for prev, cur in zip(seq_effective, seq_effective[1:]):
            edges.add((module_to_id[prev], module_to_id[cur]))

        if include_io and seq_effective:
            edges.add((input_node_id, module_to_id[seq_effective[0]]))
            edges.add((module_to_id[seq_effective[-1]], output_node_id))

    else:
        if include_io:
            output_tensors = _flatten_tensors(outputs)
            for t in output_tensors:
                producer = tensor_producer.get(id(t))
                if producer is not None:
                    edges.add((module_to_id[producer], output_node_id))

    node_ids = set(module_to_id.values())
    nodes_with_edges: set[str] = set()

    for src, dst in edges:
        if src in node_ids:
            nodes_with_edges.add(src)
        if dst in node_ids:
            nodes_with_edges.add(dst)

    container_node_ids = {module_to_id[n.module] for n in nodes if n.children}
    container_with_edges = container_node_ids & nodes_with_edges
    extra_edges: list[tuple[str, str]] = []

    def _endpoint_in(node_id: str) -> str:
        return f"{node_id}_in" if node_id in container_with_edges else node_id

    def _endpoint_out(node_id: str) -> str:
        return f"{node_id}_out" if node_id in container_with_edges else node_id

    def _first_leaf_id(n: _ModuleNode) -> str:
        cur = n
        while cur.children:
            cur = cur.children[0]
        return module_to_id[cur.module]

    def _last_leaf_id(n: _ModuleNode) -> str:
        cur = n
        while cur.children:
            cur = cur.children[-1]
        return module_to_id[cur.module]

    lines: list[str] = []
    init_cfg: dict[str, object] = {
        "flowchart": {
            "curve": edge_curve,
            "nodeSpacing": node_spacing,
            "rankSpacing": rank_spacing,
        }
    }
    if force_text_color:
        init_cfg["themeCSS"] = (
            f".nodeLabel, .edgeLabel, .cluster text, .node text "
            f"{{ fill: {force_text_color} !important; }} "
            f".node foreignObject *, .cluster foreignObject * "
            f"{{ color: {force_text_color} !important; }}"
        )

    lines.append(f"%%{{init: {json.dumps(init_cfg, separators=(',', ':'))} }}%%")
    lines.append(f"flowchart {direction}")

    if edge_stroke_width and edge_stroke_width != 1.0:
        lines.append(f"  linkStyle default stroke-width:{edge_stroke_width}px")

    if use_class_defs:
        lines.append("  classDef module fill:#f9f9f9,stroke:#333,stroke-width:1px;")
        lines.append("  classDef modelio fill:#fff3cd,stroke:#a67c00,stroke-width:1px;")
        lines.append(
            "  classDef internalio fill:#e2e8f0,stroke:#64748b,stroke-width:1px;"
        )
        lines.append(
            "  classDef anchor fill:transparent,stroke:transparent,color:transparent;"
        )
        lines.append("  classDef repeat fill:#e8f1ff,stroke:#2b6cb0,stroke-width:1px;")

    def _render(n: _ModuleNode, indent: str = "  ") -> None:
        node_id = module_to_id[n.module]
        base_label = _module_label(n.module, show_params)

        if container_name_from_attr and n.children:
            attr_label = _container_attr_label(n)
            if attr_label is not None:
                base_label = attr_label

        base_label = _escape_label(base_label)
        label_text = base_label if n.count == 1 else f"{base_label} x {n.count}"

        if show_shapes and not n.children:
            in_shapes = module_in_shapes.get(n.module, [])
            out_shapes = module_out_shapes.get(n.module, [])

            if in_shapes != out_shapes and (in_shapes or out_shapes):
                ins = _shapes_brief(in_shapes)
                outs = _shapes_brief(out_shapes)
                color_css = ""
                if not force_text_color:
                    color = _shape_text_color(n.module)
                    color_css = f"color:{color};" if color else ""
                label_text = (
                    f"{label_text}<br/>"
                    f"<span style='font-size:11px;{color_css}font-weight:400'>"
                    f"{ins} \u2192 {outs}"
                    f"</span>"
                )

        label = label_text
        if (
            emphasize_model_title
            and n.module is root_module
            and model_title_font_px
            and model_title_font_px > 0
        ):
            label = (
                f"<span style='font-size:{model_title_font_px}px;font-weight:700'>"
                f"{label}"
                f"</span>"
            )

        if n.children:
            lines.append(f'{indent}subgraph sg_{node_id}["{label}"]')
            if subgraph_fill or subgraph_stroke:
                parts: list[str] = []

                fill = subgraph_fill
                fill_opacity = subgraph_fill_opacity
                parsed = _parse_rgba(fill)

                if parsed is not None:
                    fill, fill_opacity = parsed
                if fill:
                    parts.append(f"fill:{fill}")
                    parts.append(f"fill-opacity:{fill_opacity}")

                stroke = subgraph_stroke
                stroke_opacity = subgraph_stroke_opacity
                if stroke:
                    parsed = _parse_rgba(stroke)
                    if parsed is not None:
                        stroke, stroke_opacity = parsed

                    parts.append(f"stroke:{stroke}")
                    parts.append(f"stroke-opacity:{stroke_opacity}")
                    parts.append("stroke-width:1px")

                lines.append(f'{indent}style sg_{node_id} {",".join(parts)}')

            if node_id in nodes_with_edges:
                if node_id in container_with_edges:
                    in_id = f"{node_id}_in"
                    out_id = f"{node_id}_out"

                    if use_class_defs:
                        lines.append(f'{indent}  {in_id}(["Input"]):::internalio')
                        lines.append(f'{indent}  {out_id}(["Output"]):::internalio')
                    else:
                        lines.append(f'{indent}  {in_id}(["Input"])')
                        lines.append(f'{indent}  {out_id}(["Output"])')
                        lines.append(
                            f"  style {in_id} fill:#e2e8f0,stroke:#64748b,stroke-width:1px;"
                        )
                        lines.append(
                            f"  style {out_id} fill:#e2e8f0,stroke:#64748b,stroke-width:1px;"
                        )

                    extra_edges.append((in_id, _endpoint_in(_first_leaf_id(n))))
                    extra_edges.append((_endpoint_out(_last_leaf_id(n)), out_id))

                else:
                    if use_class_defs:
                        lines.append(f'{indent}  {node_id}[""]:::anchor')
                    else:
                        lines.append(f'{indent}  {node_id}["\u200b"]')
                        lines.append(
                            f"  style {node_id} fill:transparent,stroke:transparent,color:transparent;"
                        )

            for c in n.children:
                _render(c, indent + "  ")
            lines.append(f"{indent}end")

        else:
            if use_class_defs:
                class_name = "module" if n.count == 1 else "repeat"
                lines.append(f'{indent}{node_id}["{label}"]:::{class_name}')
            else:
                if n.count > 1:
                    lines.append(f'{indent}{node_id}(["{label}"])')
                else:
                    lines.append(f'{indent}{node_id}["{label}"]')

    _render(tree)

    if include_io:
        input_label = "Input"
        output_label = "Output"
        if show_shapes:
            in_s = _shapes_brief(model_input_shapes)
            out_s = _shapes_brief(model_output_shapes)
            io_color = force_text_color or "#a67c00"
            input_label = (
                f"{input_label}<br/>"
                f"<span style='font-size:11px;color:{io_color};font-weight:400'>{in_s}</span>"
            )
            output_label = (
                f"{output_label}<br/>"
                f"<span style='font-size:11px;color:{io_color};font-weight:400'>{out_s}</span>"
            )
        if use_class_defs:
            lines.append(f'  {input_node_id}["{input_label}"]:::modelio')
            lines.append(f'  {output_node_id}["{output_label}"]:::modelio')
        else:
            lines.append(f'  {input_node_id}["{input_label}"]')
            lines.append(f'  {output_node_id}["{output_label}"]')
            lines.append(
                f"  style {input_node_id} fill:#fff3cd,stroke:#a67c00,stroke-width:1px;"
            )
            lines.append(
                f"  style {output_node_id} fill:#fff3cd,stroke:#a67c00,stroke-width:1px;"
            )

    if color_by_subpackage:
        for n in nodes:
            if n.children:
                continue

            subpkg = _builtin_subpackage_key(n.module)
            if subpkg is None:
                continue
            style = _BUILTIN_SUBPACKAGE_STYLE.get(subpkg)
            if style is None:
                continue

            fill, stroke = style
            node_id = module_to_id[n.module]
            if node_id in {input_node_id, output_node_id}:
                continue

            if node_id.endswith("_in") or node_id.endswith("_out"):
                continue
            lines.append(
                f"  style {node_id} fill:{fill},stroke:{stroke},stroke-width:1px;"
            )

    render_edges: set[tuple[str, str]] = set()
    for src, dst in edges:
        src_id = _endpoint_out(src)
        dst_id = _endpoint_in(dst)
        if src_id != dst_id:
            render_edges.add((src_id, dst_id))

    for src, dst in extra_edges:
        if src != dst:
            render_edges.add((src, dst))

    indegree: dict[str, int] = {}
    for _, dst in render_edges:
        indegree[dst] = indegree.get(dst, 0) + 1

    for src, dst in sorted(render_edges):
        arrow = "-.->" if dash_multi_input_edges and indegree.get(dst, 0) > 1 else "-->"
        lines.append(f"  {src} {arrow} {dst}")

    def _finalize_lines(src_lines: list[str]) -> list[str]:
        if not end_semicolons:
            return src_lines

        out: list[str] = []
        for line in src_lines:
            stripped = line.rstrip()
            head = stripped.lstrip()
            if (
                not head
                or head.startswith("flowchart ")
                or head.startswith("linkStyle ")
                or head.startswith("subgraph ")
                or head == "end"
                or head.startswith("classDef ")
                or head.startswith("class ")
                or head.startswith("style ")
                or head.startswith("%%")
            ):
                out.append(line)
                continue

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
    import os
    import shutil
    import subprocess
    import sys

    errors: list[str] = []

    def _try(cmd: list[str]) -> bool:
        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            return True
        except Exception as e:
            errors.append(f"{cmd!r}: {type(e).__name__}: {e}")
            return False

    if sys.platform == "darwin":
        if shutil.which("pbcopy") and _try(["pbcopy"]):
            return

    elif sys.platform.startswith("win"):
        if shutil.which("clip") and _try(["clip"]):
            return
        if shutil.which("powershell"):
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Set-Clipboard"],
                    input=text.encode("utf-8"),
                    check=True,
                )
                return
            except Exception as e:
                errors.append(f"powershell Set-Clipboard: {type(e).__name__}: {e}")

    else:
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            if _try(["wl-copy"]):
                return
        if shutil.which("xclip"):
            if _try(["xclip", "-selection", "clipboard"]):
                return
        if shutil.which("xsel"):
            if _try(["xsel", "--clipboard", "--input"]):
                return

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return

    except Exception as e:
        errors.append(f"tkinter: {type(e).__name__}: {e}")

    detail = "\n".join(f"- {msg}" for msg in errors) if errors else "- (no details)"
    raise RuntimeError(
        "Failed to copy to clipboard. Install a clipboard utility (macOS: pbcopy; "
        "Wayland: wl-copy; X11: xclip/xsel; Windows: clip/powershell) or enable tkinter.\n"
        f"{detail}"
    )
