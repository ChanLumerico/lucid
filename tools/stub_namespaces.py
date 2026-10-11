"""Engine sub-namespace stubs vs the loaded engine (LCD-402).

``_nn_namespace``, ``_linalg_namespace`` and the rest of ``lucid/_C/engine.pyi``'s
sub-namespace classes are typed out in gen_pyi's template, so regenerating
cannot correct them and the freshness diff cannot see them drift:
``_linalg_namespace`` declared a ``foo`` the engine never had and omitted the
real ``matrix_power``.  ``tools/check_stubs.py`` runs this on every generated
stub.  Stdlib only, so the check type-checks on its own.
"""

import ast
import re


def _split_top(params: str) -> list[str]:
    out: list[str] = []
    depth, cur = 0, ""
    for ch in params + ",":
        if ch == "," and depth == 0:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
            continue
        depth += ch in "[("
        depth -= ch in "])"
        cur += ch
    return out


def _runtime_params(name: str, fn: object) -> list[str] | None:
    """Parameter names from pybind11's first docstring line; None when overloaded."""
    doc = getattr(fn, "__doc__", None) or ""
    first = doc.splitlines()[0] if doc else ""
    m = re.match(rf"^{re.escape(name)}\((?P<params>.*)\)\s*->", first)
    if m is None:
        return None
    return [p.split(":")[0].split("=")[0].strip() for p in _split_top(m["params"])]


def _stub_params(fn: ast.FunctionDef) -> list[str]:
    a = fn.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    if a.vararg is not None:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg is not None:
        names.append("**" + a.kwarg.arg)
    return names


def _stub_members(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | None]:
    members: dict[str, ast.FunctionDef | None] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            members[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members[node.target.id] = None
        elif isinstance(node, ast.Assign):
            members.update({t.id: None for t in node.targets if isinstance(t, ast.Name)})
    return members


def _namespace_attrs(body: list[ast.stmt], classes: set[str]) -> dict[str, str]:
    """``attr: _x_namespace`` declarations in *body*, attribute -> class name."""
    return {
        node.target.id: node.annotation.id
        for node in body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.annotation, ast.Name)
        and node.annotation.id in classes
    }


def namespace_drift(stub: str, engine: object) -> list[str]:
    """Every disagreement between the stub's sub-namespace classes and *engine*."""
    tree = ast.parse(stub)
    classes = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name.endswith("_namespace")
    }
    todo = [(attr, cls, engine) for attr, cls in _namespace_attrs(tree.body, set(classes)).items()]
    problems: list[str] = []
    while todo:
        path, cls_name, parent = todo.pop()
        runtime = getattr(parent, path.rsplit(".", 1)[-1], None)
        if runtime is None:  # optional build feature (``quantized``): hasattr-guarded
            continue
        cls = classes[cls_name]
        members = _stub_members(cls)
        have = {n for n in dir(runtime) if not n.startswith("__")}
        problems += [
            f"{path}.{n}: in the stub, not in the engine" for n in sorted(set(members) - have)
        ]
        problems += [
            f"{path}.{n}: in the engine, not in the stub" for n in sorted(have - set(members))
        ]
        for name, fn in members.items():
            if fn is None or name not in have:
                continue
            want = _runtime_params(name, getattr(runtime, name))
            got = _stub_params(fn)
            if want is not None and got != want:
                problems.append(
                    f"{path}.{name}: stub ({', '.join(got)}), engine ({', '.join(want)})"
                )
        todo += [
            (f"{path}.{attr}", sub, runtime)
            for attr, sub in _namespace_attrs(cls.body, set(classes)).items()
        ]
    return problems


# Drift that existed when the check arrived, pinned so the gate is green while
# the template fix waits for tools/gen_pyi.py to be free (LCD-402).  Strict
# both ways: a new mismatch fails, and so does a pin that no longer occurs —
# the fix commit empties this set.
KNOWN_DRIFT: frozenset[str] = frozenset(
    (
        "einops.einsum: stub (pattern, *tensors), engine (pattern, operands)",
        "linalg.foo: in the stub, not in the engine",
        "linalg.householder_product: stub (a, tau), engine (H, tau)",
        "linalg.lu_solve: stub (b, lu, piv), engine (LU, pivots, b)",
        "linalg.matrix_power: in the engine, not in the stub",
        "nn.adaptive_avg_pool1d: stub (x, sz), engine (x, output_l)",
        "nn.adaptive_avg_pool2d: stub (x, oh, ow), engine (x, output_h, output_w)",
        "nn.adaptive_avg_pool3d: stub (x, od, oh, ow), engine (x, output_d, output_h, output_w)",
        "nn.adaptive_max_pool1d: stub (x, sz), engine (x, output_l)",
        "nn.adaptive_max_pool2d: stub (x, oh, ow), engine (x, output_h, output_w)",
        "nn.adaptive_max_pool3d: stub (x, od, oh, ow), engine (x, output_d, output_h, output_w)",
        "nn.alpha_dropout: stub (x, p, training), engine (a, p, training, generator)",
        "nn.batch_norm_eval: stub (x, rm, rv, w, b, eps), engine (x, mean, var, gamma, beta, eps)",
        "nn.bce_loss: in the engine, not in the stub",
        "nn.bce_with_logits: in the engine, not in the stub",
        "nn.bilinear_layer: stub (x1, x2, w), engine (x1, x2, weight, bias)",
        "nn.conv_transpose1d: stub (x, w, b, stride, pad, opad, dilation, groups), engine (x, W, b, stride_l, pad_l, opad_l, dilation_l, groups)",
        "nn.conv_transpose2d: stub (x, w, b, stride_h, stride_w, pad_h, pad_w, opad_h, opad_w, dil_h, dil_w, groups), engine (x, W, b, stride_h, stride_w, pad_h, pad_w, opad_h, opad_w, dilation_h, dilation_w, groups)",
        "nn.conv_transpose3d: stub (x, w, b, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, opad_d, opad_h, opad_w, dil_d, dil_h, dil_w, groups), engine (x, W, b, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, opad_d, opad_h, opad_w, dilation_d, dilation_h, dilation_w, groups)",
        "nn.conv1d: stub (x, w, b, stride, padding, dilation, groups), engine (x, W, b, stride_l, pad_l, dilation_l, groups)",
        "nn.conv2d: stub (x, w, b, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups), engine (x, W, b, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups)",
        "nn.conv3d: stub (x, w, b, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups), engine (x, W, b, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w, groups)",
        "nn.cross_entropy_loss: in the engine, not in the stub",
        "nn.ctc_loss: stub (input, targets, input_lengths, target_lengths, blank, zero_infinity), engine (log_probs, targets, input_lengths, target_lengths, blank, zero_infinity)",
        "nn.drop_block: in the engine, not in the stub",
        "nn.drop_path: in the engine, not in the stub",
        "nn.dropout: stub (x, p, training), engine (a, p, training, generator)",
        "nn.dropoutnd: stub (x, p, training), engine (a, p, training, generator)",
        "nn.embedding_bag: stub (w, x, offsets, mode, padding_idx, include_last_offset), engine (weight, indices, offsets, mode, padding_idx, include_last_offset)",
        "nn.embedding: stub (w, x, pad), engine (weight, indices, padding_idx)",
        "nn.global_response_norm: in the engine, not in the stub",
        "nn.grid_sample: stub (x, grid, mode, padding_mode, align_corners), engine (input, grid, mode, padding_mode, align_corners)",
        "nn.group_norm: stub (x, w, b, num_groups, eps), engine (x, gamma, beta, num_groups, eps)",
        "nn.interpolate_bilinear: stub (x, oh, ow, align_corners), engine (input, H_out, W_out, align_corners)",
        "nn.interpolate_nearest_2d: stub (x, oh, ow), engine (input, H_out, W_out)",
        "nn.interpolate_nearest_3d: stub (x, od, oh, ow), engine (input, D_out, H_out, W_out)",
        "nn.interpolate_trilinear: stub (x, od, oh, ow, align_corners), engine (input, D_out, H_out, W_out, align_corners)",
        "nn.layer_norm: stub (x, w, b, eps), engine (x, gamma, beta, eps)",
        "nn.linear: stub (x, w, b), engine (x, W, b)",
        "nn.lstm_forward: stub (x, h0, c0, weights, hidden_size, num_layers, batch_first, bidirectional, training, proj_size), engine (input, h0, c0, weights, hidden_size, num_layers, batch_first, bidirectional, has_bias, proj_size)",
        "nn.nll_loss: in the engine, not in the stub",
        "nn.one_hot: stub (x, num_classes), engine (input, num_classes, dtype)",
        "nn.rms_norm: stub (x, w, eps), engine (x, gamma, eps)",
        "nn.rotary_pos_embedding: in the engine, not in the stub",
        "nn.rotate: in the engine, not in the stub",
        "nn.scaled_dot_product_attention_with_weights: in the engine, not in the stub",
        "nn.scaled_dot_product_attention: stub (q, k, v, mask, scale, is_causal), engine (query, key, value, attn_mask, scale, is_causal)",
        "nn.sinusoidal_pos_embedding: in the engine, not in the stub",
    )
)


def unpinned(problems: list[str]) -> list[str]:
    """*problems* not in KNOWN_DRIFT, then every pin that no longer occurs."""
    found = set(problems)
    return [p for p in problems if p not in KNOWN_DRIFT] + [
        f"pinned in KNOWN_DRIFT but no longer occurs (remove it): {p}"
        for p in sorted(KNOWN_DRIFT - found)
    ]
