"""Engine sub-namespace stubs vs the loaded engine (LCD-402).

``_nn_namespace``, ``_linalg_namespace`` and the rest of ``lucid/_C/engine.pyi``'s
sub-namespace classes are typed out in gen_pyi's template, so regenerating
cannot correct them and the freshness diff cannot see them drift:
``_linalg_namespace`` had declared a ``foo`` the engine never had and omitted
the real ``matrix_power``.  ``tools/check_stubs.py`` runs this on every generated
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
