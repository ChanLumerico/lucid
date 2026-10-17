"""engine: bindings in the engine stub that no production module reaches (DS-24).

A function of ``lucid/_C/engine.pyi``, at the top level (``cube``) or in a
submodule (``nn.bce_loss``), is *unbound* when no production Python file
refers to it.  A reference is an attribute path rooted at the engine module
(``_C_engine.nn.bce_loss``), also through an alias (``_R = _C_engine``,
``from lucid.linalg import _la``), a ``from lucid._C.engine import f``, or
``getattr(<engine path>, "f")``.  A name matched by attribute alone would let
``self.rotate`` keep ``nn.rotate`` alive, so the path must start at the engine.
Tests are not production: a binding only a test calls counts, unless
``config.ENGINE_TEST_ONLY_BINDINGS`` names it as test or tool infrastructure.

    python -m tools.quality_gate.collectors.bindings   # list them
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import Counts, GateError, Scope, scope_of, tracked_files

_ENGINE = "lucid._C.engine"
EnginePath = tuple[str, ...]  # attributes below the engine module; () is the module


def stub_functions(stub: str) -> set[str]:
    """The stub's functions as dotted names; ``gen_pyi`` renders a submodule
    ``nn`` as ``class _nn_namespace`` plus ``nn: _nn_namespace``."""
    tree = ast.parse(stub)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    found: set[str] = set()

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef):
                found.add(prefix + node.name)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Name)
                and node.annotation.id.endswith("_namespace")
                and node.annotation.id in classes
            ):
                walk(classes[node.annotation.id].body, f"{prefix}{node.target.id}.")

    walk(tree.body, "")
    return found


def _resolve(expr: ast.expr, aliases: dict[str, EnginePath]) -> EnginePath | None:
    if isinstance(expr, ast.Name):
        return aliases.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _resolve(expr.value, aliases)
        return None if base is None else (*base, expr.attr)
    return None


@dataclass
class _Module:
    """One production file: its engine aliases and the expressions that may use them."""

    name: str
    aliases: dict[str, EnginePath] = field(default_factory=dict)
    imports: list[tuple[str, str, str]] = field(default_factory=list)  # local, module, name
    assigns: list[tuple[str, ast.expr]] = field(default_factory=list)
    uses: list[tuple[ast.expr, str | None]] = field(default_factory=list)  # expr, getattr name


def _module_name(path: str) -> tuple[str, str]:
    """(module, package) for a repository path such as ``lucid/linalg/_ex.py``."""
    parts = path.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        dotted = ".".join(parts[:-1])
        return dotted, dotted
    return ".".join(parts), ".".join(parts[:-1])


def _imported_module(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _scan(path: str, source: str) -> _Module:
    name, package = _module_name(path)
    mod = _Module(name)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise GateError(f"engine: {path} does not parse ({exc.msg}, line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == _ENGINE and a.asname:
                    mod.aliases[a.asname] = ()
        elif isinstance(node, ast.ImportFrom):
            source_mod = _imported_module(node, package)
            for a in node.names:
                local = a.asname or a.name
                if source_mod == "lucid._C" and a.name == "engine":
                    mod.aliases[local] = ()
                elif source_mod == _ENGINE or source_mod.startswith(_ENGINE + "."):
                    below = source_mod.removeprefix(_ENGINE).split(".")[1:]
                    mod.aliases[local] = (*below, a.name)
                else:
                    mod.imports.append((local, source_mod, a.name))
        elif isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            mod.assigns += [(t.id, node.value) for t in targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.Attribute):
            mod.uses.append((node, None))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            mod.uses.append((node.args[0], node.args[1].value))
    return mod


def _bind_aliases(modules: dict[str, _Module]) -> None:
    """Follow ``X = <engine path>`` and imports of such an ``X`` across modules
    to a fixpoint; a name keeps the first engine path it is bound to."""
    changed = True
    while changed:
        changed = False
        for mod in modules.values():
            for local, source_mod, name in mod.imports:
                found = modules[source_mod].aliases.get(name) if source_mod in modules else None
                if found is not None and local not in mod.aliases:
                    mod.aliases[local] = found
                    changed = True
            for target, value in mod.assigns:
                # () is the engine module itself: a valid, falsy path.
                found = None if target in mod.aliases else _resolve(value, mod.aliases)
                if found is not None:
                    mod.aliases[target] = found
                    changed = True


def references(root: Path, py_files: list[str]) -> set[str]:
    """Every engine path the files name, dotted (``nn.bce_loss``)."""
    modules: dict[str, _Module] = {}
    for path in py_files:
        mod = _scan(path, (root / path).read_text(encoding="utf-8", errors="replace"))
        modules[mod.name] = mod
    _bind_aliases(modules)
    named: set[str] = set()
    for mod in modules.values():
        named |= {".".join(p) for p in mod.aliases.values()}
        for expr, attr in mod.uses:
            found = _resolve(expr, mod.aliases)
            if found is not None:
                named.add(".".join(found if attr is None else (*found, attr)))
    return named


def unbound_bindings(root: Path, py_files: list[str]) -> list[str]:
    """The stub's functions nothing in *py_files* reaches, minus the test-only allowlist."""
    stub = root / config.ENGINE_STUB
    if not stub.is_file():
        return []
    unreached = stub_functions(stub.read_text(encoding="utf-8")) - references(root, py_files)
    return sorted(unreached - set(config.ENGINE_TEST_ONLY_BINDINGS))


class EngineBindingsCollector(Collector):
    name = "engine"
    # Cross-file; ~2 s for the whole tree, so CI and land run it, pre-commit does not.

    @override
    def version(self) -> str:
        return "1"

    @override
    def collect(self, scope: Scope) -> Counts:
        n = len(unbound_bindings(scope.root, scope.py))
        return {("engine", "unbound", config.ENGINE_STUB): n} if n else {}


if __name__ == "__main__":
    here = Path.cwd()
    for binding in unbound_bindings(here, scope_of(here, tracked_files(here), here).py):
        print(binding)
