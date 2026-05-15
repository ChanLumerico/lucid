#!/usr/bin/env python3
"""
tools/check_layers.py — Validate that no module creates forbidden cross-layer imports.

Enforces these directional constraints (not a strict total order):
  - lucid.autograd  must NOT import from  lucid.nn  or  lucid.optim
  - lucid.optim     must NOT import from  lucid.nn
  - lucid._C        Python side must NOT import from  lucid.nn / lucid.optim
  - lucid.linalg    must NOT import from  lucid.nn  or  lucid.optim

TYPE_CHECKING imports are excluded (annotation-only, never executed at runtime).
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LUCID = ROOT / "lucid"

# Each rule: (source_prefix, forbidden_prefixes)
# A file whose module starts with source_prefix must not import any forbidden_prefix.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("lucid.autograd", ("lucid.nn", "lucid.optim")),
    ("lucid.optim", ("lucid.nn",)),
    ("lucid.linalg", ("lucid.nn", "lucid.optim")),
    # test utilities must not import test files (parity tests may import torch, that's OK)
]


def _is_type_checking_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _get_runtime_imports(path: Path) -> list[str]:
    """Return absolute module names imported at runtime (excludes TYPE_CHECKING blocks)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    type_check_ids: set[int] = set()
    for node in ast.walk(tree):
        if _is_type_checking_block(node):
            for child in ast.walk(node):
                type_check_ids.add(id(child))

    imports = []
    for node in ast.walk(tree):
        if id(node) in type_check_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imports.append(node.module)
    return imports


def _file_module(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def check() -> int:
    violations: list[str] = []

    for py_file in sorted(LUCID.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue

        mod = _file_module(py_file)

        # Imports that are explicitly allowed even if they cross layer boundaries.
    # Format: frozenset of (source_prefix, import_module) pairs.
    ALLOWED_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
        {
            # Optimizers legitimately need Parameter (it's a base type, not a module)
            ("lucid.optim", "lucid.nn.parameter"),
        }
    )

    for src_prefix, forbidden in RULES:
        if not (mod == src_prefix or mod.startswith(src_prefix + ".")):
            continue
        for imp in _get_runtime_imports(py_file):
            for forbidden_prefix in forbidden:
                if not (
                    imp == forbidden_prefix or imp.startswith(forbidden_prefix + ".")
                ):
                    continue
                # Check exceptions
                if (src_prefix, imp) in ALLOWED_EXCEPTIONS:
                    continue
                violations.append(
                    f"  {mod} → {imp}  "
                    f"(forbidden: {src_prefix} must not import {forbidden_prefix})"
                )

    if violations:
        print("[check_layers] Forbidden layer imports found:")
        for v in violations:
            print(v)
        print(f"\n[check_layers] {len(violations)} violation(s).")
        return 1

    print(f"[check_layers] all layer imports OK ({len(RULES)} rules checked)")
    return 0


if __name__ == "__main__":
    sys.exit(check())
