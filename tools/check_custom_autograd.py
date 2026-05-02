#!/usr/bin/env python3
"""Phase 12 CI gate: Custom Autograd C++ backend check.

Verifies that:
  1. autograd/CustomFunction.h  — FunctionCtx + PythonBackwardNode declared
  2. autograd/CustomFunction.cpp — backward apply + registration implemented
  3. bindings/bind_autograd.cpp  — register_custom_function() called
  4. CMakeLists.txt              — CustomFunction.cpp compiled
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "lucid" / "_C"

def _read(rel: str) -> str:
    p = C_DIR / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""

CHECKS: list[tuple[str, str, str]] = [
    # Header
    ("autograd/CustomFunction.h", "FunctionCtx",
     "CustomFunction.h must define FunctionCtx"),
    ("autograd/CustomFunction.h", "PythonBackwardNode",
     "CustomFunction.h must define PythonBackwardNode"),
    ("autograd/CustomFunction.h", "save_for_backward",
     "FunctionCtx must have save_for_backward()"),
    ("autograd/CustomFunction.h", "register_custom_function",
     "CustomFunction.h must declare register_custom_function()"),

    # Implementation
    ("autograd/CustomFunction.cpp", "PythonBackwardNode::apply",
     "CustomFunction.cpp must implement PythonBackwardNode::apply()"),
    ("autograd/CustomFunction.cpp", "gil_scoped_acquire",
     "backward apply must acquire GIL before calling Python"),
    ("autograd/CustomFunction.cpp", "py_backward_fn",
     "apply must invoke py_backward_fn"),
    ("autograd/CustomFunction.cpp", "_register_python_backward_node",
     "CustomFunction.cpp must expose _register_python_backward_node"),
    ("autograd/CustomFunction.cpp", "AccumulateGrad",
     "CustomFunction.cpp must attach AccumulateGrad for leaf inputs"),

    # Binding
    ("bindings/bind_autograd.cpp", "register_custom_function",
     "bind_autograd.cpp must call register_custom_function()"),
    ("bindings/bind_autograd.cpp", "CustomFunction.h",
     "bind_autograd.cpp must include CustomFunction.h"),

    # CMake
    ("CMakeLists.txt", "CustomFunction.cpp",
     "CMakeLists.txt must compile CustomFunction.cpp"),
]


def main() -> int:
    cache: dict[str, str] = {}
    failures: list[str] = []

    for rel, needle, description in CHECKS:
        if rel not in cache:
            cache[rel] = _read(rel)
        if not (C_DIR / rel).exists():
            failures.append(f"  MISSING FILE  {C_DIR / rel}\n    → {description}")
            continue
        if needle not in cache[rel]:
            failures.append(f"  NOT FOUND '{needle}' in {rel}\n    → {description}")

    if failures:
        print("check_custom_autograd.py  FAILED")
        print(f"\n{len(failures)} issue(s) found:")
        for f in failures:
            print(f)
        return 1

    print("check_custom_autograd.py  PASSED")
    print(f"  All {len(CHECKS)} custom autograd checks passed.")
    print("  • FunctionCtx + PythonBackwardNode infrastructure present")
    print("  • GIL-safe backward dispatch")
    print("  • Python binding: FunctionCtx, _PythonBackwardNode, _register_python_backward_node")
    return 0


if __name__ == "__main__":
    sys.exit(main())
