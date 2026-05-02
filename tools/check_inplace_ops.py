#!/usr/bin/env python3
"""Phase 16 CI gate: In-place autograd correctness checks.

Verifies that the version-tracking pipeline is intact end-to-end:

  1. bfunc/Inplace.cpp  and  ufunc/Inplace.cpp  call bump_version()
  2. UnaryKernel.h, BinaryKernel.h, NaryKernel.h, ReduceKernel.h
     all call set_saved_versions(...)
  3. autograd/Engine.cpp calls validate_versions()
  4. autograd/Helpers.cpp implements check_version_match
  5. autograd/AutogradNode.h overrides validate_versions()

Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "lucid" / "_C"


def _read(rel: str) -> str:
    path = C_DIR / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


CHECKS: list[tuple[str, str, str]] = [
    # (file-relative-to-_C, required-substring, description)
    (
        "ops/bfunc/Inplace.cpp",
        "bump_version",
        "bfunc Inplace.cpp must call bump_version()",
    ),
    (
        "ops/ufunc/Inplace.cpp",
        "bump_version",
        "ufunc Inplace.cpp must call bump_version()",
    ),
    (
        "kernel/UnaryKernel.h",
        "set_saved_versions",
        "UnaryKernel must call set_saved_versions()",
    ),
    (
        "kernel/BinaryKernel.h",
        "set_saved_versions",
        "BinaryKernel must call set_saved_versions()",
    ),
    (
        "kernel/NaryKernel.h",
        "set_saved_versions",
        "NaryKernel must call set_saved_versions()",
    ),
    (
        "kernel/ReduceKernel.h",
        "set_saved_versions",
        "ReduceKernel must call set_saved_versions()",
    ),
    (
        "autograd/Engine.cpp",
        "validate_versions",
        "Engine::backward must call validate_versions()",
    ),
    (
        "autograd/Helpers.cpp",
        "check_version_match",
        "Helpers.cpp must implement check_version_match()",
    ),
    (
        "autograd/AutogradNode.h",
        "validate_versions",
        "AutogradNode must override validate_versions()",
    ),
    (
        "autograd/Helpers.h",
        "check_version_match",
        "Helpers.h must declare check_version_match()",
    ),
]


def main() -> int:
    failures: list[str] = []
    for rel_path, needle, description in CHECKS:
        content = _read(rel_path)
        full_path = C_DIR / rel_path
        if not (C_DIR / rel_path).exists():
            failures.append(f"  MISSING FILE  {full_path}")
            continue
        if needle not in content:
            failures.append(f"  NOT FOUND '{needle}' in {rel_path}\n    → {description}")

    if failures:
        print("check_inplace_ops.py  FAILED")
        print("\nIssues found:")
        for f in failures:
            print(f)
        return 1

    print("check_inplace_ops.py  PASSED")
    print(f"  All {len(CHECKS)} in-place autograd correctness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
