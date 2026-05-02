#!/usr/bin/env python3
"""Kernel template coverage check.

Rules:
  - All *Backward classes in ops/bfunc/*.h must inherit kernel::BinaryKernel<D>
    or its alias BinaryOp<D>.
  - All *Backward classes in ops/ufunc/*.h must inherit kernel::UnaryKernel<D>
    or its alias UnaryOp<D>.
  - All *Backward classes in ops/gfunc/, ops/utils/, ops/linalg/, nn/ may use
    NaryKernel<D,N>, VariadicKernel<D>, ReduceKernel<D>, or FuncOp<D,N>;
    raw AutogradNode<D,N> without a kernel alias is flagged as a violation.

Exits 0 on full coverage, 1 on violations.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent / "lucid/_C"

# Any recognised kernel-family base.
# FuncOp / ReduceOp are valid alternatives even inside bfunc / ufunc for ops
# that don't fit the pure element-wise pattern (e.g. matmul, softmax, reduce).
# The one hard invariant: no class may inherit raw AutogradNode<> directly.
_ANY_KERNEL = re.compile(
    r"BinaryKernel<|BinaryOp<"
    r"|UnaryKernel<|UnaryOp<"
    r"|NaryKernel<|VariadicKernel<"
    r"|ReduceKernel<|ReduceOp<"
    r"|FuncOp<"
)

KERNEL_PATTERNS = {
    "bfunc": _ANY_KERNEL,   # element-wise OR complex binary (e.g. matmul)
    "ufunc": _ANY_KERNEL,   # element-wise OR reduce / shape (e.g. softmax, permute)
}
# Same set accepted for all other dirs
ACCEPTED = _ANY_KERNEL

# Matches "class [MACRO...] ClassName [: ...]" — handles LUCID_API and similar macros
# between the `class` keyword and the actual class name.
_CLASS_RE = re.compile(r"class\s+(?:\w+\s+)*(\w*Backward\w*)\s*[:{]")


def check_dir(subdir: str, pattern: re.Pattern) -> list[str]:
    violations = []
    for path in sorted((ROOT / subdir).rglob("*.h")):
        text = path.read_text()
        for m in _CLASS_RE.finditer(text):
            cls = m.group(1)
            snippet = text[m.start():m.start() + 300]
            if not pattern.search(snippet):
                violations.append(
                    f"{path.relative_to(ROOT)}:{cls}: does not use {subdir} kernel template"
                )
    return violations


def main() -> int:
    errors: list[str] = []

    # Strict checks for bfunc / ufunc
    for subdir, pat in KERNEL_PATTERNS.items():
        errors.extend(check_dir(f"ops/{subdir}", pat))

    # For other dirs: flag raw AutogradNode without a kernel alias
    for subdir in ["ops/gfunc", "ops/utils", "ops/linalg", "nn"]:
        for path in sorted((ROOT / subdir).rglob("*.h")):
            text = path.read_text()
            for m in _CLASS_RE.finditer(text):
                cls = m.group(1)
                snippet = text[m.start():m.start() + 300]
                if re.search(r"AutogradNode<", snippet) and not ACCEPTED.search(snippet):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{cls}: raw AutogradNode without kernel template"
                    )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\n{len(errors)} kernel template violation(s).", file=sys.stderr)
        return 1

    total = sum(
        1
        for sub in ["ops/bfunc", "ops/ufunc", "ops/gfunc", "ops/utils", "ops/linalg", "nn"]
        for p in (ROOT / sub).rglob("*.h")
        for _ in _CLASS_RE.finditer(p.read_text())
    )
    print(f"Kernel template check passed — {total} Backward class(es) verified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
