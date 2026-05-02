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

KERNEL_PATTERNS = {
    "bfunc": re.compile(r"BinaryKernel<|BinaryOp<"),
    "ufunc": re.compile(r"UnaryKernel<|UnaryOp<"),
}
# Accepted bases for any op file
ACCEPTED = re.compile(
    r"NaryKernel<|VariadicKernel<|ReduceKernel<|FuncOp<"
    r"|BinaryKernel<|BinaryOp<|UnaryKernel<|UnaryOp<"
)

def check_dir(subdir: str, pattern: re.Pattern) -> list[str]:
    violations = []
    for path in sorted((ROOT / subdir).rglob("*.h")):
        text = path.read_text()
        # find all Backward class declarations
        for m in re.finditer(r"class\s+(\w*Backward\w*)\s*[:{]", text):
            cls = m.group(1)
            # Look for the inheritance line following the class declaration
            snippet = text[m.start():m.start()+300]
            if not pattern.search(snippet):
                violations.append(f"{path.relative_to(ROOT)}:{cls}: does not use {subdir} kernel template")
    return violations

def main() -> int:
    errors: list[str] = []

    # Strict checks for bfunc / ufunc
    for subdir, pat in KERNEL_PATTERNS.items():
        errors.extend(check_dir(f"ops/{subdir}", pat))

    # For other dirs: just ensure no raw AutogradNode without alias
    for subdir in ["ops/gfunc", "ops/utils", "ops/linalg", "nn"]:
        for path in sorted((ROOT / subdir).rglob("*.h")):
            text = path.read_text()
            for m in re.finditer(r"class\s+(\w*Backward\w*)\s*[:{]", text):
                cls = m.group(1)
                snippet = text[m.start():m.start()+300]
                if re.search(r"AutogradNode<", snippet) and not ACCEPTED.search(snippet):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{cls}: raw AutogradNode without kernel template"
                    )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\n{len(errors)} kernel template violation(s).", file=sys.stderr)
        return 1

    # Count total classes checked
    total = sum(
        1 for sub in ["ops/bfunc","ops/ufunc","ops/gfunc","ops/utils","ops/linalg","nn"]
        for p in (ROOT / sub).rglob("*.h")
        for _ in re.finditer(r"class\s+\w*Backward\w*\s*[:{]", p.read_text())
    )
    print(f"Kernel template check passed — {total} Backward class(es) verified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
