#!/usr/bin/env python3
"""Phase 10.1 — Op template conformance checker.

Verifies that every op header under lucid/_C/ops/ that declares a *Backward
class follows the canonical _TEMPLATE.h structure:

  1. *Backward class declares `static const OpSchema schema_v1`
  2. *Backward class declares `grad_formula` or `apply`
  3. Corresponding .cpp registers the op with LUCID_REGISTER_OP

Run from project root:
    python tools/check_op_template.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS_DIR = ROOT / "lucid" / "_C" / "ops"

# Headers we intentionally skip.
SKIP_FILES = {"_TEMPLATE.h", "_BinaryOp.h", "_UnaryOp.h", "_ReduceOp.h"}

_BACKWARD_CLASS_RE = re.compile(r"class\s+LUCID_API\s+(\w+Backward)\b")
_SCHEMA_V1_RE = re.compile(r"static\s+const\s+OpSchema\s+schema_v1\b")
_GRAD_RE = re.compile(r"\b(grad_formula|apply)\s*\(")
_REGISTER_RE = re.compile(r"\bLUCID_REGISTER_OP\s*\(\s*(\w+)\s*\)")


def check_header(hdr: Path) -> list[str]:
    errors: list[str] = []
    text = hdr.read_text(errors="replace")

    # Find all *Backward class declarations in this header.
    for m in _BACKWARD_CLASS_RE.finditer(text):
        cls = m.group(1)
        # Extract the class body (from '{' after the match to the next '};' at top level).
        body_start = text.find("{", m.end())
        if body_start == -1:
            continue
        depth = 0
        body_end = body_start
        for i in range(body_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        body = text[body_start:body_end]

        rel = hdr.relative_to(ROOT)

        if not _SCHEMA_V1_RE.search(body):
            errors.append(f"{rel}: {cls} missing 'static const OpSchema schema_v1'")

        if not _GRAD_RE.search(body):
            errors.append(
                f"{rel}: {cls} missing 'grad_formula' or 'apply' method declaration"
            )

    return errors


def check_cpp(cpp: Path, header_backward_classes: set[str]) -> list[str]:
    errors: list[str] = []
    if not cpp.exists():
        return errors
    text = cpp.read_text(errors="replace")
    registered = {m.group(1) for m in _REGISTER_RE.finditer(text)}
    for cls in header_backward_classes:
        if cls not in registered:
            rel = cpp.relative_to(ROOT)
            errors.append(
                f"{rel}: {cls} not registered with LUCID_REGISTER_OP "
                f"(or check if it uses a different registration mechanism)"
            )
    return errors


def main() -> int:
    all_errors: list[str] = []
    warned_cpps: list[str] = []

    for hdr in sorted(OPS_DIR.rglob("*.h")):
        if hdr.name in SKIP_FILES:
            continue

        hdr_errors = check_header(hdr)
        all_errors.extend(hdr_errors)

        # For every Backward class found in this header, check the matching .cpp.
        text = hdr.read_text(errors="replace")
        classes = {m.group(1) for m in _BACKWARD_CLASS_RE.finditer(text)}
        if classes:
            cpp = hdr.with_suffix(".cpp")
            # Some ops have .cpp in a different location; skip missing ones with a warning.
            if not cpp.exists():
                warned_cpps.append(str(hdr.relative_to(ROOT)))
            else:
                all_errors.extend(check_cpp(cpp, classes))

    if warned_cpps:
        print(f"Note: {len(warned_cpps)} header(s) have no matching .cpp (non-fatal):")
        for p in warned_cpps[:10]:
            print(f"  {p}")
        if len(warned_cpps) > 10:
            print(f"  … and {len(warned_cpps) - 10} more")

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        print(f"\n{len(all_errors)} template conformance error(s) found.", file=sys.stderr)
        return 1

    n = sum(
        1
        for hdr in OPS_DIR.rglob("*.h")
        if hdr.name not in SKIP_FILES
        and _BACKWARD_CLASS_RE.search(hdr.read_text(errors="replace"))
    )
    print(f"Op template check passed — {n} Backward class(es) verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
