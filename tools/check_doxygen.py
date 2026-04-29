#!/usr/bin/env python3
"""
Phase 10.1 — Doxygen coverage checker.

Scans all public headers under lucid/_C/ for `LUCID_API` declarations and
verifies each has a preceding `///` Doxygen comment.  Prints a coverage
report and exits non-zero if coverage falls below the threshold.

Usage:
    python tools/check_doxygen.py              # report only
    python tools/check_doxygen.py --strict     # fail if any undocumented
    python tools/check_doxygen.py --threshold 80  # fail if < 80% covered
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP_ROOT = ROOT / "lucid" / "_C"

# Headers we intentionally skip (internal implementation details, generated).
SKIP_DIRS = {"bindings"}
SKIP_FILES = {"FuncOp.h", "_BinaryOp.h", "_ReduceOp.h", "_UnaryOp.h"}

# Regex: any line that declares a public API item.
_API_RE = re.compile(
    r"^\s*(?:class|struct|LUCID_API\s+\w[\w*&\s<>,:]*?\s+\w+\s*[\(;]|"
    r"LUCID_API\b)"
)
_LUCID_API_RE = re.compile(r"\bLUCID_API\b")
_DOC_RE = re.compile(r"^\s*///")


def scan_file(path: Path) -> tuple[int, int]:
    """Return (documented, total) LUCID_API declarations in file."""
    lines = path.read_text(errors="replace").splitlines()
    documented = total = 0
    for i, line in enumerate(lines):
        if not _LUCID_API_RE.search(line):
            continue
        # Skip macro definitions themselves and forward decls inside macros.
        if line.strip().startswith("#define") or line.strip().startswith("//"):
            continue
        total += 1
        # Look for a /// comment on the immediately preceding non-blank line.
        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        if j >= 0 and _DOC_RE.match(lines[j]):
            documented += 1
    return documented, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any LUCID_API item is undocumented.")
    parser.add_argument("--threshold", type=int, default=70,
                        help="Minimum %% coverage (default 70).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    total_doc = total_all = 0
    undocumented: list[tuple[Path, int, str]] = []

    for hdr in sorted(CPP_ROOT.rglob("*.h")):
        # Skip internal/generated dirs.
        parts = hdr.relative_to(CPP_ROOT).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        if hdr.name in SKIP_FILES:
            continue

        lines = hdr.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if not _LUCID_API_RE.search(line):
                continue
            if line.strip().startswith("#define") or line.strip().startswith("//"):
                continue
            total_all += 1
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j >= 0 and _DOC_RE.match(lines[j]):
                total_doc += 1
            else:
                undocumented.append((hdr.relative_to(ROOT), i + 1, line.strip()[:80]))

    coverage = 100.0 * total_doc / max(total_all, 1)
    print(f"Doxygen coverage: {total_doc}/{total_all} LUCID_API items documented "
          f"({coverage:.0f}%)")

    if undocumented and (args.verbose or args.strict):
        print(f"\nUndocumented ({len(undocumented)}):")
        for path, lineno, text in undocumented[:40]:
            print(f"  {path}:{lineno}  {text}")
        if len(undocumented) > 40:
            print(f"  … and {len(undocumented) - 40} more")

    if args.strict and undocumented:
        print("\n✗ --strict: undocumented API items found.")
        return 1
    if coverage < args.threshold:
        print(f"\n✗ Coverage {coverage:.0f}% < threshold {args.threshold}%.")
        return 1
    print(f"\n✓ Coverage {coverage:.0f}% meets threshold {args.threshold}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
