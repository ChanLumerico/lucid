#!/usr/bin/env python3
"""Phase 1 foundation checks.

This linter keeps the Phase 1 consolidation rules from regressing:
  - typed errors live in core/Error.{h,cpp}, not Exceptions.{h,cpp}
  - string-only throws route through ErrorBuilder outside ErrorBuilder itself
  - low-level helper definitions have a single source of truth
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOTS = [Path("lucid/_C")]
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp"}


def iter_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                files.append(path)
    return sorted(files)


def main() -> int:
    errors: list[str] = []
    sources = iter_sources()

    for path in sources:
        text = path.read_text()
        if "Exceptions.h" in text or "Exceptions.cpp" in text:
            errors.append(f"{path}: references old Exceptions filename")

        if path != Path("lucid/_C/core/ErrorBuilder.cpp"):
            for lineno, line in enumerate(text.splitlines(), start=1):
                if re.search(r"throw\s+(LucidError|NotImplementedError|IndexError)\s*\(", line):
                    errors.append(f"{path}:{lineno}: string-only throw must use ErrorBuilder")

    helper_defs = {
        "allocate_cpu": [],
        "mlx_shape_to_lucid": [],
    }
    for path in sources:
        text = path.read_text()
        if re.search(r"\bCpuStorage\s+allocate_cpu\s*\(", text):
            helper_defs["allocate_cpu"].append(path)
        if re.search(r"\bShape\s+mlx_shape_to_lucid\s*\(", text):
            helper_defs["mlx_shape_to_lucid"].append(path)

    expected = {
        "allocate_cpu": [Path("lucid/_C/core/Helpers.h")],
        "mlx_shape_to_lucid": [Path("lucid/_C/backend/gpu/MlxBridge.h")],
    }
    for name, paths in helper_defs.items():
        if paths != expected[name]:
            errors.append(
                f"{name} helper definitions should be {expected[name]}, found {paths}"
            )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"\n{len(errors)} Phase 1 foundation error(s) found.", file=sys.stderr)
        return 1

    print("Phase 1 foundation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
