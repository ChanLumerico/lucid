#!/usr/bin/env python3
"""Dump public C++ op signatures as CSV.

Phase 1.6 keeps this as an audit aid rather than a hard gate. The gate lives
in `tools/check_op_api.py`; this script gives a stable inventory when reviewing
signature churn across op families.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path


FUNC_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:LUCID_API\s+)?"
    r"(?P<return_type>(?:std::(?:vector|pair)\s*<[^>]+>\s*|[\w:]+(?:\s*<[^>]+>)?)\s*[&*]?\s*)"
    r"(?P<func_name>\w+_op)\s*\("
    r"(?P<params>[^)]*)"
    r"\)\s*;",
)


def split_params(params: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in params:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def iter_headers(root: Path) -> list[Path]:
    scan_dirs = [root / "ops", root / "nn"]
    out: list[Path] = []
    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for dirpath, _, filenames in os.walk(scan_dir):
            for filename in filenames:
                if filename.endswith((".h", ".hpp")):
                    out.append(Path(dirpath) / filename)
    return sorted(out)


def rows(root: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for header in iter_headers(root):
        content = header.read_text()
        for match in FUNC_PATTERN.finditer(content):
            params = split_params(match.group("params").strip())
            result.append(
                {
                    "file": str(header),
                    "line": str(content.count("\n", 0, match.start()) + 1),
                    "op": match.group("func_name").removesuffix("_op"),
                    "function": match.group("func_name"),
                    "return_type": " ".join(match.group("return_type").split()),
                    "param_count": str(0 if params == [""] else len(params)),
                    "params": " | ".join(params),
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="lucid/_C", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    fieldnames = ["file", "line", "op", "function", "return_type", "param_count", "params"]
    records = rows(args.root)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
