#!/usr/bin/env python3
"""Report direct external TensorImpl field access before Phase 2 encapsulation.

The scan is intentionally lightweight, but it avoids counting backward-node
state such as ``this->dtype_`` and ``bwd->device_``. Those fields are not
TensorImpl internals and are valid op-node state.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


FIELD_RE = re.compile(
    r"(?P<owner>[A-Za-z_][A-Za-z0-9_]*)\s*(?:->|\.)(?P<field>storage_|shape_|stride_|"
    r"dtype_|device_|requires_grad_|is_leaf_|version_|grad_fn_|grad_storage_)\b"
)

COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)

NON_TENSOR_OWNERS = {
    "this",  # backward-node state inside Node subclasses
    "bwd",   # newly allocated backward-node state
}


def iter_sources(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix in {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="lucid/_C", help="C++ source root")
    parser.add_argument(
        "--fail-on-any",
        action="store_true",
        help="Return nonzero when any direct field access remains.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    counts: Counter[str] = Counter()
    sites: list[tuple[Path, int, str]] = []
    total = 0

    for source in iter_sources(root):
        if source.name == "TensorImpl.cpp":
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = source.read_text(encoding="latin-1")
        text = COMMENT_RE.sub("", text)
        for match in FIELD_RE.finditer(text):
            if match.group("owner") in NON_TENSOR_OWNERS:
                continue
            field = match.group("field")
            counts[field] += 1
            total += 1
            line_no = text.count("\n", 0, match.start()) + 1
            sites.append((source, line_no, match.group(0)))

    print(f"direct TensorImpl field accesses: {total}")
    for field, count in counts.most_common():
        print(f"{field},{count}")
    for source, line_no, snippet in sites:
        print(f"{source}:{line_no}: {snippet.strip()}")

    if args.fail_on_any and total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
