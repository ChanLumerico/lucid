#!/usr/bin/env python3
"""Report direct TensorImpl field access before Phase 2 encapsulation."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


FIELD_RE = re.compile(
    r"(?:->|\.)(storage_|shape_|stride_|dtype_|device_|requires_grad_|"
    r"is_leaf_|version_|grad_fn_|grad_storage_)"
)


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
    total = 0

    for source in iter_sources(root):
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = source.read_text(encoding="latin-1")
        for match in FIELD_RE.finditer(text):
            counts[match.group(1)] += 1
            total += 1

    print(f"direct TensorImpl field accesses: {total}")
    for field, count in counts.most_common():
        print(f"{field},{count}")

    if args.fail_on_any and total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
