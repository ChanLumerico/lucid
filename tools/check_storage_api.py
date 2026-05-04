#!/usr/bin/env python3
"""
tools/check_storage_api.py — Verify Storage API surface in Storage.h.

Checks that CpuStorage, GpuStorage, and SharedStorage all appear in
lucid/_C/core/Storage.h (they must be defined there as the canonical types).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STORAGE_H = ROOT / "lucid/_C/core/Storage.h"

REQUIRED_TYPES = ["CpuStorage", "GpuStorage", "SharedStorage", "VersionCounter"]
REQUIRED_METHODS = ["bump_version()", "get_version()", "nbytes"]


def main() -> int:
    if not STORAGE_H.exists():
        print(f"[check_storage_api] Storage.h not found at {STORAGE_H}")
        return 1

    text = STORAGE_H.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []

    for t in REQUIRED_TYPES:
        if t not in text:
            issues.append(f"  MISSING TYPE: {t} not found in Storage.h")

    for m in REQUIRED_METHODS:
        # check for the method name without parentheses (definitions vary)
        method_name = m.split("(")[0]
        if method_name not in text:
            issues.append(f"  MISSING METHOD: {method_name} not found in Storage.h")

    if issues:
        print("[check_storage_api] Storage API violations:")
        for i in issues:
            print(i)
        return 1

    print(f"[check_storage_api] Storage.h has all required types and methods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
