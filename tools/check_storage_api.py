#!/usr/bin/env python3
"""Phase 2.5 Storage ownership model compliance check.

Verifies:
  1. core/Storage.h exports all required Phase 2.5 types:
       DataBuffer, StoragePtr, VersionCounter, bump_version, get_version
  2. Both CpuStorage and GpuStorage have a `version` field.
  3. No source file outside Storage.h / CpuBackend.h calls
     mutable_storage() without the call appearing in a context that
     also calls bump_version() in the same function body (heuristic:
     flag files with mutable_storage but no bump_version anywhere).
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent / "lucid/_C"
STORAGE_H = ROOT / "core/Storage.h"

REQUIRED_TOKENS = [
    "DataBuffer",
    "StoragePtr",
    "VersionCounter",
    "bump_version",
    "get_version",
]

def main() -> int:
    errors: list[str] = []

    # 1. Check Storage.h exports
    storage_text = STORAGE_H.read_text()
    for tok in REQUIRED_TOKENS:
        if tok not in storage_text:
            errors.append(f"core/Storage.h: missing required symbol '{tok}'")

    # 2. Check version field on both storage structs
    cpu_match = re.search(r"struct CpuStorage.*?};", storage_text, re.DOTALL)
    if cpu_match:
        if "version" not in cpu_match.group(0):
            errors.append("core/Storage.h: CpuStorage missing 'version' field")
    else:
        errors.append("core/Storage.h: CpuStorage struct not found")

    gpu_match = re.search(r"struct GpuStorage.*?};", storage_text, re.DOTALL)
    if gpu_match:
        if "version" not in gpu_match.group(0):
            errors.append("core/Storage.h: GpuStorage missing 'version' field")
    else:
        errors.append("core/Storage.h: GpuStorage struct not found")

    # 3. Scan for mutable_storage() callers that lack bump_version
    skip = {STORAGE_H, ROOT / "backend/cpu/CpuBackend.h", ROOT / "core/TensorImpl.h",
            ROOT / "core/TensorImpl.cpp"}
    for path in sorted(ROOT.rglob("*.cpp")) + sorted(ROOT.rglob("*.h")):
        if path in skip:
            continue
        text = path.read_text()
        if "mutable_storage()" in text and "bump_version" not in text:
            errors.append(
                f"{path.relative_to(ROOT)}: calls mutable_storage() but never calls "
                f"bump_version() — possible untracked in-place mutation"
            )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\n{len(errors)} Storage API violation(s).", file=sys.stderr)
        return 1

    print("Storage API check passed — DataBuffer/StoragePtr/VersionCounter present, "
          "no untracked mutable_storage() callers.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
