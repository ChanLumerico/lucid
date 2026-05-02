#!/usr/bin/env python3
"""Phase 9 CI gate: Unified Memory infrastructure check.

Verifies that:
  1. SharedStorage is defined in core/Storage.h (variant index 2)
  2. MetalAllocator.h exists with allocate_shared declaration
  3. MetalAllocator.mm exists with Metal allocation implementation
  4. CMakeLists.txt lists MetalAllocator.mm and links -framework Metal
  5. MlxBridge.h/cpp has shared_storage_to_gpu (zero-copy path)
  6. IBackend.h has to_shared_storage method

Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
C_DIR  = ROOT / "lucid" / "_C"


def _read(rel: str) -> str:
    p = C_DIR / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _exists(rel: str) -> bool:
    return (C_DIR / rel).exists()


CHECKS: list[tuple[str, str, str]] = [
    # SharedStorage in Storage.h
    ("core/Storage.h",       "SharedStorage",
     "Storage.h must define SharedStorage struct"),
    ("core/Storage.h",       "storage_is_metal_shared",
     "Storage.h must provide storage_is_metal_shared() helper"),
    ("core/Storage.h",       "cpu_view()",
     "SharedStorage must have cpu_view() method"),
    ("core/Storage.h",       "variant<CpuStorage, GpuStorage, SharedStorage>",
     "Storage must be a 3-way variant"),

    # MetalAllocator header
    ("backend/gpu/MetalAllocator.h", "allocate_shared",
     "MetalAllocator.h must declare allocate_shared()"),
    ("backend/gpu/MetalAllocator.h", "deallocate_shared",
     "MetalAllocator.h must declare deallocate_shared()"),
    ("backend/gpu/MetalAllocator.h", "wrap_existing",
     "MetalAllocator.h must declare wrap_existing()"),
    ("backend/gpu/MetalAllocator.h", "make_metal_shared",
     "MetalAllocator.h must declare make_metal_shared()"),

    # MetalAllocator implementation
    ("backend/gpu/MetalAllocator.mm", "MTLCreateSystemDefaultDevice",
     "MetalAllocator.mm must call MTLCreateSystemDefaultDevice"),
    ("backend/gpu/MetalAllocator.mm", "MTLResourceStorageModeShared",
     "MetalAllocator.mm must allocate with MTLResourceStorageModeShared"),
    ("backend/gpu/MetalAllocator.mm", "CFRetain",
     "MetalAllocator.mm must retain the MTLBuffer"),
    ("backend/gpu/MetalAllocator.mm", "CFRelease",
     "MetalAllocator.mm must release in deallocate_shared"),

    # CMakeLists.txt
    ("CMakeLists.txt", "MetalAllocator.mm",
     "CMakeLists.txt must compile MetalAllocator.mm"),
    ("CMakeLists.txt", "-framework Metal",
     "CMakeLists.txt must link -framework Metal"),
    ("CMakeLists.txt", "OBJCXX",
     "CMakeLists.txt must enable OBJCXX language for .mm compilation"),

    # MlxBridge zero-copy path
    ("backend/gpu/MlxBridge.h",   "shared_storage_to_gpu",
     "MlxBridge.h must declare shared_storage_to_gpu()"),
    ("backend/gpu/MlxBridge.cpp", "shared_storage_to_gpu",
     "MlxBridge.cpp must implement shared_storage_to_gpu()"),

    # IBackend
    ("backend/IBackend.h", "to_shared_storage",
     "IBackend.h must have to_shared_storage() method"),
]


def main() -> int:
    cache: dict[str, str] = {}
    failures: list[str] = []

    for rel, needle, description in CHECKS:
        if rel not in cache:
            cache[rel] = _read(rel)
        full_path = C_DIR / rel
        if not full_path.exists():
            failures.append(f"  MISSING FILE  {full_path}\n    → {description}")
            continue
        if needle not in cache[rel]:
            failures.append(f"  NOT FOUND '{needle}' in {rel}\n    → {description}")

    if failures:
        print("check_unified_mem.py  FAILED")
        print(f"\n{len(failures)} issue(s) found:")
        for f in failures:
            print(f)
        return 1

    print("check_unified_mem.py  PASSED")
    print(f"  All {len(CHECKS)} unified-memory checks passed.")
    print("  • Phase 9.1: MetalAllocator.h/mm present")
    print("  • Phase 9.2: SharedStorage in Storage.h (3-way variant)")
    print("  • Phase 9.3: shared_storage_to_gpu zero-copy path in MlxBridge")
    print("  • CMakeLists.txt: OBJCXX + MetalAllocator.mm + -framework Metal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
