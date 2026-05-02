#!/usr/bin/env python3
"""Phase 18 CI gate: Metal Shader Escape Hatch check.

Verifies that the Metal kernel runner infrastructure is present:
  1. MetalKernelRunner.h  — compile_metal_kernel + run_metal_kernel declared
  2. MetalKernelRunner.mm — MTL pipeline + dispatch implemented
  3. IBackend.h            — run_custom_metal_kernel interface
  4. GpuBackend.h          — run_custom_metal_kernel override
  5. CMakeLists.txt        — MetalKernelRunner.mm compiled + Metal framework
  6. bind.cpp              — _run_metal_kernel Python binding
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "lucid" / "_C"


def _read(rel: str) -> str:
    p = C_DIR / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


CHECKS: list[tuple[str, str, str]] = [
    # Header
    ("backend/gpu/MetalKernelRunner.h", "compile_metal_kernel",
     "MetalKernelRunner.h must declare compile_metal_kernel()"),
    ("backend/gpu/MetalKernelRunner.h", "run_metal_kernel",
     "MetalKernelRunner.h must declare run_metal_kernel()"),
    ("backend/gpu/MetalKernelRunner.h", "KernelLaunchConfig",
     "MetalKernelRunner.h must define KernelLaunchConfig"),
    ("backend/gpu/MetalKernelRunner.h", "MetalKernel",
     "MetalKernelRunner.h must define MetalKernel struct"),

    # Implementation
    ("backend/gpu/MetalKernelRunner.mm", "MTLCreateSystemDefaultDevice",
     "MetalKernelRunner.mm must create MTL device"),
    ("backend/gpu/MetalKernelRunner.mm", "newLibraryWithSource",
     "MetalKernelRunner.mm must compile MSL source"),
    ("backend/gpu/MetalKernelRunner.mm", "newComputePipelineStateWithFunction",
     "MetalKernelRunner.mm must create pipeline state"),
    ("backend/gpu/MetalKernelRunner.mm", "dispatchThreadgroups",
     "MetalKernelRunner.mm must dispatch compute"),
    ("backend/gpu/MetalKernelRunner.mm", "waitUntilCompleted",
     "MetalKernelRunner.mm must synchronize GPU"),

    # IBackend interface
    ("backend/IBackend.h", "run_custom_metal_kernel",
     "IBackend.h must declare run_custom_metal_kernel()"),

    # GpuBackend override
    ("backend/gpu/GpuBackend.h", "run_custom_metal_kernel",
     "GpuBackend.h must override run_custom_metal_kernel()"),
    ("backend/gpu/GpuBackend.h", "compile_metal_kernel",
     "GpuBackend.h must call compile_metal_kernel()"),
    ("backend/gpu/GpuBackend.h", "run_metal_kernel",
     "GpuBackend.h must call run_metal_kernel()"),

    # CMake
    ("CMakeLists.txt", "MetalKernelRunner.mm",
     "CMakeLists.txt must compile MetalKernelRunner.mm"),

    # Python binding
    ("bindings/bind.cpp", "_run_metal_kernel",
     "bind.cpp must expose _run_metal_kernel Python binding"),
]


def main() -> int:
    cache: dict[str, str] = {}
    failures: list[str] = []

    for rel, needle, description in CHECKS:
        if rel not in cache:
            cache[rel] = _read(rel)
        if not (C_DIR / rel).exists():
            failures.append(f"  MISSING FILE  {C_DIR / rel}\n    → {description}")
            continue
        if needle not in cache[rel]:
            failures.append(f"  NOT FOUND '{needle}' in {rel}\n    → {description}")

    if failures:
        print("check_metal_escape.py  FAILED")
        print(f"\n{len(failures)} issue(s) found:")
        for f in failures:
            print(f)
        return 1

    print("check_metal_escape.py  PASSED")
    print(f"  All {len(CHECKS)} Metal Escape Hatch checks passed.")
    print("  • MetalKernelRunner.h/mm: compile + dispatch infrastructure")
    print("  • IBackend/GpuBackend: run_custom_metal_kernel interface + override")
    print("  • Python binding: engine._run_metal_kernel()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
