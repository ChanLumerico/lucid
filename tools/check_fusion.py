#!/usr/bin/env python3
"""Phase 19 CI gate: Op Fusion check.

Verifies that:
  1. autograd/FusionPass.h  — FusionPass class + FusionPattern enum
  2. autograd/FusionPass.cpp — run() + try_fuse_* helpers
  3. backend/cpu/CpuBackend.h — fused_linear_relu_forward + fused_linear_gelu_forward
  4. backend/gpu/GpuBackend.h — mlx::core::fast::scaled_dot_product_attention SDPA
  5. backend/IBackend.h       — fused kernel virtual interface
  6. CMakeLists.txt           — FusionPass.cpp compiled
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
    # FusionPass header
    ("autograd/FusionPass.h", "FusionPattern",
     "FusionPass.h must define FusionPattern enum"),
    ("autograd/FusionPass.h", "LinearRelu",
     "FusionPattern must include LinearRelu"),
    ("autograd/FusionPass.h", "ScaledDotProduct",
     "FusionPattern must include ScaledDotProduct (SDPA)"),
    ("autograd/FusionPass.h", "FusionPass",
     "FusionPass.h must declare FusionPass class"),
    ("autograd/FusionPass.h", "int run(",
     "FusionPass must have run() method"),
    ("autograd/FusionPass.h", "try_fuse_linear_activation",
     "FusionPass must declare try_fuse_linear_activation()"),
    ("autograd/FusionPass.h", "try_fuse_sdpa",
     "FusionPass must declare try_fuse_sdpa()"),

    # FusionPass implementation
    ("autograd/FusionPass.cpp", "FusionPass::run",
     "FusionPass.cpp must implement run()"),
    ("autograd/FusionPass.cpp", "try_fuse_linear_activation",
     "FusionPass.cpp must implement try_fuse_linear_activation()"),
    ("autograd/FusionPass.cpp", "try_fuse_sdpa",
     "FusionPass.cpp must implement try_fuse_sdpa()"),
    ("autograd/FusionPass.cpp", "stats_",
     "FusionPass::run must update stats_"),

    # CPU fused kernels
    ("backend/cpu/CpuBackend.h", "fused_linear_relu_forward",
     "CpuBackend must implement fused_linear_relu_forward()"),
    ("backend/cpu/CpuBackend.h", "fused_linear_gelu_forward",
     "CpuBackend must implement fused_linear_gelu_forward()"),
    ("backend/cpu/CpuBackend.h", "vrelu_f32",
     "fused_linear_relu_forward must use vrelu_f32 (vDSP)"),
    ("backend/cpu/CpuBackend.h", "vtanh_f32",
     "fused_linear_gelu_forward must use vtanh_f32 (vForce)"),

    # GPU SDPA fast path
    ("backend/gpu/GpuBackend.h",
     "mlx::core::fast::scaled_dot_product_attention",
     "GpuBackend sdpa_forward must use mlx::core::fast::scaled_dot_product_attention"),
    ("backend/gpu/GpuBackend.h", "mlx/fast.h",
     "GpuBackend.h must include mlx/fast.h"),

    # IBackend interface
    ("backend/IBackend.h", "fused_linear_relu_forward",
     "IBackend.h must declare fused_linear_relu_forward()"),
    ("backend/IBackend.h", "fused_linear_gelu_forward",
     "IBackend.h must declare fused_linear_gelu_forward()"),

    # CMake
    ("CMakeLists.txt", "FusionPass.cpp",
     "CMakeLists.txt must compile FusionPass.cpp"),
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
        print("check_fusion.py  FAILED")
        print(f"\n{len(failures)} issue(s) found:")
        for f in failures:
            print(f)
        return 1

    print("check_fusion.py  PASSED")
    print(f"  All {len(CHECKS)} fusion checks passed.")
    print("  • FusionPass: LinearRelu + SDPA pattern detection")
    print("  • CPU: fused_linear_relu (SGEMM + vDSP vrelu), fused_linear_gelu (SGEMM + vForce tanh)")
    print("  • GPU: SDPA via mlx::core::fast::scaled_dot_product_attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
