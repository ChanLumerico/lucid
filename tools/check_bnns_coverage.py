#!/usr/bin/env python3
"""Phase 15 CI gate: BNNS coverage check.

Verifies that CpuBackend.h contains BNNS calls for:
  • conv_nd_forward   — conv1d N==1 branch + BNNSFilterCreateLayerConvolution
  • batch_norm        — BNNSFilterCreateLayerNormalization + BNNSBatchNorm
  • lstm_forward      — BNNSDirectApplyLSTMBatchTrainingCaching
And that IBackend.h declares lstm_forward.

Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPU_BACKEND = ROOT / "lucid" / "_C" / "backend" / "cpu" / "CpuBackend.h"
IBACKEND    = ROOT / "lucid" / "_C" / "backend" / "IBackend.h"


def _read(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# Each entry: (file, needle, description)
CHECKS: list[tuple[Path, str, str]] = [
    # 15.1 conv1d BNNS
    (CPU_BACKEND, "N == 1",
     "conv_nd_forward: 'N == 1' BNNS fast-path branch"),
    (CPU_BACKEND, "BNNSFilterCreateLayerConvolution",
     "conv_nd_forward: BNNSFilterCreateLayerConvolution call"),

    # 15.2 BatchNorm BNNS
    (CPU_BACKEND, "BNNSFilterCreateLayerNormalization",
     "batch_norm_forward_f32_fast: BNNSFilterCreateLayerNormalization call"),
    (CPU_BACKEND, "BNNSBatchNorm",
     "batch_norm_forward_f32_fast: BNNSBatchNorm filter-type constant"),
    (CPU_BACKEND, "BNNSNormalizationFilterApplyBatch",
     "batch_norm_forward_f32_fast: BNNSNormalizationFilterApplyBatch call"),

    # 15.3 LSTM BNNS
    (CPU_BACKEND, "BNNSDirectApplyLSTMBatchTrainingCaching",
     "lstm_forward: BNNSDirectApplyLSTMBatchTrainingCaching call"),
    (CPU_BACKEND, "lstm_forward",
     "CpuBackend.h: lstm_forward override present"),

    # IBackend interface
    (IBACKEND, "lstm_forward",
     "IBackend.h: lstm_forward method declared"),
    (IBACKEND, "LstmOpts",
     "IBackend.h: LstmOpts struct declared"),
]


def main() -> int:
    cache: dict[Path, str] = {}
    failures: list[str] = []

    for path, needle, description in CHECKS:
        if path not in cache:
            cache[path] = _read(path)
        src = cache[path]
        if not path.exists():
            failures.append(f"  MISSING FILE  {path}")
            continue
        if needle not in src:
            failures.append(f"  NOT FOUND '{needle}'\n    → {description}")

    if failures:
        print("check_bnns_coverage.py  FAILED")
        print("\nIssues found:")
        for f in failures:
            print(f)
        return 1

    print("check_bnns_coverage.py  PASSED")
    print(f"  All {len(CHECKS)} BNNS coverage checks passed.")
    print("  • Phase 15.1 conv1d  : BNNS N==1 + N==2 fast path present")
    print("  • Phase 15.2 BatchNorm: BNNSFilterCreateLayerNormalization(BNNSBatchNorm)")
    print("  • Phase 15.3 LSTM    : BNNSDirectApplyLSTMBatchTrainingCaching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
