#pragma once

// =====================================================================
// Lucid C++ engine — ReduceOp: backward-compat alias for ReduceKernel.
// =====================================================================
//
// Phase 3: ReduceOp<D> is now an alias for kernel::ReduceKernel<D>.
// All existing code that inherits from ReduceOp continues to work unchanged.
// New code should prefer `kernel::ReduceKernel<D>` directly.
//
// Layer: ops/ufunc/. Depends on kernel/ReduceKernel.h.

#include "../../kernel/ReduceKernel.h"

namespace lucid {

/// Backward-compat alias. Prefer kernel::ReduceKernel for new ops.
template <class Derived>
using ReduceOp = ReduceKernel<Derived>;

}  // namespace lucid
