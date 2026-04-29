#pragma once

// =====================================================================
// Lucid C++ engine — BinaryOp: backward-compat alias for BinaryKernel.
// =====================================================================
//
// Phase 3: BinaryOp<D> is now an alias for kernel::BinaryKernel<D>.
// All existing code that inherits from BinaryOp continues to work unchanged.
// New code should prefer `kernel::BinaryKernel<D>` directly.
//
// Also re-exports `detail::*` helpers that op .cpp files use directly:
//   - detail::ensure_grad_fn
//   - detail::broadcast_shapes / try_broadcast_shapes
//   - detail::broadcast_cpu
//
// Layer: ops/bfunc/. Depends on kernel/BinaryKernel.h.

#include "../../kernel/BinaryKernel.h"

namespace lucid {

/// Backward-compat alias. Prefer kernel::BinaryKernel for new ops.
template <class Derived>
using BinaryOp = BinaryKernel<Derived>;

}  // namespace lucid
