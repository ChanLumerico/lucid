#pragma once

// =====================================================================
// Lucid C++ engine — UnaryOp: backward-compat alias for UnaryKernel.
// =====================================================================
//
// Phase 3: UnaryOp<D> is now an alias for kernel::UnaryKernel<D>.
// All existing code that inherits from UnaryOp continues to work unchanged.
// New code should prefer `kernel::UnaryKernel<D>` directly.
//
// Layer: ops/ufunc/. Depends on kernel/UnaryKernel.h.

#include "../../kernel/UnaryKernel.h"

namespace lucid {

/// Backward-compat alias. Prefer kernel::UnaryKernel for new ops.
template <class Derived>
using UnaryOp = UnaryKernel<Derived>;

}  // namespace lucid
