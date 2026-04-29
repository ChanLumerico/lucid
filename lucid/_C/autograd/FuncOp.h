#pragma once

// =====================================================================
// Lucid C++ engine — FuncOp: backward-compat alias for AutogradNode.
// =====================================================================
//
// Phase 3: FuncOp<D, N> is an alias for AutogradNode<D, N>.
// All existing code that inherits from FuncOp continues to work unchanged.
// New code should prefer `AutogradNode<D, N>` directly.
//
// Layer: autograd/.

#include "AutogradNode.h"

namespace lucid {

/// Backward-compat alias. Prefer AutogradNode for new ops.
template <class Derived, std::size_t N_IN>
using FuncOp = AutogradNode<Derived, N_IN>;

}  // namespace lucid
