// lucid/_C/autograd/FuncOp.h
//
// Provides FuncOp<Derived, N_IN>, a convenience alias for
// AutogradNode<Derived, N_IN>.  All concrete backward node types in the
// ops/ layer inherit from FuncOp rather than AutogradNode directly, so that
// the name better reflects their role as "functional operation" backward
// nodes and a future divergence between FuncOp and AutogradNode is possible
// without touching every op file.

#pragma once

#include "AutogradNode.h"

namespace lucid {

// Type alias: FuncOp<Derived, N_IN> == AutogradNode<Derived, N_IN>.
//
// Derived must satisfy the AutogradNode CRTP contract:
//   - Expose a static constexpr schema_v1 with at least a .name field.
//   - Implement apply(Storage grad_out) -> vector<Storage>.
// N_IN is the number of forward input tensors whose saved Storage values are
// available in saved_inputs_ during backward.
template <class Derived, std::size_t N_IN>
using FuncOp = AutogradNode<Derived, N_IN>;

}  // namespace lucid
