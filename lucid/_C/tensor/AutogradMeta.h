// lucid/_C/tensor/AutogradMeta.h
//
// Optional autograd bookkeeping embedded in TensorImpl. A TensorImpl only
// carries AutogradMeta when it has been marked requires_grad = true or
// when it is the output of an op that produces a differentiable result.
// Tensors that never participate in differentiation (e.g., integer index
// tensors, random samples) have no AutogradMeta to avoid the overhead of
// allocating the struct and populating the optional grad storage.
//
// The version counter is incremented by TensorImpl::bump_version() on every
// in-place modification (optimizer step, in-place ops). Any autograd node
// that captured this tensor during forward() records the version at capture
// time; validate_versions() in the backward pass checks for mismatches to
// detect illegal in-place mutations.

#pragma once

#include <cstdint>
#include <optional>

#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd bookkeeping for a single tensor.
//
// requires_grad controls whether operations involving this tensor build
// an autograd graph. is_leaf is true for parameters created by the user
// (model weights) and false for intermediate tensors produced by ops.
// Leaf tensors accumulate gradients into grad; non-leaf tensors do not
// accumulate unless retain_grad has been called on them (not yet exposed
// in this struct; a future extension could add a retain_grad flag here).
//
// grad_fn points to the backward Node that produced this tensor. It is
// null for leaf tensors (which use AccumulateGrad installed lazily by
// ensure_grad_fn) and for tensors detached from the graph.
struct AutogradMeta {
    // Whether this tensor participates in gradient computation.
    bool requires_grad = false;
    // True for user-created parameters; false for op outputs.
    bool is_leaf = true;
    // Monotonically increasing in-place mutation counter.
    std::int64_t version = 0;
    // Backward function that produced this tensor (null for leaves).
    NodePtr grad_fn;
    // Accumulated gradient storage; set on leaves after backward().
    std::optional<Storage> grad;
};

}  // namespace lucid
