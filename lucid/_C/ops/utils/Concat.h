// lucid/_C/ops/utils/Concat.h
//
// Public API for concatenation, stacking, splitting, chunking, and unbinding
// operations.  All functions allocate fresh output tensors and wire autograd
// nodes where required.  The corresponding backward passes are defined in
// Concat.cpp and are not exposed here because they are purely internal.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Concatenate a list of tensors along `axis`.  All tensors must share the
// same dtype, device, and rank; every dimension except `axis` must agree.
// The output dimension at `axis` is the sum of the corresponding input
// dimensions.  Backward: slice the incoming gradient back into per-input
// pieces using the recorded input sizes along `axis`.
LUCID_API TensorImplPtr concatenate_op(const std::vector<TensorImplPtr>& xs, int axis);

// Stack a list of same-shape tensors by inserting a new dimension at `axis`.
// The output rank is (input rank + 1) and the new dimension has size
// equal to the number of inputs.  Backward: slice out each size-1 slice
// along `axis` and squeeze it back to the original input shape.
LUCID_API TensorImplPtr stack_op(const std::vector<TensorImplPtr>& xs, int axis);

// Horizontal stack: concatenate along axis 1 for >=2-D inputs, or along
// axis 0 for 1-D inputs (matches NumPy / reference semantics).
LUCID_API TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs);

// Vertical stack: concatenate along axis 0 for >=2-D inputs, or call
// stack_op(xs, 0) for 1-D inputs to produce a 2-D output.
LUCID_API TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs);

// Split `a` into `num_splits` equal pieces along `axis`.  The dimension size
// at `axis` must be divisible by `num_splits`.  Each piece carries a
// SplitSliceBackward node that scatters its gradient back into the full
// input gradient via insert_axis_slice.
LUCID_API std::vector<TensorImplPtr>
split_op(const TensorImplPtr& a, std::int64_t num_splits, int axis);

// Split `a` at the given `indices` along `axis`, producing
// (indices.size() + 1) pieces with potentially unequal sizes.
LUCID_API std::vector<TensorImplPtr>
split_at_op(const TensorImplPtr& a, std::vector<std::int64_t> indices, int axis);

// Alias for split_op with `num_splits = chunks`.
LUCID_API std::vector<TensorImplPtr>
chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis);

// Split `a` along `axis` into individual slices and remove the split
// dimension, returning `shape[axis]` tensors each with rank (ndim - 1).
// Backward: inserts the gradient slice back at offset k along `axis` and
// then unsqueezes the axis dimension before scattering.
LUCID_API std::vector<TensorImplPtr> unbind_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
