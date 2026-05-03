// lucid/_C/ops/utils/Sort.h
//
// Declares sorting, index-finding, and uniqueness operations.  sort_op is
// differentiable for floating-point inputs; the remaining ops (argsort,
// argmax, argmin, nonzero, unique, topk) return integer indices or
// non-differentiable results.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Sort `a` along `axis` in ascending order and return the sorted values.
// Backward: scatter the incoming gradient back to original positions using
// the sort indices returned internally by the backend (IndexScatterBackward).
// Only differentiable for F32 and F64 dtypes.
LUCID_API TensorImplPtr sort_op(const TensorImplPtr& a, int axis);

// Return the indices that would sort `a` along `axis`.  Output dtype is I32.
// Not differentiable (indices are discontinuous).
LUCID_API TensorImplPtr argsort_op(const TensorImplPtr& a, int axis);

// Return the index of the maximum value along `axis`.  Output dtype is I64.
// If `keepdims` is true the axis dimension is retained as size 1 in the
// output shape.  Not differentiable.
LUCID_API TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims);

// Return the index of the minimum value along `axis`.  Same semantics and
// output dtype as argmax_op.  Not differentiable.
LUCID_API TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims);

// Return the flat indices of all non-zero elements in `a` as a 2-D tensor
// of shape (count, ndim).  Always materialises data on the CPU because the
// output size is data-dependent.  Not differentiable.
LUCID_API TensorImplPtr nonzero_op(const TensorImplPtr& a);

// Return the unique sorted elements of `a` as a 1-D tensor.  Implemented on
// the CPU for all input devices; always outputs to Device::CPU.  Supported
// dtypes: F32, F64, I32, I64.  Not differentiable.
LUCID_API TensorImplPtr unique_op(const TensorImplPtr& a);

// Return the top-k values and their indices along `axis`.  The output
// shape has dimension `axis` set to k; all other dimensions are unchanged.
// Returns {values, indices} where values is differentiable (F32/F64) via
// IndexScatterBackward and indices has dtype I32.
LUCID_API std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis);

}  // namespace lucid
