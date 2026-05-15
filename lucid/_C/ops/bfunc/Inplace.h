// lucid/_C/ops/bfunc/Inplace.h
//
// Declares the in-place arithmetic operators: +=, -=, *=, /=, **=, and the
// in-place maximum/minimum helpers.  Each function mutates tensor a in-place
// by overwriting its Storage with the result of the corresponding out-of-place
// operation, then bumps a's version counter.
//
// Invariants:
//   - The result of the corresponding out-of-place op must have the same shape
//     as a; otherwise ShapeMismatch is thrown.
//   - a must not share storage with a view (storage_is_shared() must be false);
//     in-place mutation through a shared-storage alias is rejected with an
//     error that instructs the caller to clone() first.
//   - Neither a nor b may carry requires_grad=true.  If gradient tracking is
//     active, BinaryKernel::forward will set requires_grad on the intermediate
//     out-of-place result, but the subsequent version-counter bump on a will
//     correctly invalidate any saved reference in the backward graph.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// a += b
LUCID_API TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a -= b
LUCID_API TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a *= b
LUCID_API TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a /= b
LUCID_API TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a **= b
LUCID_API TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a = max(a, b)  in-place
LUCID_API TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// a = min(a, b)  in-place
LUCID_API TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
