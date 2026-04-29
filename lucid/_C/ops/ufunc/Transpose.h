#pragma once

// =====================================================================
// Lucid C++ engine — axis-permutation shape ops.
// =====================================================================
//
//   permute(t, dims)        general axis permutation
//   transpose(t)            reverse all axes (alias for `_T`)
//   _T(t)                   same as transpose
//   _mT(t)                  swap last two axes (matrix transpose)
//   swapaxes(t, a1, a2)     swap two named axes
//
// All five materialize via a single `permute_copy_<dtype>` kernel; backward
// applies the inverse permutation. The forward signatures differ but they
// all reduce to `(input, perm: vector<int>)` for the underlying kernel.
//
// Backward: `dx = permute(grad_out, inverse_perm)`.

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Permute.
class LUCID_API PermuteBackward : public FuncOp<PermuteBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<int> perm_;  // forward permutation

    static TensorImplPtr forward(const TensorImplPtr& a, const std::vector<int>& perm);

    std::vector<Storage> apply(Storage grad_out) override;
};

/// Permute.
LUCID_API TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm);
/// Transpose.
LUCID_API TensorImplPtr transpose_op(const TensorImplPtr& a);
/// T.
LUCID_API TensorImplPtr T_op(const TensorImplPtr& a);
/// Mt.
LUCID_API TensorImplPtr mT_op(const TensorImplPtr& a);
/// Swapaxes.
LUCID_API TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2);

}  // namespace lucid
