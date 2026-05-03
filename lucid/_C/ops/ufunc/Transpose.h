// lucid/_C/ops/ufunc/Transpose.h
//
// Backward node and public entry points for axis-permutation operations:
// permute, transpose (full reversal), T (.T property, alias for transpose),
// mT (batch matrix-transpose, swaps the last two axes), and swapaxes.
//
// All five public functions delegate to PermuteBackward::forward after building
// the appropriate permutation vector.  The forward pass is a metadata-only
// operation: it shares the underlying Storage and rewrites shape/stride, paying
// only an O(ndim) cost.
//
// Backward: applying the inverse permutation to grad_out restores it to the
// input's axis order.  inverse_perm(p)[i] = j such that p[j] == i.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Backward node for an arbitrary axis permutation: y = permute(x, perm).
//
// perm_ is the normalised (non-negative) permutation used in the forward pass.
// apply() computes the inverse permutation and routes grad_out through the
// backend's permute kernel, restoring the original axis order.
class LUCID_API PermuteBackward : public FuncOp<PermuteBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Normalised forward permutation saved from forward(); used to derive the
    // inverse permutation in apply().
    std::vector<int> perm_;

    // Validate perm, rewrite shape/stride, dispatch the forward permute, and
    // wire the backward node.  perm_user may contain negative indices; they are
    // normalised to [0, ndim) before storage.
    static TensorImplPtr forward(const TensorImplPtr& a, const std::vector<int>& perm);

    // Computes dL/dx = permute(dL/dy, inverse(perm_)).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Permute all axes of `a` according to `perm`.
LUCID_API TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm);

// Reverse all axes: equivalent to permute with perm = [ndim-1, ..., 1, 0].
LUCID_API TensorImplPtr transpose_op(const TensorImplPtr& a);

// Alias for transpose_op — implements the .T tensor property.
LUCID_API TensorImplPtr T_op(const TensorImplPtr& a);

// Batch matrix-transpose: swap the last two axes (requires ndim >= 2).
LUCID_API TensorImplPtr mT_op(const TensorImplPtr& a);

// Swap two named axes, wrapping negative indices.
LUCID_API TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2);

}  // namespace lucid
