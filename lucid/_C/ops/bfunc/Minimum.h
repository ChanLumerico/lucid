// lucid/_C/ops/bfunc/Minimum.h
//
// Declares MinimumBackward, the autograd node for element-wise minimum, and the
// public free function minimum_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise minimum: c[i] = min(a[i], b[i]).
//
// Forward:  c[i] = a[i] <= b[i] ? a[i] : b[i]  (broadcasting supported).
// Backward: Gradient flows only through the operand that "won" the comparison.
//   mask_a[i] = (b[i] >= a[i]) ? 1 : 0   (a won or tied)
//   mask_b[i] = (b[i] <  a[i]) ? 1 : 0   (b strictly won)
//   dA = grad_out ⊙ mask_a
//   dB = grad_out ⊙ mask_b
//
// The mask sense is the mirror image of Maximum: the masks are computed with
// the arguments reversed (b >= a rather than a >= b).  When a[i] == b[i] the
// gradient is assigned entirely to a.  Both inputs are saved to reconstruct
// the masks during backward.
class LUCID_API MinimumBackward : public BinaryOp<MinimumBackward> {
public:
    // Op registration metadata: name "minimum", schema version 1, dtype
    // promotion, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's minimum primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.minimum(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Public entry point: compute min(a, b) with full broadcasting and autograd support.
LUCID_API TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
