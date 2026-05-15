// lucid/_C/ops/bfunc/Maximum.h
//
// Declares MaximumBackward, the autograd node for element-wise maximum, and the
// public free function maximum_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise maximum: c[i] = max(a[i], b[i]).
//
// Forward:  c[i] = a[i] >= b[i] ? a[i] : b[i]  (broadcasting supported).
// Backward: Gradient flows only through the operand that "won" the comparison.
//   mask_a[i] = (a[i] >= b[i]) ? 1 : 0
//   mask_b[i] = (a[i] <  b[i]) ? 1 : 0
//   dA = grad_out ⊙ mask_a
//   dB = grad_out ⊙ mask_b
//
// When a[i] == b[i] the gradient is assigned entirely to a (via >=).
// Both inputs are saved (kSavesInputs = true, inherited default) to reconstruct
// the masks during backward.
class LUCID_API MaximumBackward : public BinaryOp<MaximumBackward> {
public:
    // Op registration metadata: name "maximum", schema version 1, dtype
    // promotion, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's maximum primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.maximum(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Public entry point: compute max(a, b) with full broadcasting and autograd support.
LUCID_API TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
