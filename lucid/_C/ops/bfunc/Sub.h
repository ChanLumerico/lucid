// lucid/_C/ops/bfunc/Sub.h
//
// Declares SubBackward, the autograd node for element-wise tensor subtraction,
// and the public free function sub_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise subtraction: c = a - b.
//
// Forward:  c[i] = a[i] - b[i]  (broadcasting supported).
// Backward: dA = +grad_out  (gradient passes through to a unchanged)
//           dB = -grad_out  (gradient is negated for b because c decreases as b
//                            increases)
//
// kSavesInputs = false: the subtraction gradient depends only on the sign of
// the output gradient, not on the forward input values.
class LUCID_API SubBackward : public BinaryOp<SubBackward> {
public:
    // Disables retaining the forward input Storage objects for the backward pass.
    static constexpr bool kSavesInputs = false;

    // Op registration metadata: name "sub", schema version 1, dtype promotion,
    // deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's sub primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.sub(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Public entry point: compute a - b with full broadcasting and autograd support.
LUCID_API TensorImplPtr sub_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
