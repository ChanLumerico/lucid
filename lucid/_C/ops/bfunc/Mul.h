// lucid/_C/ops/bfunc/Mul.h
//
// Declares MulBackward, the autograd node for element-wise tensor
// multiplication, and the public free function mul_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise multiplication: c = a * b.
//
// Forward:  c[i] = a[i] * b[i]  (broadcasting supported).
// Backward: dA = grad_out * b   (product rule: gradient w.r.t. a is scaled by b)
//           dB = grad_out * a   (product rule: gradient w.r.t. b is scaled by a)
//
// kSavesInputs defaults to true (inherited from BinaryKernel).  Both forward
// input Storage objects are retained so that grad_formula can multiply each
// operand by the other during the backward pass.
class LUCID_API MulBackward : public BinaryOp<MulBackward> {
public:
    // Op registration metadata: name "mul", schema version 1, dtype promotion,
    // deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's mul primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.mul(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Public entry point: compute a * b with full broadcasting and autograd support.
LUCID_API TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
