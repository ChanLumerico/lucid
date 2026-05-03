// lucid/_C/ops/bfunc/Pow.h
//
// Declares PowBackward, the autograd node for element-wise exponentiation, and
// the public free function pow_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise power: c = a ^ b.
//
// Forward:  c[i] = a[i] ^ b[i]  (broadcasting supported).
// Backward: dA = b * a^(b-1) * grad_out   (power rule)
//           dB = log(a) * a^b * grad_out   (exponential rule)
//
// AmpPolicy::ForceFP32 is used because log() and fractional exponentiation are
// numerically unsafe in reduced-precision formats.  Both inputs are saved
// (kSavesInputs = true, inherited default) since both are needed in the
// backward formulas.
class LUCID_API PowBackward : public BinaryOp<PowBackward> {
public:
    // Op registration metadata: name "pow", schema version 1, always computed
    // in FP32 regardless of input dtype, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's pow primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.pow(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Public entry point: compute a ^ b with full broadcasting and autograd support.
LUCID_API TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
