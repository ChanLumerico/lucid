// lucid/_C/ops/bfunc/Div.h
//
// Declares DivBackward, the autograd node for element-wise tensor division,
// and the public free function div_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise division: c = a / b.
//
// Forward:  c[i] = a[i] / b[i]  (broadcasting supported).
// Backward: dA =  grad_out / b
//           dB = -grad_out * a / b²
//
// Both forward inputs are retained (kSavesInputs = true, inherited default)
// because the backward formula for dB requires both a and b.
class LUCID_API DivBackward : public BinaryOp<DivBackward> {
public:
    // Op registration metadata: name "div", schema version 1, dtype promotion,
    // deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's div primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.div(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode: da = grad_out / b,  db = -grad_out * a / b².
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// Public entry point: compute a / b with full broadcasting and autograd support.
LUCID_API TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
