// lucid/_C/ops/bfunc/Add.h
//
// Declares AddBackward, the autograd node for element-wise tensor addition, and
// the public free function add_op that serves as the engine entry point.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise addition: c = a + b.
//
// Inherits the forward pass, broadcasting logic, and apply() trampoline from
// BinaryOp<AddBackward> (alias for BinaryKernel<AddBackward>).  Only the
// backend dispatch hook and the gradient formula need to be defined here.
//
// Forward:  c[i] = a[i] + b[i]  (NumPy-style broadcasting supported).
// Backward: dA = reduce_to_shape(grad_out, out_shape → a.shape)
//           dB = reduce_to_shape(grad_out, out_shape → b.shape)
//
// kSavesInputs = false because the addition gradient does not require the
// forward inputs — the upstream gradient is passed through to both operands
// unchanged (up to broadcast reduction).
class LUCID_API AddBackward : public BinaryOp<AddBackward> {
public:
    // Disables saving of the forward input Storage objects.  The add backward
    // formula only needs grad_out, so retaining the inputs would waste memory.
    static constexpr bool kSavesInputs = false;

    // Op registration metadata: name "add", schema version 1, dtype promotion
    // policy (both inputs are cast to the wider type), deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's add primitive.
    // Called by BinaryKernel::forward after broadcasting both inputs to
    // out_shape.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.add(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode gradient: da = grad_out, db = grad_out (identity; BinaryKernel
    // handles broadcast reduction back to input shapes).
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(
        const TensorImplPtr& grad_out, const TensorImplPtr& /*a*/, const TensorImplPtr& /*b*/) {
        return {grad_out, grad_out};
    }
};

// Public entry point: compute a + b with full broadcasting and autograd support.
LUCID_API TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
