#pragma once

// =====================================================================
// Lucid C++ engine — element-wise divide (a / b).
// =====================================================================
//
// @op           div
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = a[i] / b[i]
// Backward: dx =  grad_out / b_saved
//           dy = -grad_out * a_saved / b_saved^2

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

/// Autograd backward node for Div.
class LUCID_API DivBackward : public BinaryOp<DivBackward> {
public:
    static const OpSchema schema_v1;

    // Phase 4.5: dispatch through IBackend — no device check in call site.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.div(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

/// Div.
LUCID_API TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
