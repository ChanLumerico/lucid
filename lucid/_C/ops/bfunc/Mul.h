#pragma once

// =====================================================================
// Lucid C++ engine — element-wise multiply (a * b).
// =====================================================================
//
// @op           mul
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = a[i] * b[i]
// Backward: dx = grad_out * b_saved,  dy = grad_out * a_saved   (saves both inputs)

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

/// Autograd backward node for Mul.
class LUCID_API MulBackward : public BinaryOp<MulBackward> {
public:
    static const OpSchema schema_v1;

    // Phase 4.5: dispatch through IBackend — no device check in call site.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.mul(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

/// Mul.
LUCID_API TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
