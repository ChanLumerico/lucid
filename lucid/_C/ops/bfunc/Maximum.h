#pragma once

// =====================================================================
// Lucid C++ engine — element-wise maximum (max(a, b)).
// =====================================================================
//
// @op           maximum
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = max(a[i], b[i])
// Backward: dx = grad_out * (a >= b),  dy = grad_out * (a < b)
//   Tied case (a == b) flows entirely to a — matches PyTorch's `maximum`
//   convention so gradient sums equal `grad_out` element-wise.

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

/// Autograd backward node for Maximum.
class LUCID_API MaximumBackward : public BinaryOp<MaximumBackward> {
public:
    static const OpSchema schema_v1;

    // Phase 4.5: dispatch through IBackend — no device check in call site.
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Storage& b, const Shape& shape, Dtype dt) {
        return be.maximum(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

/// Maximum.
LUCID_API TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
