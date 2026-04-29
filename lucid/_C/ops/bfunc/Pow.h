#pragma once

// =====================================================================
// Lucid C++ engine — element-wise power (a ** b).
// =====================================================================
//
// @op           pow
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   ForceFP32  (precision matters; pow blows up under fp16)
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = a[i] ^ b[i]
// Backward: dx = b * a^(b-1) * grad_out
//           dy = log(a) * a^b * grad_out  =  log(a) * c * grad_out

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

/// Autograd backward node for Pow.
class LUCID_API PowBackward : public BinaryOp<PowBackward> {
public:
    static const OpSchema schema_v1;

    // Phase 4.5: dispatch through IBackend — no device check in call site.
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Storage& b, const Shape& shape, Dtype dt) {
        return be.pow(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

/// Pow.
LUCID_API TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
