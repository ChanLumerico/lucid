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

class LUCID_API DivBackward : public BinaryOp<DivBackward> {
public:
    static const OpSchema schema_v1;

    static CpuStorage cpu_kernel(const CpuStorage& a, const CpuStorage& b,
                                 const Shape& out_shape, Dtype dt);

    static GpuStorage gpu_kernel(const GpuStorage& a, const GpuStorage& b,
                                 const Shape& out_shape, Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

LUCID_API TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
