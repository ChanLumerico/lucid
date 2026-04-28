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

class LUCID_API MaximumBackward : public BinaryOp<MaximumBackward> {
public:
    static const OpSchema schema_v1;

    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const CpuStorage& b,
                                 const Shape& out_shape,
                                 Dtype dt);

    static GpuStorage gpu_kernel(const GpuStorage& a,
                                 const GpuStorage& b,
                                 const Shape& out_shape,
                                 Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

LUCID_API TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
