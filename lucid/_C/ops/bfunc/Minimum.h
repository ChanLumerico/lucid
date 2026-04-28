#pragma once

// =====================================================================
// Lucid C++ engine — element-wise minimum (min(a, b)).
// =====================================================================
//
// @op           minimum
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = min(a[i], b[i])
// Backward: dx = grad_out * (a <= b),  dy = grad_out * (a > b)
//   Mirrors `Maximum` with masks swapped. Ties to a (a <= b includes equality).

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

class LUCID_API MinimumBackward : public BinaryOp<MinimumBackward> {
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

LUCID_API TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
