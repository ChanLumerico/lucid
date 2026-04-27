#pragma once

// =====================================================================
// Lucid C++ engine — element-wise add (a + b).
// =====================================================================
//
// First op rebuilt on the new BinaryOp<Derived> CRTP. The kernel calls
// Apple Accelerate's vDSP_vadd (F32) / vDSP_vaddD (F64); integer paths use
// scalar loops (vDSP doesn't ship integer add).
//
// @op           add
// @schema_v     1
// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64, I32, I64}
// @outputs      (c: Tensor<T,*>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(numel(out))
//
// Forward:  c[i] = a[i] + b[i]
// Backward: dx = grad_out, dy = grad_out  (broadcast-back handled by base)
//
// Layer: autograd/ops/binary/.

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

class LUCID_API AddBackward : public BinaryOp<AddBackward> {
public:
    // Add doesn't need input *values* for backward — d(a+b)/da = 1, irrespective
    // of a or b. Save metadata only (BinaryOp always saves shapes/dtype/device).
    static constexpr bool kSavesInputs = false;

    static const OpSchema schema_v1;

    static CpuStorage cpu_kernel(const CpuStorage& a, const CpuStorage& b,
                                 const Shape& out_shape, Dtype dt);

    static GpuStorage gpu_kernel(const GpuStorage& a, const GpuStorage& b,
                                 const Shape& out_shape, Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

/// Public free function — pybind11 binds this.
LUCID_API TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
