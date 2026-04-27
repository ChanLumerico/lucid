#pragma once

// =====================================================================
// Lucid C++ engine — 2-D matrix multiply (a @ b).
// =====================================================================
//
// @op           matmul
// @schema_v     1
// @inputs       (a: Tensor<T, [M, K]>, b: Tensor<T, [K, N]>)  T in {F32, F64}
// @outputs      (c: Tensor<T, [M, N]>)
// @amp_policy   Promote
// @determinism  deterministic
// @complexity   O(M * N * K)
//
// Forward:  C = A @ B   via cblas_sgemm/dgemm (uses Apple AMX coprocessor on M-series).
// Backward: dA = dC @ B^T,  dB = A^T @ dC.
//
// Phase 3.1: 2-D only. Batched matmul + ND broadcasting arrives in Phase 3.5
// (`backend/cpu/Im2Col.h` family).
//
// Layer: autograd/ops/linalg/. Inherits FuncOp directly because BinaryOp's
// equal-shape contract doesn't apply.

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "../../autograd/FuncOp.h"

namespace lucid {

class LUCID_API MatmulBackward : public FuncOp<MatmulBackward, 2> {
public:
    static const OpSchema schema_v1;

    static TensorImplPtr forward(const TensorImplPtr& a, const TensorImplPtr& b);

    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
