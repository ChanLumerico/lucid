#pragma once

// =====================================================================
// Lucid C++ engine — BatchNorm{1,2,3}d (pure-function, train-mode stats).
// =====================================================================
//
//   batch_norm{N}d(x, γ, β, eps)
//     x : (B, C, *S)        |S| = N
//     γ, β : (C,)
//     y[b,c, *s] = γ_c · (x[b,c, *s] − μ_c)/√(σ²_c + ε) + β_c
//   where μ_c, σ²_c are computed across (B, *S) per channel — i.e. across
//   the batch axis and every spatial axis.
//
// Pure function: no running_mean / running_var update. The Module wrapper
// (Phase 5) updates running stats separately via `_copy_from`.
//
// AMP policy: ForceFP32 (precision-sensitive across batch).

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

template <int N>
/// Autograd backward node for BatchNormNd.
class LUCID_API BatchNormNdBackward : public FuncOp<BatchNormNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;  // (1, C, 1, ..., 1) for broadcast
    Storage saved_rstd_;  // same shape
    int B_ = 0, C_ = 0;
    int S_[N > 0 ? N : 1];  // spatial extent (size N; placeholder when N=0)

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

using BatchNorm1dBackward = BatchNormNdBackward<1>;
using BatchNorm2dBackward = BatchNormNdBackward<2>;
using BatchNorm3dBackward = BatchNormNdBackward<3>;

/// Batch norm1d.
LUCID_API TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);
/// Batch norm.
LUCID_API TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps = 1e-5);  // 2D, kept name for API compat
/// Batch norm3d.
LUCID_API TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);

}  // namespace lucid
