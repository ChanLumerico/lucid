#pragma once

// =====================================================================
// Lucid C++ engine — extended normalization ops.
// =====================================================================
//
// 1. batch_norm_eval(x, mean, var, gamma, beta, eps)
//      Inference-mode BatchNorm: uses precomputed running_mean / running_var.
//      x : (B, C, *S)
//      mean, var, gamma, beta : (C,)
//      y = γ · (x − μ)/√(σ²+ε) + β
//      All four (C,) tensors are 1-D and broadcast over (B, *S).
//
// 2. lp_normalize(x, ord, axis, eps)
//      y = x / max(||x||_p, eps)
//      ||x||_p along `axis` with keepdims=True.
//
// 3. global_response_norm(x, gamma, beta, eps)
//      ConvNeXt v2's GRN.
//      Gx = ||x||_2 over (H, W) (spatial axes -2, -1) keepdims  → (B, 1, 1, C)
//      Nx = Gx / (mean(Gx, axis=-1, keepdims) + eps)
//      y  = γ · (x · Nx) + β · x
//      For 4-D NHWC layout. γ, β shape (1, 1, 1, C). The legacy lucid
//      keeps NCHW everywhere; we accept (B, C, H, W) and reduce over
//      (-1, -2) on H,W per channel, then mean over C.
//
// All ops support F32 / F64; CPU only initially. Backward derivatives
// computed analytically and saved-tensor based.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// ----- BatchNorm inference (no statistics computed) -----
class LUCID_API BatchNormEvalBackward : public FuncOp<BatchNormEvalBackward, 5> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-5;
    Storage rstd_;  // 1/sqrt(var + eps), shape (C,)

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                           const TensorImplPtr& mean,
                                           const TensorImplPtr& var,
                                           const TensorImplPtr& gamma,
                                           const TensorImplPtr& beta,
                                           double eps);

// ----- Lp Normalize -----
class LUCID_API LpNormalizeBackward : public FuncOp<LpNormalizeBackward, 1> {
public:
    static const OpSchema schema_v1;
    double ord_ = 2.0;
    int axis_ = 1;
    double eps_ = 1e-12;
    Storage saved_norm_;  // ||x||_p along axis (broadcastable shape)

    static TensorImplPtr forward(const TensorImplPtr& x, double ord, int axis, double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps);

// ----- Global Response Norm -----
class LUCID_API GlobalResponseNormBackward : public FuncOp<GlobalResponseNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-6;
    // Saved tensors keep the shape (B, C, H, W) input + intermediate Nx and Gx
    // for backward composition.
    Storage saved_Nx_;  // shape broadcastable to x

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                                const TensorImplPtr& gamma,
                                                const TensorImplPtr& beta,
                                                double eps);

}  // namespace lucid
