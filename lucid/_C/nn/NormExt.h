// lucid/_C/nn/NormExt.h
//
// Additional normalization variants not covered by the core norm files:
//
//   BatchNormEval   – Batch Normalization in inference mode using externally
//                     supplied running mean/variance tensors.
//   LpNormalize     – Lp-norm normalization along a chosen axis.
//   GlobalResponseNorm – ConvNeXt-style GRN that normalizes each channel by
//                     its global L2 norm across the spatial dimensions.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for inference-mode Batch Normalization.
//
// Uses caller-supplied running mean and variance (not computed from the
// current batch).  This is the path taken after training when the model is
// in eval mode.  rstd_ (1 / sqrt(var + eps)) is saved for the backward pass.
// Takes five input tensors: x, mean, var, gamma, beta.
class LUCID_API BatchNormEvalBackward : public FuncOp<BatchNormEvalBackward, 5> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-5;
    Storage rstd_;  // Pre-computed reciprocal std, shape (C,).

    // x     – input (B, C, ...).
    // mean  – running mean (C,).
    // var   – running variance (C,).
    // gamma – scale (C,).
    // beta  – shift (C,).
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Returns gradients for [x, mean, var, gamma, beta].
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point for inference-mode batch normalization.
LUCID_API TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                           const TensorImplPtr& mean,
                                           const TensorImplPtr& var,
                                           const TensorImplPtr& gamma,
                                           const TensorImplPtr& beta,
                                           double eps);

// Autograd node for Lp normalization along a single axis.
//
// Computes y = x / max(||x||_p, eps) along `axis`.
// saved_norm_ holds the per-slice Lp norm needed by the backward rule.
class LUCID_API LpNormalizeBackward : public FuncOp<LpNormalizeBackward, 1> {
public:
    static const OpSchema schema_v1;
    double ord_ = 2.0;     // Norm order (e.g. 2.0 for L2).
    int axis_ = 1;         // Axis along which to normalize (negative allowed).
    double eps_ = 1e-12;   // Minimum denominator to avoid division by zero.
    Storage saved_norm_;   // Per-slice norm values needed for backward.

    // axis is resolved to a non-negative index inside forward.
    static TensorImplPtr forward(const TensorImplPtr& x, double ord, int axis, double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point for Lp normalization.
LUCID_API TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps);

// Autograd node for Global Response Normalization (GRN), as used in ConvNeXt V2.
//
// For 4-D (N, C, H, W) input:
//   Nx[c] = ||x[:, c, :, :]||_2 / mean_c(||x[:, c, :, :]||_2)
//   y = x * Nx  (broadcast) * gamma + beta
// saved_Nx_ stores the normalized global response for backward.
class LUCID_API GlobalResponseNormBackward : public FuncOp<GlobalResponseNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-6;

    Storage saved_Nx_;  // Normalized global response, shape (N, C, 1, 1).

    // x     – 4-D input (N, C, H, W).
    // gamma – per-channel scale (C,).
    // beta  – per-channel shift (C,).
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Returns gradients for [x, gamma, beta].
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point for Global Response Normalization.
LUCID_API TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                                const TensorImplPtr& gamma,
                                                const TensorImplPtr& beta,
                                                double eps);

}  // namespace lucid
