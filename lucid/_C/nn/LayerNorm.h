// lucid/_C/nn/LayerNorm.h
//
// Autograd-aware Layer Normalization: normalizes over the last D dimensions
// specified by the shape of gamma (and beta), then applies an affine transform
// y = gamma * x_hat + beta.  D may be any value from 1 to rank(x).
//
// The operation is forced to FP32 (AmpPolicy::ForceFP32) to avoid numerical
// instability in half-precision accumulations.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Layer Normalization.
//
// Saves the per-slice mean (saved_mean_) and reciprocal standard deviation
// (saved_rstd_) computed during the forward pass; these are required by the
// backward formula.  outer_ * N_ == numel(x).
// Thread safety: single-use; create during forward, consume during backward.
class LUCID_API LayerNormBackward : public FuncOp<LayerNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;     // Shape (outer,), one mean per normalized slice.
    Storage saved_rstd_;     // Shape (outer,), reciprocal std per slice.
    std::size_t outer_ = 0;  // Product of all leading (non-normalized) dims.
    std::size_t N_ = 0;      // Product of all normalized (trailing) dims.

    // Run the forward pass.
    // x      – input of any shape.
    // gamma  – scale, shape must match the trailing dims of x.
    // beta   – shift, same shape as gamma.
    // eps    – small constant added to variance for numerical stability.
    // Returns a tensor with the same shape as x.
    // Throws ShapeMismatch if gamma has more dims than x, or if the trailing
    // dims of x do not match gamma, or if gamma and beta shapes differ.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Compute gradients for x (dx), gamma (d_gamma), and beta (d_beta).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: delegates to LayerNormBackward::forward.
LUCID_API TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps);

}  // namespace lucid
