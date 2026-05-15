// lucid/_C/nn/RMSNorm.h
//
// Autograd-aware Root Mean Square Layer Normalization (RMSNorm).
// Normalizes the input by its RMS over the trailing D dimensions and applies
// an element-wise scale: y = (x / rms(x)) * gamma.  No bias term.
//
// Compared with LayerNorm, RMSNorm omits the mean subtraction step, halving
// the number of required statistics.  The operation is forced to FP32.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for RMS Normalization.
//
// Saves the per-slice reciprocal RMS (saved_rstd_) from the forward pass
// for use in the backward formula.  No mean is saved because RMSNorm does
// not center the input.
class LUCID_API RMSNormBackward : public FuncOp<RMSNormBackward, 2> {
public:
    static const OpSchema schema_v1;
    Storage saved_rstd_;     // Reciprocal RMS per normalized slice.
    std::size_t outer_ = 0;  // Product of all non-normalized leading dims.
    std::size_t N_ = 0;      // Product of all normalized trailing dims.

    // Run the forward pass.
    // x     – input of any rank >= rank(gamma).
    // gamma – scale; shape must match the trailing dims of x.
    // eps   – added to the variance for numerical stability.
    // Returns a tensor with the same shape as x.
    static TensorImplPtr forward(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);

    // Compute gradients for x (dx) and gamma (d_gamma).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: delegates to RMSNormBackward::forward.
LUCID_API TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);

}  // namespace lucid
