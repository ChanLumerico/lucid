// lucid/_C/nn/GroupNorm.h
//
// Autograd-aware Group Normalization for spatial inputs of rank >= 2.
// The C channels are split into G groups of C/G channels each.  Within each
// (batch, group, spatial) slice the activations are normalized, then the
// affine transform y = gamma * x_hat + beta is applied per-channel.
//
// GroupNorm is equivalent to LayerNorm when G==1 and to InstanceNorm when
// G==C.  The operation is forced to FP32.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Group Normalization.
//
// Saves per-group mean (saved_mean_) and reciprocal std (saved_rstd_) for
// backward.  spatial_dims_ holds the per-spatial-axis sizes used to recompute
// spatial_total in the backward pass.
class LUCID_API GroupNormBackward : public FuncOp<GroupNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;             // Per-(batch, group) mean, shape (B*G,).
    Storage saved_rstd_;             // Per-(batch, group) reciprocal std.
    int B_ = 0, C_ = 0, G_ = 0;      // Batch, channel, and group counts.
    std::vector<int> spatial_dims_;  // Per-axis spatial sizes.

    // Run the forward pass.
    // x         – input of shape (B, C, S0, ...).  Rank must be >= 2.
    // gamma     – scale of shape (C,).
    // beta      – shift of shape (C,).
    // num_groups – must divide C evenly.
    // eps       – numerical stability constant.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 int num_groups,
                                 double eps);

    // Compute gradients for x (dx), gamma (d_gamma), and beta (d_beta).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: delegates to GroupNormBackward::forward.
LUCID_API TensorImplPtr group_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      int num_groups,
                                      double eps);

}  // namespace lucid
