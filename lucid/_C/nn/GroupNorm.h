#pragma once

// =====================================================================
// Lucid C++ engine — GroupNorm.
// =====================================================================
//
//   group_norm(x, γ, β, num_groups, eps)
//     x : (B, C, H, W)   (or (B, C, *) — only requires the channel axis at 1)
//     γ, β : (C,)
//
//   Channels are split into `num_groups` contiguous chunks of `C/num_groups`
//   channels each. Within each (B, group) slice, mean/var are computed
//   across (C/G, H, W). Then per-channel γ/β affine.
//
//   InstanceNorm == GroupNorm with num_groups = C.
//   LayerNorm    ≠ GroupNorm: LN normalizes over a fixed trailing-dim shape
//                              and applies γ/β shaped like that trailing
//                              shape; GN is per-channel γ/β.
//
// AMP policy: ForceFP32.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API GroupNormBackward : public FuncOp<GroupNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;  // shape (B, G)
    Storage saved_rstd_;  // shape (B, G)
    int B_ = 0, C_ = 0, G_ = 0;
    std::vector<int> spatial_dims_;  // arbitrary spatial rank (size N)

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 int num_groups,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr group_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      int num_groups,
                                      double eps);

}  // namespace lucid
