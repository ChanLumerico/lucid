#pragma once

// =====================================================================
// Lucid C++ engine — LayerNorm.
// =====================================================================
//
//   y = γ · (x - mean) / √(var + ε) + β
//
// `normalized_shape` specifies which trailing dims to normalize over;
// γ and β have shape == normalized_shape. Most common: D=1 (transformer
// pre-/post-norm). The kernel flattens to (outer, N) where N = product of
// normalized dims, outer = product of all leading dims.
//
// Backward: returns (dx, dγ, dβ).
// AMP policy: ForceFP32 (precision-sensitive — fp16 layernorm collapses).
//
// Layer: autograd/ops/norm/.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

/// Autograd backward node for LayerNorm.
class LUCID_API LayerNormBackward : public FuncOp<LayerNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;
    Storage saved_rstd_;
    std::size_t outer_ = 0;
    std::size_t N_ = 0;

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Layer norm.
LUCID_API TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps);

}  // namespace lucid
