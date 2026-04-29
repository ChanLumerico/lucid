#pragma once

// =====================================================================
// Lucid C++ engine — RMSNorm.
// =====================================================================
//
//   rms = √(mean(x²) + ε)
//   y   = γ · x / rms
//
// Cheaper variant of LayerNorm — no mean subtraction, no β. Common in modern
// LLMs (LLaMA, T5).
//
// Backward: returns (dx, dγ).
// AMP policy: ForceFP32.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

/// Autograd backward node for RMSNorm.
class LUCID_API RMSNormBackward : public FuncOp<RMSNormBackward, 2> {
public:
    static const OpSchema schema_v1;
    Storage saved_rstd_;
    std::size_t outer_ = 0;
    std::size_t N_ = 0;

    static TensorImplPtr forward(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Rms norm.
LUCID_API TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);

}  // namespace lucid
