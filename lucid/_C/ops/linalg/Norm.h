#pragma once

// =====================================================================
// norm: Lp-norm reduction. Backward for L1 and L2.
// =====================================================================

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Norm.
class LUCID_API NormBackward : public FuncOp<NormBackward, 1> {
public:
    static const OpSchema schema_v1;
    double ord_ = 2.0;
    std::vector<int> axis_;
    bool keepdims_ = false;
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Norm.
LUCID_API TensorImplPtr norm_op(const TensorImplPtr& a,
                                double ord,
                                std::vector<int> axis,
                                bool keepdims);

}  // namespace lucid
