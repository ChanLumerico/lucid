#pragma once

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
    Storage saved_mean_;
    Storage saved_rstd_;
    int B_ = 0, C_ = 0, G_ = 0;
    std::vector<int> spatial_dims_;

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
