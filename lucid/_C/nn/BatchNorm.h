#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

template <int N>

class LUCID_API BatchNormNdBackward : public FuncOp<BatchNormNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;
    Storage saved_rstd_;
    int B_ = 0, C_ = 0;
    int S_[N > 0 ? N : 1];

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

using BatchNorm1dBackward = BatchNormNdBackward<1>;
using BatchNorm2dBackward = BatchNormNdBackward<2>;
using BatchNorm3dBackward = BatchNormNdBackward<3>;

LUCID_API TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);

LUCID_API TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps = 1e-5);

LUCID_API TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);

}  // namespace lucid
