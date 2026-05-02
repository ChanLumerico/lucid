#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API BatchNormEvalBackward : public FuncOp<BatchNormEvalBackward, 5> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-5;
    Storage rstd_;

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                           const TensorImplPtr& mean,
                                           const TensorImplPtr& var,
                                           const TensorImplPtr& gamma,
                                           const TensorImplPtr& beta,
                                           double eps);

class LUCID_API LpNormalizeBackward : public FuncOp<LpNormalizeBackward, 1> {
public:
    static const OpSchema schema_v1;
    double ord_ = 2.0;
    int axis_ = 1;
    double eps_ = 1e-12;
    Storage saved_norm_;

    static TensorImplPtr forward(const TensorImplPtr& x, double ord, int axis, double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps);

class LUCID_API GlobalResponseNormBackward : public FuncOp<GlobalResponseNormBackward, 3> {
public:
    static const OpSchema schema_v1;
    double eps_ = 1e-6;

    Storage saved_Nx_;

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                                const TensorImplPtr& gamma,
                                                const TensorImplPtr& beta,
                                                double eps);

}  // namespace lucid
