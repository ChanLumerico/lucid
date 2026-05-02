#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API SoftmaxBackward : public FuncOp<SoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr softmax_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
