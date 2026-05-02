#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API LinearBackward : public FuncOp<LinearBackward, 3> {
public:
    static const OpSchema schema_v1;
    static TensorImplPtr
    forward(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr linear_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b);

}  // namespace lucid
