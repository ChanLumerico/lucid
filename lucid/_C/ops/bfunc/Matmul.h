#pragma once

#include <utility>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API MatmulBackward : public FuncOp<MatmulBackward, 2> {
public:
    static const OpSchema schema_v1;

    static TensorImplPtr forward(const TensorImplPtr& a, const TensorImplPtr& b);

    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
