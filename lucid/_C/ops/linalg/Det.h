#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API DetBackward : public FuncOp<DetBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr det_op(const TensorImplPtr& a);

}  // namespace lucid
