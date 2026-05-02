#include "Sub.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema SubBackward::schema_v1{"sub", 1, AmpPolicy::Promote, true};

std::pair<Storage, Storage> SubBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    return {
        clone_storage(grad_out, n, dtype_, device_),
        negate_storage(grad_out, n, dtype_, device_),
    };
}

TensorImplPtr sub_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return SubBackward::forward(a, b);
}

LUCID_REGISTER_OP(SubBackward)

}  // namespace lucid
