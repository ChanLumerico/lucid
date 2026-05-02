#include "Mul.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MulBackward::schema_v1{"mul", 1, AmpPolicy::Promote, true};

std::pair<Storage, Storage> MulBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    auto a_b = saved_input_broadcasted(0);
    auto b_b = saved_input_broadcasted(1);
    return {
        multiply_storages(grad_out, b_b, n, dtype_, device_),
        multiply_storages(grad_out, a_b, n, dtype_, device_),
    };
}

TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MulBackward)

}  // namespace lucid
