#include "Div.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema DivBackward::schema_v1{"div", /*version=*/1, AmpPolicy::Promote,
                                      /*deterministic=*/true};


std::pair<Storage, Storage> DivBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    // Broadcast saved inputs so all element-wise ops below are well-defined
    // when forward used broadcasting (e.g. (4,5) / (5,)).
    auto a_b = saved_input_broadcasted(0);
    auto b_b = saved_input_broadcasted(1);
    // dx = g / b
    Storage dx = divide_storages(grad_out, b_b, n, dtype_, device_);
    // dy = -g * a / b^2
    Storage b_sq = square_storage(b_b, n, dtype_, device_);
    Storage g_times_a = multiply_storages(grad_out, a_b, n, dtype_, device_);
    Storage div_by_b_sq = divide_storages(g_times_a, b_sq, n, dtype_, device_);
    Storage dy = negate_storage(div_by_b_sq, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return DivBackward::forward(a, b);
}

LUCID_REGISTER_OP(DivBackward)

}  // namespace lucid
