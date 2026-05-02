#include "Pow.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema PowBackward::schema_v1{"pow", 1, AmpPolicy::ForceFP32, true};

std::pair<Storage, Storage> PowBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    Storage a_buf = saved_input_broadcasted(0);
    Storage b_buf = saved_input_broadcasted(1);
    const auto& a = a_buf;
    const auto& b = b_buf;

    Storage b_minus_one = add_scalar_storage(b, -1.0, n, dtype_, device_);
    Storage a_pow_bm1 = pow_storage(a, b_minus_one, n, dtype_, device_);
    Storage b_times = multiply_storages(b, a_pow_bm1, n, dtype_, device_);
    Storage dx = multiply_storages(b_times, grad_out, n, dtype_, device_);

    Storage log_a = log_storage(a, n, dtype_, device_);
    Storage a_pow_b = pow_storage(a, b, n, dtype_, device_);
    Storage prod = multiply_storages(log_a, a_pow_b, n, dtype_, device_);
    Storage dy = multiply_storages(prod, grad_out, n, dtype_, device_);

    return {std::move(dx), std::move(dy)};
}

TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return PowBackward::forward(a, b);
}

LUCID_REGISTER_OP(PowBackward)

}  // namespace lucid
