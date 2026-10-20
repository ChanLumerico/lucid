// lucid/_C/ops/bfunc/Pow.cpp
//
// Implements PowBackward::grad_formula and the pow_op free function.

#include "Pow.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

// ForceFP32: under autocast a floating power runs in float32, as the
// reference runs it — log() and fractional exponents are unstable in half
// precision, and ``x ** 2.0`` (pow_scalar) already answered float32.
// ``integral_closed``: an integer or bool power is exact in its own dtype,
// inside autocast and out, so the policy's int-to-float32 promotion is off.
const OpSchema PowBackward::schema_v1{
    "pow", 1, AmpPolicy::ForceFP32, true, "", /*real=*/false, /*closed=*/true};

// Gradient of element-wise power (a ^ b).
//
// Power rule (gradient w.r.t. a):
//   dA = b * a^(b-1) * grad_out
//      = b * pow(a, b - 1) * grad_out
//
// Exponential rule (gradient w.r.t. b):
//   dB = log(a) * a^b * grad_out
//
// Both inputs are retrieved at the broadcast output shape before any arithmetic.
std::pair<Storage, Storage> PowBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const bool need_a = needs_input_grad(0);
    const bool need_b = needs_input_grad(1);
    if (!need_a && !need_b)
        return {Storage{}, Storage{}};

    Storage a_buf = saved_input_broadcasted(0);
    Storage b_buf = saved_input_broadcasted(1);
    const auto& a = a_buf;
    const auto& b = b_buf;

    // dA = b * a^(b-1) * grad_out
    Storage dx;
    if (need_a) {
        Storage b_minus_one = add_scalar_storage(b, -1.0, n, dtype_, device_);
        Storage a_pow_bm1 = pow_storage(a, b_minus_one, n, dtype_, device_);
        Storage b_times = multiply_storages(b, a_pow_bm1, n, dtype_, device_);
        dx = multiply_storages(b_times, grad_out, n, dtype_, device_);
    }

    // dB = log(a) * a^b * grad_out
    Storage dy;
    if (need_b) {
        Storage log_a = log_storage(a, n, dtype_, device_);
        Storage a_pow_b = pow_storage(a, b, n, dtype_, device_);
        Storage prod = multiply_storages(log_a, a_pow_b, n, dtype_, device_);
        dy = multiply_storages(prod, grad_out, n, dtype_, device_);
    }

    return {std::move(dx), std::move(dy)};
}

TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return PowBackward::forward(a, b);
}

LUCID_REGISTER_OP(PowBackward)

}  // namespace lucid
