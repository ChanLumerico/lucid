#include "Pow.h"

#include <mlx/ops.h>

#include "../../backend/cpu/Vforce.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"
#include "../../autograd/Helpers.h"

namespace lucid {

const OpSchema PowBackward::schema_v1{
    "pow", /*version=*/1, AmpPolicy::ForceFP32, /*deterministic=*/true};

CpuStorage PowBackward::cpu_kernel(const CpuStorage& a, const CpuStorage& b,
                                   const Shape& out_shape, Dtype dt) {
    const std::size_t numel = shape_numel(out_shape);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);

    switch (dt) {
        case Dtype::F32:
            backend::cpu::vpow_f32(
                reinterpret_cast<const float*>(a.ptr.get()),
                reinterpret_cast<const float*>(b.ptr.get()),
                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vpow_f64(
                reinterpret_cast<const double*>(a.ptr.get()),
                reinterpret_cast<const double*>(b.ptr.get()),
                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError("pow: dtype not supported");
    }
    return out;
}


GpuStorage PowBackward::gpu_kernel(const GpuStorage& a,
                                  const GpuStorage& b,
                                  const Shape& /*out_shape*/, Dtype dt) {
    if (!a.arr || !b.arr) {
        throw LucidError("pow: null GPU input");
    }
    auto out = ::mlx::core::power(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

std::pair<Storage, Storage> PowBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    // Broadcast saved inputs so a/b are aligned with grad_out's shape.
    Storage a_buf = saved_input_broadcasted(0);
    Storage b_buf = saved_input_broadcasted(1);
    const auto& a = a_buf;
    const auto& b = b_buf;

    // dx = b * a^(b-1) * grad_out
    Storage b_minus_one = add_scalar_storage(b, -1.0, n, dtype_, device_);
    Storage a_pow_bm1   = pow_storage(a, b_minus_one, n, dtype_, device_);
    Storage b_times     = multiply_storages(b, a_pow_bm1, n, dtype_, device_);
    Storage dx          = multiply_storages(b_times, grad_out, n, dtype_, device_);

    // dy = log(a) * a^b * grad_out
    Storage log_a   = log_storage(a, n, dtype_, device_);
    Storage a_pow_b = pow_storage(a, b, n, dtype_, device_);
    Storage prod    = multiply_storages(log_a, a_pow_b, n, dtype_, device_);
    Storage dy      = multiply_storages(prod, grad_out, n, dtype_, device_);

    return {std::move(dx), std::move(dy)};
}

TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return PowBackward::forward(a, b);
}

LUCID_REGISTER_OP(PowBackward)

}  // namespace lucid
