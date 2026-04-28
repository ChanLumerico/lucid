#include "Div.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema DivBackward::schema_v1{"div", /*version=*/1, AmpPolicy::Promote,
                                      /*deterministic=*/true};

CpuStorage DivBackward::cpu_kernel(const CpuStorage& a,
                                   const CpuStorage& b,
                                   const Shape& out_shape,
                                   Dtype dt) {
    const std::size_t numel = shape_numel(out_shape);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);

    switch (dt) {
        case Dtype::F32:
            backend::cpu::vdiv_f32(reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<const float*>(b.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vdiv_f64(reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<const double*>(b.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError("div: dtype not supported");
    }
    return out;
}

GpuStorage DivBackward::gpu_kernel(const GpuStorage& a,
                                   const GpuStorage& b,
                                   const Shape& /*out_shape*/,
                                   Dtype dt) {
    if (!a.arr || !b.arr) {
        throw LucidError("div: null GPU input");
    }
    auto out = ::mlx::core::divide(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

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
