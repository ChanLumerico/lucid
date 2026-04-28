#include "Mul.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MulBackward::schema_v1{"mul", /*version=*/1, AmpPolicy::Promote,
                                      /*deterministic=*/true};

CpuStorage MulBackward::cpu_kernel(const CpuStorage& a,
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
            backend::cpu::vmul_f32(reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<const float*>(b.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vmul_f64(reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<const double*>(b.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError("mul: dtype not supported");
    }
    return out;
}

GpuStorage MulBackward::gpu_kernel(const GpuStorage& a,
                                   const GpuStorage& b,
                                   const Shape& /*out_shape*/,
                                   Dtype dt) {
    if (!a.arr || !b.arr) {
        throw LucidError("mul: null GPU input");
    }
    auto out = ::mlx::core::multiply(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

std::pair<Storage, Storage> MulBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g * b_saved, dy = g * a_saved. Broadcast saved inputs to out_shape_
    // first so the element-wise multiply is well-defined under broadcasting.
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
