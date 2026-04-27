#include "Sub.h"

#include <mlx/ops.h>

#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"
#include "../../autograd/Helpers.h"

namespace lucid {

const OpSchema SubBackward::schema_v1{
    "sub", /*version=*/1, AmpPolicy::Promote, /*deterministic=*/true};

CpuStorage SubBackward::cpu_kernel(const CpuStorage& a, const CpuStorage& b,
                                   const Shape& out_shape, Dtype dt) {
    const std::size_t numel = shape_numel(out_shape);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);

    switch (dt) {
        case Dtype::F32:
            backend::cpu::vsub_f32(
                reinterpret_cast<const float*>(a.ptr.get()),
                reinterpret_cast<const float*>(b.ptr.get()),
                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vsub_f64(
                reinterpret_cast<const double*>(a.ptr.get()),
                reinterpret_cast<const double*>(b.ptr.get()),
                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError("sub: dtype not supported");
    }
    return out;
}


GpuStorage SubBackward::gpu_kernel(const GpuStorage& a,
                                  const GpuStorage& b,
                                  const Shape& /*out_shape*/, Dtype dt) {
    if (!a.arr || !b.arr) {
        throw LucidError("sub: null GPU input");
    }
    auto out = ::mlx::core::subtract(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

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
