#include "Maximum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MaximumBackward::schema_v1{"maximum", /*version=*/1, AmpPolicy::Promote,
                                          /*deterministic=*/true};

CpuStorage MaximumBackward::cpu_kernel(const CpuStorage& a,
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
            backend::cpu::vmax_f32(reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<const float*>(b.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vmax_f64(reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<const double*>(b.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("maximum").not_implemented("dtype not supported");
    }
    return out;
}

GpuStorage MaximumBackward::gpu_kernel(const GpuStorage& a,
                                       const GpuStorage& b,
                                       const Shape& /*out_shape*/,
                                       Dtype dt) {
    if (!a.arr || !b.arr) {
        ErrorBuilder("maximum").fail("null GPU input");
    }
    auto out = ::mlx::core::maximum(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

std::pair<Storage, Storage> MaximumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask_a = ge_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_,
                                     device_);  // a >= b (ties to a)
    Storage mask_b =
        lt_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_, device_);  // a < b
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MaximumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MaximumBackward)

}  // namespace lucid
