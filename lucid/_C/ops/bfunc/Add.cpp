#include "Add.h"

#include <cstdint>
#include <cstring>
#include <utility>

#include <mlx/ops.h>

#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"
#include "../../core/Shape.h"

namespace lucid {

const OpSchema AddBackward::schema_v1{"add", /*version=*/1, AmpPolicy::Promote,
                                      /*deterministic=*/true};

CpuStorage AddBackward::cpu_kernel(const CpuStorage& a,
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
            backend::cpu::vadd_f32(reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<const float*>(b.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vadd_f64(reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<const double*>(b.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        case Dtype::I32:
            backend::cpu::vadd_i32(reinterpret_cast<const std::int32_t*>(a.ptr.get()),
                                   reinterpret_cast<const std::int32_t*>(b.ptr.get()),
                                   reinterpret_cast<std::int32_t*>(out.ptr.get()), numel);
            break;
        case Dtype::I64:
            backend::cpu::vadd_i64(reinterpret_cast<const std::int64_t*>(a.ptr.get()),
                                   reinterpret_cast<const std::int64_t*>(b.ptr.get()),
                                   reinterpret_cast<std::int64_t*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError(std::string("add: dtype ") + std::string(dtype_name(dt)) +
                                      " not supported in Phase 3.0");
    }
    return out;
}

GpuStorage AddBackward::gpu_kernel(const GpuStorage& a,
                                   const GpuStorage& b,
                                   const Shape& /*out_shape*/,
                                   Dtype dt) {
    if (!a.arr || !b.arr) {
        throw LucidError("add: null GPU input");
    }
    auto out = ::mlx::core::add(*a.arr, *b.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

std::pair<Storage, Storage> AddBackward::grad_formula(const Storage& grad_out) {
    // d(a+b)/da = 1, d(a+b)/db = 1. Both grads are exactly grad_out.
    // We clone so each downstream consumer owns its buffer (engine accumulates
    // into pending grads). reduce_grad_to_shape with same shape returns a
    // clone, so we can just call it with the trivial mapping — but that's
    // overkill. Build a direct clone via a no-op reduce.
    return {
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
    };
}

TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return AddBackward::forward(a, b);
}

LUCID_REGISTER_OP(AddBackward)

}  // namespace lucid
