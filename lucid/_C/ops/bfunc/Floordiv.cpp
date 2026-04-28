#include "Floordiv.h"

#include <cmath>
#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair_eq_shape;

}  // namespace

TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair_eq_shape(a, b, "floordiv");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScopeFull scope{"floordiv", device, dt, a->shape_};

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        // mlx::floor_divide on integer inputs truncates toward zero, which
        // disagrees with numpy/PyTorch (floor toward -∞) for negative
        // numerators. Cast to float32, take the true floor, then cast back
        // to int64 — matches the CPU branch's std::floor semantics.
        auto a_f = ::mlx::core::astype(*ga.arr, ::mlx::core::float32);
        auto b_f = ::mlx::core::astype(*gb.arr, ::mlx::core::float32);
        auto q = ::mlx::core::floor(::mlx::core::divide(a_f, b_f));
        auto out_i = ::mlx::core::astype(q, ::mlx::core::int64);
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out_i), Dtype::I64)}, a->shape_,
                     Dtype::I64, device);
    }

    const auto& ca = std::get<CpuStorage>(a->storage_);
    const auto& cb = std::get<CpuStorage>(b->storage_);
    const std::size_t n = shape_numel(a->shape_);
    CpuStorage out;
    out.dtype = Dtype::I64;
    out.nbytes = n * sizeof(std::int64_t);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    auto* dst = reinterpret_cast<std::int64_t*>(out.ptr.get());
    auto run = [&](const auto* p, const auto* q) {
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = static_cast<std::int64_t>(
                std::floor(static_cast<double>(p[i]) / static_cast<double>(q[i])));
    };
    switch (dt) {
        case Dtype::F32:
            run(reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
            break;
        case Dtype::F64:
            run(reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
            break;
        case Dtype::I32:
            run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int32_t*>(cb.ptr.get()));
            break;
        case Dtype::I64:
            run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int64_t*>(cb.ptr.get()));
            break;
        default:
            ErrorBuilder("floordiv").not_implemented("dtype not supported");
    }
    return fresh(Storage{std::move(out)}, a->shape_, Dtype::I64, device);
}

}  // namespace lucid
