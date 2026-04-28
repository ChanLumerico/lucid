#include "Compare.h"

#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair_eq_shape;

template <typename Cmp>
CpuStorage cmp_kernel(const CpuStorage& a, const CpuStorage& b, std::size_t numel, Cmp cmp) {
    CpuStorage out;
    out.dtype = Dtype::Bool;
    out.nbytes = numel;
    out.ptr = allocate_aligned_bytes(out.nbytes);
    auto* dst = reinterpret_cast<std::uint8_t*>(out.ptr.get());
    auto run = [&](auto* p, auto* q) {
        for (std::size_t i = 0; i < numel; ++i)
            dst[i] = cmp(p[i], q[i]) ? 1u : 0u;
    };
    switch (a.dtype) {
        case Dtype::F32:
            run(reinterpret_cast<const float*>(a.ptr.get()),
                reinterpret_cast<const float*>(b.ptr.get()));
            break;
        case Dtype::F64:
            run(reinterpret_cast<const double*>(a.ptr.get()),
                reinterpret_cast<const double*>(b.ptr.get()));
            break;
        case Dtype::I32:
            run(reinterpret_cast<const std::int32_t*>(a.ptr.get()),
                reinterpret_cast<const std::int32_t*>(b.ptr.get()));
            break;
        case Dtype::I64:
            run(reinterpret_cast<const std::int64_t*>(a.ptr.get()),
                reinterpret_cast<const std::int64_t*>(b.ptr.get()));
            break;
        case Dtype::I16:
            run(reinterpret_cast<const std::int16_t*>(a.ptr.get()),
                reinterpret_cast<const std::int16_t*>(b.ptr.get()));
            break;
        case Dtype::I8:
            run(reinterpret_cast<const std::int8_t*>(a.ptr.get()),
                reinterpret_cast<const std::int8_t*>(b.ptr.get()));
            break;
        case Dtype::Bool:
            run(reinterpret_cast<const std::uint8_t*>(a.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(b.ptr.get()));
            break;
        default:
            throw NotImplementedError("compare: dtype not supported");
    }
    return out;
}

template <typename MlxFn, typename Cmp>
TensorImplPtr cmp_dispatch(
    const TensorImplPtr& a, const TensorImplPtr& b, const char* name, MlxFn mlx_fn, Cmp cmp) {
    validate_pair_eq_shape(a, b, name);
    OpScope scope{name, a->device_, a->dtype_, a->shape_};
    if (a->device_ == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        auto out = mlx_fn(*ga.arr, *gb.arr);
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), Dtype::Bool)}, a->shape_,
                     Dtype::Bool, a->device_);
    }
    auto out = cmp_kernel(std::get<CpuStorage>(a->storage_), std::get<CpuStorage>(b->storage_),
                          shape_numel(a->shape_), cmp);
    return fresh(Storage{std::move(out)}, a->shape_, Dtype::Bool, a->device_);
}

}  // namespace

TensorImplPtr equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "equal",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::equal(x, y);
        },
        [](auto x, auto y) { return x == y; });
}

TensorImplPtr not_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "not_equal",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::not_equal(x, y);
        },
        [](auto x, auto y) { return x != y; });
}

TensorImplPtr greater_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "greater",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::greater(x, y);
        },
        [](auto x, auto y) { return x > y; });
}

TensorImplPtr greater_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "greater_equal",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::greater_equal(x, y);
        },
        [](auto x, auto y) { return x >= y; });
}

TensorImplPtr less_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "less",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::less(x, y);
        },
        [](auto x, auto y) { return x < y; });
}

TensorImplPtr less_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(
        a, b, "less_equal",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::less_equal(x, y);
        },
        [](auto x, auto y) { return x <= y; });
}

}  // namespace lucid
