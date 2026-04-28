#include "Bitwise.h"

#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
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

bool is_integer_or_bool(Dtype dt) {
    switch (dt) {
        case Dtype::Bool:
        case Dtype::I8:
        case Dtype::I16:
        case Dtype::I32:
        case Dtype::I64:
            return true;
        default:
            return false;
    }
}

template <typename Op>
CpuStorage bit_kernel(
    const CpuStorage& a, const CpuStorage& b, std::size_t numel, Dtype dt, Op op) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    auto run = [&](auto* dst, auto* p, auto* q) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<T>(op(p[i], q[i]));
    };
    switch (dt) {
        case Dtype::I32:
            run(reinterpret_cast<std::int32_t*>(out.ptr.get()),
                reinterpret_cast<const std::int32_t*>(a.ptr.get()),
                reinterpret_cast<const std::int32_t*>(b.ptr.get()));
            break;
        case Dtype::I64:
            run(reinterpret_cast<std::int64_t*>(out.ptr.get()),
                reinterpret_cast<const std::int64_t*>(a.ptr.get()),
                reinterpret_cast<const std::int64_t*>(b.ptr.get()));
            break;
        case Dtype::I16:
            run(reinterpret_cast<std::int16_t*>(out.ptr.get()),
                reinterpret_cast<const std::int16_t*>(a.ptr.get()),
                reinterpret_cast<const std::int16_t*>(b.ptr.get()));
            break;
        case Dtype::I8:
            run(reinterpret_cast<std::int8_t*>(out.ptr.get()),
                reinterpret_cast<const std::int8_t*>(a.ptr.get()),
                reinterpret_cast<const std::int8_t*>(b.ptr.get()));
            break;
        case Dtype::Bool:
            run(reinterpret_cast<std::uint8_t*>(out.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(a.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(b.ptr.get()));
            break;
        default:
            ErrorBuilder("bitwise").not_implemented("dtype must be integer or bool");
    }
    return out;
}

template <typename MlxFn, typename Op>
TensorImplPtr bit_dispatch(
    const TensorImplPtr& a, const TensorImplPtr& b, const char* name, MlxFn mlx_fn, Op op) {
    validate_pair_eq_shape(a, b, name);
    if (!is_integer_or_bool(a->dtype()))
        ErrorBuilder(name).fail("dtype must be integer or bool");
    OpScopeFull scope{name, a->device(), a->dtype(), a->shape()};
    if (a->device() == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        const auto& gb = std::get<GpuStorage>(b->storage());
        auto out = mlx_fn(*ga.arr, *gb.arr);
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())}, a->shape(),
                     a->dtype(), a->device());
    }
    auto out = bit_kernel(std::get<CpuStorage>(a->storage()), std::get<CpuStorage>(b->storage()),
                          shape_numel(a->shape()), a->dtype(), op);
    return fresh(Storage{std::move(out)}, a->shape(), a->dtype(), a->device());
}

}  // namespace

TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(
        a, b, "bitwise_and",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::bitwise_and(x, y);
        },
        [](auto x, auto y) -> std::int64_t {
            return static_cast<std::int64_t>(x) & static_cast<std::int64_t>(y);
        });
}

TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(
        a, b, "bitwise_or",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::bitwise_or(x, y);
        },
        [](auto x, auto y) -> std::int64_t {
            return static_cast<std::int64_t>(x) | static_cast<std::int64_t>(y);
        });
}

TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(
        a, b, "bitwise_xor",
        [](const ::mlx::core::array& x, const ::mlx::core::array& y) {
            return ::mlx::core::bitwise_xor(x, y);
        },
        [](auto x, auto y) -> std::int64_t {
            return static_cast<std::int64_t>(x) ^ static_cast<std::int64_t>(y);
        });
}

}  // namespace lucid
