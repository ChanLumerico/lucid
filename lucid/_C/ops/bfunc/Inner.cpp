#include "Inner.h"

#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

}  // namespace

TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "inner");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"inner", device, dt, Shape{}};

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        auto out = ::mlx::core::inner(*ga.arr, *gb.arr);
        Shape out_shape;
        for (auto d : out.shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                     std::move(out_shape), dt, device);
    }

    const auto& sa = a->shape_;
    const auto& sb = b->shape_;
    if (sa.empty() || sb.empty() || sa.back() != sb.back())
        throw ShapeMismatch(sa, sb, "inner");
    const std::int64_t K = sa.back();
    Shape out_shape(sa.begin(), sa.end() - 1);
    out_shape.insert(out_shape.end(), sb.begin(), sb.end() - 1);
    auto out_cpu = allocate_cpu(out_shape, dt);
    const std::size_t pa = shape_numel(Shape(sa.begin(), sa.end() - 1));
    const std::size_t pb = shape_numel(Shape(sb.begin(), sb.end() - 1));
    const auto& ca = std::get<CpuStorage>(a->storage_);
    const auto& cb = std::get<CpuStorage>(b->storage_);
    auto run = [&](auto* op, const auto* ap, const auto* bp) {
        using T = std::remove_pointer_t<decltype(op)>;
        for (std::size_t i = 0; i < pa; ++i)
            for (std::size_t j = 0; j < pb; ++j) {
                T s{};
                for (std::int64_t k = 0; k < K; ++k)
                    s = s + ap[i * K + k] * bp[j * K + k];
                op[i * pb + j] = s;
            }
    };
    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out_cpu.ptr.get()),
            reinterpret_cast<const float*>(ca.ptr.get()),
            reinterpret_cast<const float*>(cb.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out_cpu.ptr.get()),
            reinterpret_cast<const double*>(ca.ptr.get()),
            reinterpret_cast<const double*>(cb.ptr.get()));
    else
        throw NotImplementedError("inner: dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
}

}  // namespace lucid
