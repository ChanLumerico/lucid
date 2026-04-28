#include "Helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../backend/cpu/Vdsp.h"
#include "../backend/cpu/Vforce.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/TensorImpl.h"

namespace lucid {

namespace {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <typename T>
void fill_typed(std::byte* dst, std::size_t numel, T value) {
    auto* tdst = reinterpret_cast<T*>(dst);
    for (std::size_t i = 0; i < numel; ++i)
        tdst[i] = value;
}

void fill_storage(CpuStorage& s, const Shape& shape, double value) {
    const std::size_t n = shape_numel(shape);
    switch (s.dtype) {
        case Dtype::Bool: {
            // bool: 0 or 1
            std::byte v = (value != 0.0) ? std::byte{1} : std::byte{0};
            std::memset(s.ptr.get(), static_cast<int>(v), n);
            break;
        }
        case Dtype::I8:
            fill_typed<std::int8_t>(s.ptr.get(), n, static_cast<std::int8_t>(value));
            break;
        case Dtype::I16:
            fill_typed<std::int16_t>(s.ptr.get(), n, static_cast<std::int16_t>(value));
            break;
        case Dtype::I32:
            fill_typed<std::int32_t>(s.ptr.get(), n, static_cast<std::int32_t>(value));
            break;
        case Dtype::I64:
            fill_typed<std::int64_t>(s.ptr.get(), n, static_cast<std::int64_t>(value));
            break;
        case Dtype::F32:
            fill_typed<float>(s.ptr.get(), n, static_cast<float>(value));
            break;
        case Dtype::F64:
            fill_typed<double>(s.ptr.get(), n, value);
            break;
        case Dtype::F16:
            // Phase 2 doesn't ship F16 ops; punt until Phase 3 brings in
            // a half-precision shim.
            ErrorBuilder("autograd.fill").not_implemented("F16 fill not yet implemented");
        case Dtype::C64: {
            std::complex<float> v(static_cast<float>(value), 0.0f);
            fill_typed<std::complex<float>>(s.ptr.get(), n, v);
            break;
        }
    }
}

CpuStorage allocate_autograd_cpu(const Shape& shape, Dtype dtype) {
    const std::size_t total = shape_numel(shape) * dtype_size(dtype);
    CpuStorage s;
    s.ptr = allocate_aligned_bytes(total);
    s.nbytes = total;
    s.dtype = dtype;
    return s;
}

template <typename T>
void add_typed(std::byte* dst, const std::byte* src, std::size_t numel) {
    auto* td = reinterpret_cast<T*>(dst);
    const auto* ts = reinterpret_cast<const T*>(src);
    for (std::size_t i = 0; i < numel; ++i)
        td[i] = td[i] + ts[i];
}

void cpu_add_inplace(CpuStorage& dst, const CpuStorage& src) {
    if (dst.dtype != src.dtype) {
        throw DtypeMismatch(std::string(dtype_name(dst.dtype)), std::string(dtype_name(src.dtype)),
                            "accumulate_into");
    }
    if (dst.nbytes != src.nbytes) {
        ErrorBuilder("accumulate_into").fail("nbytes mismatch");
    }
    const std::size_t n = dst.nbytes / dtype_size(dst.dtype);
    switch (dst.dtype) {
        case Dtype::F32:
            add_typed<float>(dst.ptr.get(), src.ptr.get(), n);
            break;
        case Dtype::F64:
            add_typed<double>(dst.ptr.get(), src.ptr.get(), n);
            break;
        case Dtype::I32:
            add_typed<std::int32_t>(dst.ptr.get(), src.ptr.get(), n);
            break;
        case Dtype::I64:
            add_typed<std::int64_t>(dst.ptr.get(), src.ptr.get(), n);
            break;
        default:
            ErrorBuilder("accumulate_into").not_implemented("dtype not yet supported in Phase 2");
    }
}

// Compute the axes of `grad_shape` that need to be sum-reduced to align with
// `target_shape` under right-aligned broadcasting (numpy semantics).
std::vector<std::size_t> broadcast_reduce_axes(const Shape& grad_shape, const Shape& target_shape) {
    std::vector<std::size_t> axes;
    const std::size_t gn = grad_shape.size();
    const std::size_t tn = target_shape.size();
    if (gn < tn) {
        throw ShapeMismatch(target_shape, grad_shape,
                            "reduce_grad_to_shape (grad rank < target rank)");
    }
    const std::size_t lead = gn - tn;
    for (std::size_t i = 0; i < lead; ++i)
        axes.push_back(i);
    for (std::size_t i = 0; i < tn; ++i) {
        const std::int64_t g = grad_shape[lead + i];
        const std::int64_t t = target_shape[i];
        if (g == t)
            continue;
        if (t == 1) {
            axes.push_back(lead + i);
        } else {
            throw ShapeMismatch(target_shape, grad_shape,
                                "reduce_grad_to_shape: incompatible broadcast");
        }
    }
    return axes;
}

// Brute-force reduce-along-axes for arbitrary rank, F32/F64 only in Phase 2.
// We iterate over the gradient's flat index, decompose into multi-dim coords,
// project into the target shape (collapsing reduced axes to 0), and accumulate.
template <typename T>
void reduce_axes_typed(const std::byte* src_bytes,
                       std::byte* dst_bytes,
                       const Shape& grad_shape,
                       const Shape& target_shape,
                       const std::vector<std::size_t>& reduce_axes_set) {
    const std::size_t grad_numel = shape_numel(grad_shape);
    const std::size_t target_numel = shape_numel(target_shape);
    const auto* src = reinterpret_cast<const T*>(src_bytes);
    auto* dst = reinterpret_cast<T*>(dst_bytes);
    std::fill_n(dst, target_numel, T{});

    const std::size_t gn = grad_shape.size();
    const std::size_t tn = target_shape.size();
    const std::size_t lead = gn - tn;

    std::vector<bool> reduce_mask(gn, false);
    for (auto a : reduce_axes_set)
        reduce_mask[a] = true;

    Stride grad_idx_stride(gn);
    if (gn > 0) {
        grad_idx_stride[gn - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(gn) - 2; i >= 0; --i) {
            grad_idx_stride[static_cast<std::size_t>(i)] =
                grad_idx_stride[static_cast<std::size_t>(i) + 1] *
                grad_shape[static_cast<std::size_t>(i) + 1];
        }
    }

    Stride target_idx_stride(tn);
    if (tn > 0) {
        target_idx_stride[tn - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(tn) - 2; i >= 0; --i) {
            target_idx_stride[static_cast<std::size_t>(i)] =
                target_idx_stride[static_cast<std::size_t>(i) + 1] *
                target_shape[static_cast<std::size_t>(i) + 1];
        }
    }

    for (std::size_t flat = 0; flat < grad_numel; ++flat) {
        std::size_t rem = flat;
        std::size_t target_flat = 0;
        for (std::size_t d = 0; d < gn; ++d) {
            const std::size_t coord = rem / static_cast<std::size_t>(grad_idx_stride[d]);
            rem -= coord * static_cast<std::size_t>(grad_idx_stride[d]);
            if (d < lead)
                continue;  // squashed by leading-1 padding
            if (reduce_mask[d])
                continue;  // collapsed axis
            const std::size_t td = d - lead;
            target_flat += coord * static_cast<std::size_t>(target_idx_stride[td]);
        }
        dst[target_flat] += src[flat];
    }
}

}  // namespace

Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device) {
    if (device == Device::GPU) {
        auto out = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dtype));
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
    }
    auto cpu = allocate_autograd_cpu(shape, dtype);
    fill_storage(cpu, shape, 0.0);
    return Storage{std::move(cpu)};
}

Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device) {
    if (device == Device::GPU) {
        auto out = ::mlx::core::ones(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dtype));
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
    }
    auto cpu = allocate_autograd_cpu(shape, dtype);
    fill_storage(cpu, shape, 1.0);
    return Storage{std::move(cpu)};
}

Storage reduce_grad_to_shape(const Storage& grad,
                             const Shape& grad_shape,
                             const Shape& target_shape,
                             Dtype dtype,
                             Device device) {
    if (device == Device::GPU) {
        const auto& src_gpu = std::get<GpuStorage>(grad);
        if (!src_gpu.arr) {
            ErrorBuilder("reduce_grad_to_shape").fail("null GPU array");
        }
        if (grad_shape == target_shape) {
            auto cloned = ::mlx::core::copy(*src_gpu.arr);
            return Storage{gpu::wrap_mlx_array(std::move(cloned), dtype)};
        }
        // Broadcast-back over multi-axis: derive the same axis set we use on
        // CPU, then call mlx::core::sum with `keepdims=false` and reshape to
        // the target shape. MLX expects axes in descending order to be safe,
        // but it sorts internally. The CPU helper returns ascending; pass as-is.
        const auto reduce_axes = broadcast_reduce_axes(grad_shape, target_shape);
        std::vector<int> axes_i;
        axes_i.reserve(reduce_axes.size());
        for (auto a : reduce_axes)
            axes_i.push_back(static_cast<int>(a));
        auto reduced = ::mlx::core::sum(*src_gpu.arr, axes_i, /*keepdims=*/false);
        // The reduced shape may be missing leading-1 dims relative to target;
        // reshape to the exact target_shape to match Lucid's contract.
        auto reshaped = ::mlx::core::reshape(reduced, gpu::to_mlx_shape(target_shape));
        return Storage{gpu::wrap_mlx_array(std::move(reshaped), dtype)};
    }
    if (grad_shape == target_shape) {
        // Fast-path: no reduction needed; just clone (so the engine owns its
        // own buffer for further accumulation).
        const auto& src_cpu = std::get<CpuStorage>(grad);
        CpuStorage out = allocate_autograd_cpu(grad_shape, dtype);
        if (out.nbytes > 0) {
            std::memcpy(out.ptr.get(), src_cpu.ptr.get(), out.nbytes);
        }
        return Storage{std::move(out)};
    }

    const auto reduce_axes = broadcast_reduce_axes(grad_shape, target_shape);
    const auto& src_cpu = std::get<CpuStorage>(grad);
    CpuStorage out = allocate_autograd_cpu(target_shape, dtype);

    switch (dtype) {
        case Dtype::F32:
            reduce_axes_typed<float>(src_cpu.ptr.get(), out.ptr.get(), grad_shape, target_shape,
                                     reduce_axes);
            break;
        case Dtype::F64:
            reduce_axes_typed<double>(src_cpu.ptr.get(), out.ptr.get(), grad_shape, target_shape,
                                      reduce_axes);
            break;
        default:
            ErrorBuilder("reduce_grad_to_shape")
                .not_implemented("dtype not yet supported in Phase 2");
    }
    return Storage{std::move(out)};
}

void accumulate_into(Storage& dst, const Storage& src) {
    std::visit(overloaded{
                   [&](CpuStorage& d, const CpuStorage& s) { cpu_add_inplace(d, s); },
                   [&](GpuStorage& d, const GpuStorage& s) {
                       if (!d.arr || !s.arr) {
                           ErrorBuilder("accumulate_into").fail("null GPU array");
                       }
                       if (d.dtype != s.dtype) {
                           throw DtypeMismatch(std::string(dtype_name(d.dtype)),
                                               std::string(dtype_name(s.dtype)), "accumulate_into");
                       }
                       // MLX is functional; we replace dst's array with `dst + src`.
                       // Refcount drops on the previous array via the tracked deleter.
                       auto next = ::mlx::core::add(*d.arr, *s.arr);
                       d.arr = gpu::wrap_mlx_array(std::move(next), d.dtype).arr;
                   },
                   [&](auto&, auto&) {
                       throw DeviceMismatch("matching device", "mixed CPU/GPU", "accumulate_into");
                   },
               },
               dst, src);
}

// ----------------------------------------------------------------------
// Storage-level math primitives (Phase 3.1+).
// Each allocates a fresh CpuStorage of the same numel/dtype.
// GPU paths route through mlx::core ops (Phase 3.7.2+).
// ----------------------------------------------------------------------

namespace {

CpuStorage allocate_like(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

// GPU path helper: F is a callable that takes one or more `mlx::core::array`
// references and returns an `mlx::core::array`. The result is wrapped into a
// GpuStorage with MemoryTracker accounting.
template <class F>
Storage gpu_unary(const Storage& s, Dtype dt, F&& f) {
    const auto& g = std::get<GpuStorage>(s);
    if (!g.arr)
        ErrorBuilder("gpu_unary").fail("null GPU array");
    auto out = f(*g.arr);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

template <class F>
Storage gpu_binary(const Storage& a, const Storage& b, Dtype dt, F&& f) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gb = std::get<GpuStorage>(b);
    if (!ga.arr || !gb.arr) {
        ErrorBuilder("gpu_binary").fail("null GPU array");
    }
    auto out = f(*ga.arr, *gb.arr);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

}  // namespace

Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::negative(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vneg_f32(reinterpret_cast<const float*>(src.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vneg_f64(reinterpret_cast<const double*>(src.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("negate_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage multiply_storages(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt,
                          [](const auto& x, const auto& y) { return ::mlx::core::multiply(x, y); });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vmul_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                            reinterpret_cast<const float*>(sb.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vmul_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                            reinterpret_cast<const double*>(sb.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("multiply_storages").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage divide_storages(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt,
                          [](const auto& x, const auto& y) { return ::mlx::core::divide(x, y); });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vdiv_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                            reinterpret_cast<const float*>(sb.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vdiv_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                            reinterpret_cast<const double*>(sb.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("divide_storages").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage add_storages(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt,
                          [](const auto& x, const auto& y) { return ::mlx::core::add(x, y); });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vadd_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                            reinterpret_cast<const float*>(sb.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vadd_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                            reinterpret_cast<const double*>(sb.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("add_storages").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage subtract_storages(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt,
                          [](const auto& x, const auto& y) { return ::mlx::core::subtract(x, y); });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vsub_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                            reinterpret_cast<const float*>(sb.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vsub_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                            reinterpret_cast<const double*>(sb.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("subtract_storages").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::square(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vsq_f32(reinterpret_cast<const float*>(src.ptr.get()),
                                           reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vsq_f64(reinterpret_cast<const double*>(src.ptr.get()),
                                           reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("square_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::copy(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    if (out.nbytes > 0)
        std::memcpy(out.ptr.get(), src.ptr.get(), out.nbytes);
    return Storage{std::move(out)};
}

Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::log(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vlog_f32(reinterpret_cast<const float*>(src.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vlog_f64(reinterpret_cast<const double*>(src.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("log_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage pow_storage(
    const Storage& base, const Storage& expo, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(base, expo, dt,
                          [](const auto& x, const auto& y) { return ::mlx::core::power(x, y); });
    }
    const auto& sb = std::get<CpuStorage>(base);
    const auto& se = std::get<CpuStorage>(expo);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vpow_f32(reinterpret_cast<const float*>(sb.ptr.get()),
                                            reinterpret_cast<const float*>(se.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vpow_f64(reinterpret_cast<const double*>(sb.ptr.get()),
                                            reinterpret_cast<const double*>(se.ptr.get()),
                                            reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("pow_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage ge_mask_storage(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt, [dt](const auto& x, const auto& y) {
            return ::mlx::core::astype(::mlx::core::greater_equal(x, y), gpu::to_mlx_dtype(dt));
        });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vge_mask_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                                reinterpret_cast<const float*>(sb.ptr.get()),
                                                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vge_mask_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                                reinterpret_cast<const double*>(sb.ptr.get()),
                                                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("ge_mask_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage lt_mask_storage(
    const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_binary(a, b, dt, [dt](const auto& x, const auto& y) {
            return ::mlx::core::astype(::mlx::core::less(x, y), gpu::to_mlx_dtype(dt));
        });
    }
    const auto& sa = std::get<CpuStorage>(a);
    const auto& sb = std::get<CpuStorage>(b);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vle_mask_f32(reinterpret_cast<const float*>(sa.ptr.get()),
                                                reinterpret_cast<const float*>(sb.ptr.get()),
                                                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vle_mask_f64(reinterpret_cast<const double*>(sa.ptr.get()),
                                                reinterpret_cast<const double*>(sb.ptr.get()),
                                                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("lt_mask_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage add_scalar_storage(
    const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [scalar, dt](const auto& a) {
            ::mlx::core::array c(scalar, gpu::to_mlx_dtype(dt));
            return ::mlx::core::add(a, c);
        });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vsadd_f32(reinterpret_cast<const float*>(src.ptr.get()),
                                             static_cast<float>(scalar),
                                             reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vsadd_f64(reinterpret_cast<const double*>(src.ptr.get()), scalar,
                                             reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("add_scalar_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage mul_scalar_storage(
    const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [scalar, dt](const auto& a) {
            ::mlx::core::array c(scalar, gpu::to_mlx_dtype(dt));
            return ::mlx::core::multiply(a, c);
        });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            ::lucid::backend::cpu::vsmul_f32(reinterpret_cast<const float*>(src.ptr.get()),
                                             static_cast<float>(scalar),
                                             reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            ::lucid::backend::cpu::vsmul_f64(reinterpret_cast<const double*>(src.ptr.get()), scalar,
                                             reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("mul_scalar_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

// Generic dispatch helper for unary "vForce-style" kernels (in, out, n).
// CPU branch only — GPU is routed by the per-helper wrappers below.
namespace {

template <class F32Fn, class F64Fn>
Storage apply_unary_dispatch_cpu(
    const Storage& s, std::size_t numel, Dtype dt, F32Fn f32, F64Fn f64, const char* op_name) {
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32:
            f32(reinterpret_cast<const float*>(src.ptr.get()),
                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            f64(reinterpret_cast<const double*>(src.ptr.get()),
                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder(op_name).not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

}  // namespace

#define LUCID_UNARY_HELPER(NAME, F32, F64, MLX_OP)                                         \
    Storage NAME##_storage(const Storage& s, std::size_t n, Dtype dt, Device device) {     \
        if (device == Device::GPU) {                                                       \
            return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::MLX_OP(a); }); \
        }                                                                                  \
        return apply_unary_dispatch_cpu(s, n, dt, ::lucid::backend::cpu::F32,              \
                                        ::lucid::backend::cpu::F64, #NAME);                \
    }

LUCID_UNARY_HELPER(exp, vexp_f32, vexp_f64, exp)
LUCID_UNARY_HELPER(sqrt, vsqrt_f32, vsqrt_f64, sqrt)
LUCID_UNARY_HELPER(abs, vfabs_f32, vfabs_f64, abs)
LUCID_UNARY_HELPER(reciprocal, vrec_f32, vrec_f64, reciprocal)
LUCID_UNARY_HELPER(sin, vsin_f32, vsin_f64, sin)
LUCID_UNARY_HELPER(cos, vcos_f32, vcos_f64, cos)
LUCID_UNARY_HELPER(asin, vasin_f32, vasin_f64, arcsin)
LUCID_UNARY_HELPER(acos, vacos_f32, vacos_f64, arccos)
LUCID_UNARY_HELPER(atan, vatan_f32, vatan_f64, arctan)
LUCID_UNARY_HELPER(sinh, vsinh_f32, vsinh_f64, sinh)
LUCID_UNARY_HELPER(cosh, vcosh_f32, vcosh_f64, cosh)
LUCID_UNARY_HELPER(tanh, vtanh_f32, vtanh_f64, tanh)

#undef LUCID_UNARY_HELPER

Storage tan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::tan(a); });
    }
    return apply_unary_dispatch_cpu(s, numel, dt, ::lucid::backend::cpu::vtan_f32,
                                    ::lucid::backend::cpu::vtan_f64, "tan");
}

// Sign: out[i] = (in[i] > 0) - (in[i] < 0). Scalar loop.
Storage sign_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::sign(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(src.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] > 0.f) - (p[i] < 0.f);
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(src.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] > 0.0) - (p[i] < 0.0);
            break;
        }
        default:
            ErrorBuilder("sign_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage in_range_mask_storage(
    const Storage& s, double lo, double hi, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [lo, hi, dt](const auto& a) {
            ::mlx::core::array lo_arr(lo, gpu::to_mlx_dtype(dt));
            ::mlx::core::array hi_arr(hi, gpu::to_mlx_dtype(dt));
            auto in_range = ::mlx::core::logical_and(::mlx::core::greater_equal(a, lo_arr),
                                                     ::mlx::core::less_equal(a, hi_arr));
            return ::mlx::core::astype(in_range, gpu::to_mlx_dtype(dt));
        });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(src.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto flo = static_cast<float>(lo);
            const auto fhi = static_cast<float>(hi);
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] >= flo && p[i] <= fhi) ? 1.f : 0.f;
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(src.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] >= lo && p[i] <= hi) ? 1.0 : 0.0;
            break;
        }
        default:
            ErrorBuilder("in_range_mask_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

// --------------------------- Version check (Phase 3.3.6 retrofit) ----------

void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name,
                         std::size_t input_idx) {
    auto t = live.lock();
    if (!t)
        return;  // tensor freed; nothing to compare against
    if (t->version() != saved_version) {
        throw VersionMismatch(saved_version, t->version(),
                              std::string(op_name) + " input " + std::to_string(input_idx));
    }
}

// --------------------------- Reduce-op helpers (Phase 3.3) ---------------

std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim) {
    std::vector<int> out;
    if (axes.empty()) {
        out.reserve(ndim);
        for (int i = 0; i < ndim; ++i)
            out.push_back(i);
        return out;
    }
    std::vector<bool> seen(ndim, false);
    for (int a : axes) {
        int wrapped = a < 0 ? a + ndim : a;
        if (wrapped < 0 || wrapped >= ndim) {
            ErrorBuilder("normalize_axes")
                .index_error(std::string("axis out of range: ") + std::to_string(a) +
                             " for ndim=" + std::to_string(ndim));
        }
        if (!seen[wrapped]) {
            seen[wrapped] = true;
            out.push_back(wrapped);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

Shape reduce_output_shape(const Shape& input_shape, const std::vector<int>& axes, bool keepdims) {
    std::vector<bool> reduce_mask(input_shape.size(), false);
    for (int a : axes)
        reduce_mask[a] = true;
    Shape out;
    out.reserve(input_shape.size());
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (reduce_mask[i]) {
            if (keepdims)
                out.push_back(1);
        } else {
            out.push_back(input_shape[i]);
        }
    }
    return out;
}

namespace {

// Right-aligned multi-index decomposition. Given flat index `flat` and shape,
// fill `idx[]` so that `flat = sum_d (idx[d] * stride[d])`.
void unravel(std::size_t flat, const Shape& shape, std::vector<std::int64_t>& idx) {
    idx.resize(shape.size());
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(shape.size()) - 1; d >= 0; --d) {
        const std::size_t dim = static_cast<std::size_t>(shape[d]);
        idx[d] = static_cast<std::int64_t>(flat % dim);
        flat /= dim;
    }
}

template <typename T>
void broadcast_back_typed(const T* g, T* out, const Shape& kept_shape, const Shape& input_shape) {
    // For each position in input_shape, look up the corresponding position in
    // kept_shape (axes that were reduced are at index 0 in kept_shape).
    const std::size_t in_numel = shape_numel(input_shape);
    std::vector<std::int64_t> idx;
    for (std::size_t flat = 0; flat < in_numel; ++flat) {
        unravel(flat, input_shape, idx);
        // Map to kept index: for each dim, if kept_shape == 1 (reduced), use 0.
        std::size_t kept_flat = 0;
        std::size_t stride = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(input_shape.size()) - 1; d >= 0; --d) {
            const std::int64_t kd = kept_shape[d];
            const std::int64_t ii = (kd == 1) ? 0 : idx[d];
            kept_flat += static_cast<std::size_t>(ii) * stride;
            stride *= static_cast<std::size_t>(kd);
        }
        out[flat] = g[kept_flat];
    }
}

}  // namespace

Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims,
                                  Dtype dt,
                                  Device device) {
    // 1. Build "kept_shape" (input_shape with reduced axes set to 1).
    Shape kept_shape = input_shape;
    for (int a : axes)
        kept_shape[a] = 1;

    // 2. Validate that grad_shape matches the expected reduced shape.
    Shape expected_grad = reduce_output_shape(input_shape, axes, keepdims);
    if (grad_shape != expected_grad) {
        throw ShapeMismatch(expected_grad, grad_shape,
                            "broadcast_back_for_reduce: grad shape mismatch");
    }

    if (device == Device::GPU) {
        const auto& g = std::get<GpuStorage>(grad);
        if (!g.arr)
            ErrorBuilder("broadcast_back_for_reduce").fail("null GPU array");
        // Reshape to kept_shape (inserts the size-1 reduced axes back), then
        // broadcast_to back to input_shape. MLX's broadcast_to is a metadata
        // op, but the result of a reduce op feeding through here will be
        // backed by allocated data; we rely on MLX's eager eval to materialize
        // the broadcast when a downstream consumer reads it.
        auto reshaped = ::mlx::core::reshape(*g.arr, gpu::to_mlx_shape(kept_shape));
        auto bcast = ::mlx::core::broadcast_to(reshaped, gpu::to_mlx_shape(input_shape));
        // Materialize the broadcast — `mlx::core::contiguous` forces a fresh
        // strided=1 buffer with the broadcast values written out, so
        // downstream readers see the broadcast actually applied.
        auto cloned = ::mlx::core::contiguous(bcast);
        return Storage{gpu::wrap_mlx_array(std::move(cloned), dt)};
    }

    // 3. If !keepdims, the grad data layout matches `kept_shape` after a
    //    reshape (no copy needed — same numel, same order).
    const auto& g_cpu = std::get<CpuStorage>(grad);

    // 4. Allocate output of input_shape size and broadcast.
    auto out = allocate_like(shape_numel(input_shape), dt);
    switch (dt) {
        case Dtype::F32:
            broadcast_back_typed<float>(reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                        reinterpret_cast<float*>(out.ptr.get()), kept_shape,
                                        input_shape);
            break;
        case Dtype::F64:
            broadcast_back_typed<double>(reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                         reinterpret_cast<double*>(out.ptr.get()), kept_shape,
                                         input_shape);
            break;
        default:
            ErrorBuilder("broadcast_back_for_reduce").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage leaky_mask_storage(
    const Storage& s, double slope, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [slope, dt](const auto& a) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array slope_arr(slope, gpu::to_mlx_dtype(dt));
            return ::mlx::core::where(::mlx::core::greater_equal(a, zero), one, slope_arr);
        });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(src.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto fs = static_cast<float>(slope);
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] >= 0.f) ? 1.f : fs;
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(src.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] >= 0.0) ? 1.0 : slope;
            break;
        }
        default:
            ErrorBuilder("leaky_mask_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [](const auto& a) { return ::mlx::core::sigmoid(a); });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    // Numerically stable sigmoid:
    //   x >= 0 : 1 / (1 + exp(-x))
    //   x <  0 : exp(x) / (1 + exp(x))
    // Avoids overflow of exp(-x) for very negative x.
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(src.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) {
                const float x = p[i];
                if (x >= 0.f) {
                    const float e = std::exp(-x);
                    q[i] = 1.f / (1.f + e);
                } else {
                    const float e = std::exp(x);
                    q[i] = e / (1.f + e);
                }
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(src.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) {
                const double x = p[i];
                if (x >= 0.0) {
                    const double e = std::exp(-x);
                    q[i] = 1.0 / (1.0 + e);
                } else {
                    const double e = std::exp(x);
                    q[i] = e / (1.0 + e);
                }
            }
            break;
        }
        default:
            ErrorBuilder("sigmoid_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

Storage bernoulli_mask_storage_shape(
    double keep_prob, const Shape& shape, Dtype dt, Device device, Generator& gen) {
    if (keep_prob < 0.0 || keep_prob > 1.0) {
        ErrorBuilder("bernoulli_mask").fail("keep_prob must be in [0, 1]");
    }
    std::size_t numel = 1;
    for (auto d : shape)
        numel *= static_cast<std::size_t>(d);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto kp = static_cast<float>(keep_prob);
            for (std::size_t i = 0; i < numel; ++i) {
                q[i] = (gen.next_uniform_float() < kp) ? 1.f : 0.f;
            }
            break;
        }
        case Dtype::F64: {
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) {
                q[i] = (static_cast<double>(gen.next_uniform_float()) < keep_prob) ? 1.0 : 0.0;
            }
            break;
        }
        default:
            ErrorBuilder("bernoulli_mask").not_implemented("dtype not supported (F32/F64)");
    }
    if (device == Device::GPU) {
        return Storage{gpu::upload_cpu_to_gpu(out, shape)};
    }
    return Storage{std::move(out)};
}

Storage bernoulli_mask_storage(
    double keep_prob, std::size_t numel, Dtype dt, Device device, Generator& gen) {
    Shape flat;
    flat.push_back(static_cast<std::int64_t>(numel));
    return bernoulli_mask_storage_shape(keep_prob, flat, dt, device, gen);
}

Storage positive_mask_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    if (device == Device::GPU) {
        return gpu_unary(s, dt, [dt](const auto& a) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            return ::mlx::core::astype(::mlx::core::greater(a, zero), gpu::to_mlx_dtype(dt));
        });
    }
    const auto& src = std::get<CpuStorage>(s);
    auto out = allocate_like(numel, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(src.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] > 0.f) ? 1.f : 0.f;
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(src.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                q[i] = (p[i] > 0.0) ? 1.0 : 0.0;
            break;
        }
        default:
            ErrorBuilder("positive_mask_storage").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

// ============================================================
// Random storage helpers (Phase 3.8)
// ============================================================

namespace {

// Fill a typed buffer with uniform [lo, hi) draws from a Philox generator.
template <typename T>
void fill_uniform(T* dst, std::size_t numel, double lo, double hi, Generator& gen) {
    const T span = static_cast<T>(hi - lo);
    const T offset = static_cast<T>(lo);
    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = static_cast<T>(gen.next_uniform_float()) * span + offset;
    }
}

// Box-Muller transform: pairs of uniform → pairs of standard normal.
template <typename T>
void fill_normal(T* dst, std::size_t numel, double mean, double std, Generator& gen) {
    const T m = static_cast<T>(mean);
    const T s = static_cast<T>(std);
    constexpr T two_pi = static_cast<T>(6.28318530717958647692);
    constexpr T eps = static_cast<T>(1e-7);
    std::size_t i = 0;
    while (i + 1 < numel) {
        // u1 ∈ [eps, 1) so log(u1) is finite.
        T u1 = static_cast<T>(gen.next_uniform_float());
        if (u1 < eps)
            u1 = eps;
        T u2 = static_cast<T>(gen.next_uniform_float());
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        const T z0 = r * std::cos(two_pi * u2);
        const T z1 = r * std::sin(two_pi * u2);
        dst[i] = m + s * z0;
        dst[i + 1] = m + s * z1;
        i += 2;
    }
    if (i < numel) {  // odd tail
        T u1 = static_cast<T>(gen.next_uniform_float());
        if (u1 < eps)
            u1 = eps;
        T u2 = static_cast<T>(gen.next_uniform_float());
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        dst[i] = m + s * r * std::cos(two_pi * u2);
    }
}

// Uniform integer in [low, high). Uses raw uint32 with modulo (slight bias
// for non-power-of-2 ranges; acceptable for typical sizes).
template <typename Int>
void fill_randint(
    Int* dst, std::size_t numel, std::int64_t low, std::int64_t high, Generator& gen) {
    const std::uint64_t range = static_cast<std::uint64_t>(high - low);
    if (range == 0) {
        for (std::size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<Int>(low);
        return;
    }
    std::uint32_t buf[4];
    std::size_t i = 0;
    while (i < numel) {
        gen.next_uint32x4(buf);
        for (int k = 0; k < 4 && i < numel; ++k, ++i) {
            std::uint64_t r = buf[k];
            // For ranges > 2^32, draw two u32 to extend.
            if (range > 0xFFFFFFFFull) {
                std::uint32_t buf2[4];
                gen.next_uint32x4(buf2);
                r = (r << 32) | buf2[0];
            }
            dst[i] = static_cast<Int>(low) + static_cast<Int>(r % range);
        }
    }
}

CpuStorage allocate_for_random(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

}  // namespace

Storage random_uniform_storage(
    const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen) {
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
        case Dtype::F32:
            fill_uniform<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, lo, hi, gen);
            break;
        case Dtype::F64:
            fill_uniform<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, lo, hi, gen);
            break;
        default:
            ErrorBuilder("random_uniform").not_implemented("dtype not supported (F32/F64)");
    }
    if (device == Device::GPU) {
        return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
    }
    return Storage{std::move(cpu)};
}

Storage random_normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen) {
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
        case Dtype::F32:
            fill_normal<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, mean, std, gen);
            break;
        case Dtype::F64:
            fill_normal<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, mean, std, gen);
            break;
        default:
            ErrorBuilder("random_normal").not_implemented("dtype not supported (F32/F64)");
    }
    if (device == Device::GPU) {
        return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
    }
    return Storage{std::move(cpu)};
}

Storage random_bernoulli_storage(
    const Shape& shape, double p, Dtype dt, Device device, Generator& gen) {
    if (p < 0.0 || p > 1.0)
        ErrorBuilder("random_bernoulli").fail("p must be in [0, 1]");
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
        case Dtype::F32: {
            auto* q = reinterpret_cast<float*>(cpu.ptr.get());
            const auto fp = static_cast<float>(p);
            for (std::size_t i = 0; i < n; ++i) {
                q[i] = (gen.next_uniform_float() < fp) ? 1.f : 0.f;
            }
            break;
        }
        case Dtype::F64: {
            auto* q = reinterpret_cast<double*>(cpu.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                q[i] = (static_cast<double>(gen.next_uniform_float()) < p) ? 1.0 : 0.0;
            }
            break;
        }
        default:
            ErrorBuilder("random_bernoulli").not_implemented("dtype not supported (F32/F64)");
    }
    if (device == Device::GPU) {
        return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
    }
    return Storage{std::move(cpu)};
}

Storage random_randint_storage(const Shape& shape,
                               std::int64_t low,
                               std::int64_t high,
                               Dtype dt,
                               Device device,
                               Generator& gen) {
    if (high <= low)
        ErrorBuilder("random_randint").fail("high must be > low");
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
        case Dtype::I32:
            fill_randint<std::int32_t>(reinterpret_cast<std::int32_t*>(cpu.ptr.get()), n, low, high,
                                       gen);
            break;
        case Dtype::I64:
            fill_randint<std::int64_t>(reinterpret_cast<std::int64_t*>(cpu.ptr.get()), n, low, high,
                                       gen);
            break;
        default:
            ErrorBuilder("random_randint").not_implemented("dtype not supported (I32/I64)");
    }
    if (device == Device::GPU) {
        return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
    }
    return Storage{std::move(cpu)};
}

}  // namespace lucid
