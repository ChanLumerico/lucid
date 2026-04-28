#include "Gfunc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"

namespace lucid {

namespace {

using helpers::allocate_cpu;

template <typename T>
void fill_typed(std::byte* dst, std::size_t numel, T value) {
    auto* p = reinterpret_cast<T*>(dst);
    for (std::size_t i = 0; i < numel; ++i)
        p[i] = value;
}

void fill_cpu(CpuStorage& s, std::size_t numel, double value) {
    switch (s.dtype) {
        case Dtype::Bool:
            fill_typed<std::uint8_t>(s.ptr.get(), numel, value != 0.0 ? 1u : 0u);
            break;
        case Dtype::I8:
            fill_typed<std::int8_t>(s.ptr.get(), numel, static_cast<std::int8_t>(value));
            break;
        case Dtype::I16:
            fill_typed<std::int16_t>(s.ptr.get(), numel, static_cast<std::int16_t>(value));
            break;
        case Dtype::I32:
            fill_typed<std::int32_t>(s.ptr.get(), numel, static_cast<std::int32_t>(value));
            break;
        case Dtype::I64:
            fill_typed<std::int64_t>(s.ptr.get(), numel, static_cast<std::int64_t>(value));
            break;
        case Dtype::F32:
            fill_typed<float>(s.ptr.get(), numel, static_cast<float>(value));
            break;
        case Dtype::F64:
            fill_typed<double>(s.ptr.get(), numel, value);
            break;
        default:
            ErrorBuilder("creation").not_implemented("dtype not supported for fill");
    }
}

inline TensorImplPtr finalize(
    Storage&& storage, Shape shape, Dtype dt, Device device, bool requires_grad) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device,
                                        requires_grad);
}

}  // namespace

// ----------------------------------------------------------------------------
// zeros
// ----------------------------------------------------------------------------
TensorImplPtr zeros_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"zeros", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// ones
// ----------------------------------------------------------------------------
TensorImplPtr ones_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"ones", device, dt, shape};
    auto s = make_ones_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// full
// ----------------------------------------------------------------------------
TensorImplPtr full_op(
    const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"full", device, dt, shape};
    if (device == Device::GPU) {
        auto ones = ::mlx::core::ones(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
        ::mlx::core::array scalar = [&]() {
            switch (dt) {
                case Dtype::F32:
                    return ::mlx::core::array(static_cast<float>(fill_value));
                case Dtype::F64:
                    return ::mlx::core::array(fill_value, gpu::to_mlx_dtype(dt));
                case Dtype::I32:
                    return ::mlx::core::array(static_cast<int32_t>(fill_value));
                case Dtype::I64:
                    return ::mlx::core::array(static_cast<int64_t>(fill_value));
                case Dtype::Bool:
                    return ::mlx::core::array(fill_value != 0.0);
                default:
                    ErrorBuilder("full").not_implemented("GPU dtype not supported");
            }
        }();
        auto out = ::mlx::core::multiply(ones, scalar);
        return finalize(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, shape, dt, device,
                        requires_grad);
    }
    auto cpu = allocate_cpu(shape, dt);
    fill_cpu(cpu, shape_numel(shape), fill_value);
    return finalize(Storage{std::move(cpu)}, shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// empty (allocates without zeroing — but we still zero on CPU for safety;
//        callers shouldn't rely on garbage values)
// ----------------------------------------------------------------------------
TensorImplPtr empty_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"empty", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// eye(N, M, k): identity-like matrix of shape (N, M) with ones on the k-th
// diagonal. Negative `k` is below the main diagonal, positive above.
// ----------------------------------------------------------------------------
TensorImplPtr eye_op(
    std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt, Device device, bool requires_grad) {
    if (M <= 0)
        M = N;
    if (N < 0 || M < 0)
        ErrorBuilder("eye").fail("N and M must be >= 0");
    Shape shape{N, M};
    OpScopeFull scope{"eye", device, dt, shape};
    if (device == Device::GPU) {
        auto out = ::mlx::core::eye(static_cast<int>(N), static_cast<int>(M), static_cast<int>(k),
                                    gpu::to_mlx_dtype(dt));
        return finalize(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, shape, dt, device,
                        requires_grad);
    }
    auto cpu = allocate_cpu(shape, dt);
    auto set_one = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < N; ++i) {
            const std::int64_t j = i + k;
            if (j < 0 || j >= M)
                continue;
            p[i * M + j] = static_cast<T>(1);
        }
    };
    switch (dt) {
        case Dtype::Bool:
            set_one(reinterpret_cast<std::uint8_t*>(cpu.ptr.get()));
            break;
        case Dtype::I8:
            set_one(reinterpret_cast<std::int8_t*>(cpu.ptr.get()));
            break;
        case Dtype::I16:
            set_one(reinterpret_cast<std::int16_t*>(cpu.ptr.get()));
            break;
        case Dtype::I32:
            set_one(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
            break;
        case Dtype::I64:
            set_one(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
            break;
        case Dtype::F32:
            set_one(reinterpret_cast<float*>(cpu.ptr.get()));
            break;
        case Dtype::F64:
            set_one(reinterpret_cast<double*>(cpu.ptr.get()));
            break;
        default:
            ErrorBuilder("eye").not_implemented("dtype not supported");
    }
    return finalize(Storage{std::move(cpu)}, shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// arange(start, stop, step) — Python/numpy semantics.
// ----------------------------------------------------------------------------
TensorImplPtr arange_op(
    double start, double stop, double step, Dtype dt, Device device, bool requires_grad) {
    if (step == 0.0)
        ErrorBuilder("arange").fail("step must be non-zero");
    const double diff = stop - start;
    const std::int64_t n =
        (diff * step <= 0) ? 0 : static_cast<std::int64_t>(std::ceil(diff / step));
    Shape shape{n};
    OpScopeFull scope{"arange", device, dt, shape};

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    };

    if (device == Device::GPU) {
        auto cpu = allocate_cpu(shape, dt);
        switch (dt) {
            case Dtype::F32:
                compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
                break;
            case Dtype::F64:
                compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
                break;
            case Dtype::I32:
                compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
                break;
            case Dtype::I64:
                compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
                break;
            default:
                ErrorBuilder("arange").not_implemented("dtype not supported");
        }
        auto gpu = gpu::upload_cpu_to_gpu(cpu, shape);
        return finalize(Storage{std::move(gpu)}, shape, dt, device, requires_grad);
    }
    auto cpu = allocate_cpu(shape, dt);
    switch (dt) {
        case Dtype::F32:
            compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
            break;
        case Dtype::F64:
            compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
            break;
        case Dtype::I32:
            compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
            break;
        case Dtype::I64:
            compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
            break;
        default:
            ErrorBuilder("arange").not_implemented("dtype not supported");
    }
    return finalize(Storage{std::move(cpu)}, shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// linspace(start, stop, num) — `num >= 1` and includes both endpoints when
// `num >= 2`. For `num == 1` we return `[start]`.
// ----------------------------------------------------------------------------
TensorImplPtr linspace_op(
    double start, double stop, std::int64_t num, Dtype dt, Device device, bool requires_grad) {
    if (num < 0)
        ErrorBuilder("linspace").fail("num must be >= 0");
    Shape shape{num};
    OpScopeFull scope{"linspace", device, dt, shape};
    const double step = (num > 1) ? (stop - start) / static_cast<double>(num - 1) : 0.0;

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < num; ++i) {
            const double v = (num == 1) ? start : start + static_cast<double>(i) * step;
            p[i] = static_cast<T>(v);
        }
        // Ensure exact endpoint hit despite FP step accumulation.
        if (num >= 2)
            p[num - 1] = static_cast<T>(stop);
    };

    auto cpu = allocate_cpu(shape, dt);
    switch (dt) {
        case Dtype::F32:
            compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
            break;
        case Dtype::F64:
            compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
            break;
        case Dtype::I32:
            compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
            break;
        case Dtype::I64:
            compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
            break;
        default:
            ErrorBuilder("linspace").not_implemented("dtype not supported");
    }
    if (device == Device::GPU) {
        auto gpu = gpu::upload_cpu_to_gpu(cpu, shape);
        return finalize(Storage{std::move(gpu)}, shape, dt, device, requires_grad);
    }
    return finalize(Storage{std::move(cpu)}, shape, dt, device, requires_grad);
}

// ----------------------------------------------------------------------------
// diag(v, k):
//   1-D v of length L → 2-D matrix of shape (L+|k|, L+|k|) with v on diag k.
//   2-D v of shape (M, N) → 1-D vector with the k-th diagonal extracted.
// ----------------------------------------------------------------------------
TensorImplPtr diag_op(const TensorImplPtr& v, std::int64_t k) {
    if (!v)
        ErrorBuilder("diag").fail("input is null");
    const Dtype dt = v->dtype();
    const Device device = v->device();
    const auto& sh = v->shape();
    if (sh.size() != 1 && sh.size() != 2) {
        ErrorBuilder("diag").fail("input must be 1-D or 2-D");
    }

    if (device == Device::GPU) {
        const auto& gv = std::get<GpuStorage>(v->storage());
        if (!gv.arr)
            ErrorBuilder("diag").fail("null GPU input");
        auto out = ::mlx::core::diag(*gv.arr, static_cast<int>(k));
        Shape out_shape;
        for (auto d : out.shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        return std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                                            std::move(out_shape), dt, device,
                                            /*requires_grad=*/false);
    }

    const auto& cv = std::get<CpuStorage>(v->storage());

    if (sh.size() == 1) {
        const std::int64_t L = sh[0];
        const std::int64_t side = L + std::abs(k);
        Shape out_shape{side, side};
        auto cpu = allocate_cpu(out_shape, dt);
        auto fill_diag = [&](auto* dst, const auto* src) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::int64_t i = 0; i < L; ++i) {
                const std::int64_t row = (k >= 0) ? i : (i - k);
                const std::int64_t col = (k >= 0) ? (i + k) : i;
                dst[row * side + col] = static_cast<T>(src[i]);
            }
        };
        switch (dt) {
            case Dtype::F32:
                fill_diag(reinterpret_cast<float*>(cpu.ptr.get()),
                          reinterpret_cast<const float*>(cv.ptr.get()));
                break;
            case Dtype::F64:
                fill_diag(reinterpret_cast<double*>(cpu.ptr.get()),
                          reinterpret_cast<const double*>(cv.ptr.get()));
                break;
            case Dtype::I32:
                fill_diag(reinterpret_cast<std::int32_t*>(cpu.ptr.get()),
                          reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
                break;
            case Dtype::I64:
                fill_diag(reinterpret_cast<std::int64_t*>(cpu.ptr.get()),
                          reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
                break;
            default:
                ErrorBuilder("diag").not_implemented("dtype not supported");
        }
        return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(out_shape), dt,
                                            device, false);
    }

    // 2-D input → extract diagonal
    const std::int64_t M = sh[0], N = sh[1];
    const std::int64_t r0 = (k >= 0) ? 0 : -k;
    const std::int64_t c0 = (k >= 0) ? k : 0;
    const std::int64_t L = std::min(M - r0, N - c0);
    const std::int64_t out_len = std::max<std::int64_t>(L, 0);
    Shape out_shape{out_len};
    auto cpu = allocate_cpu(out_shape, dt);
    auto extract = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::int64_t i = 0; i < out_len; ++i) {
            dst[i] = static_cast<T>(src[(r0 + i) * N + (c0 + i)]);
        }
    };
    switch (dt) {
        case Dtype::F32:
            extract(reinterpret_cast<float*>(cpu.ptr.get()),
                    reinterpret_cast<const float*>(cv.ptr.get()));
            break;
        case Dtype::F64:
            extract(reinterpret_cast<double*>(cpu.ptr.get()),
                    reinterpret_cast<const double*>(cv.ptr.get()));
            break;
        case Dtype::I32:
            extract(reinterpret_cast<std::int32_t*>(cpu.ptr.get()),
                    reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
            break;
        case Dtype::I64:
            extract(reinterpret_cast<std::int64_t*>(cpu.ptr.get()),
                    reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
            break;
        default:
            ErrorBuilder("diag").not_implemented("dtype not supported");
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(out_shape), dt, device,
                                        false);
}

// ----------------------------------------------------------------------------
// `_like` family — shape/dtype/device-matched creation
// ----------------------------------------------------------------------------
TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "zeros_like.a").non_null();
    return zeros_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "ones_like.a").non_null();
    return ones_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "empty_like.a").non_null();
    return empty_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr full_like_op(const TensorImplPtr& a, double fill_value, bool requires_grad) {
    Validator::input(a, "full_like.a").non_null();
    return full_op(a->shape(), fill_value, a->dtype(), a->device(), requires_grad);
}

}  // namespace lucid
