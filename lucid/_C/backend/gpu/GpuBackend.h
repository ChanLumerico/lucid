#pragma once

// =====================================================================
// Lucid C++ engine — GpuBackend: IBackend for GPU via MLX.
// =====================================================================
//
// Phase 4: implements IBackend using MLX (Metal/CPU-backed on Apple).
// Registered with Dispatcher at static-init time via g_gpu_registrar.
//
// Layer: backend/gpu/. Depends on backend/ and core/ only.

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "MlxBridge.h"

namespace lucid {
namespace backend {

class GpuBackend final : public IBackend {
public:
    static void register_self() {
        Dispatcher::register_backend(Device::GPU, std::make_unique<GpuBackend>());
    }

    Device device() const noexcept override { return Device::GPU; }

    // ---- Memory -------------------------------------------------------

    Storage zeros(const Shape& shape, Dtype dt) override {
        auto arr = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(arr), dt)};
    }

    Storage ones(const Shape& shape, Dtype dt) override {
        auto arr = ::mlx::core::ones(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(arr), dt)};
    }

    Storage clone(const Storage& src, const Shape& /*shape*/, Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(src);
        auto arr = ::mlx::core::contiguous(*gs.arr);
        return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
    }

    // ---- Elementwise binary -------------------------------------------

    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::add(x, y); });
    }

    Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::subtract(x, y); });
    }

    Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::multiply(x, y); });
    }

    Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::divide(x, y); });
    }

    Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::power(x, y); });
    }

    Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::maximum(x, y); });
    }

    Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt,
                          [](auto& x, auto& y) { return ::mlx::core::minimum(x, y); });
    }

    // ---- Elementwise unary --------------------------------------------

    Storage exp(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::exp(x); });
    }

    Storage log(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::log(x); });
    }

    Storage sqrt(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sqrt(x); });
    }

    Storage rsqrt(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::rsqrt(x); });
    }

    Storage abs(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::abs(x); });
    }

    Storage neg(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::negative(x); });
    }

    Storage sign(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sign(x); });
    }

    Storage floor(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::floor(x); });
    }

    Storage ceil(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::ceil(x); });
    }

    Storage round(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::round(x); });
    }

    Storage sin(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sin(x); });
    }

    Storage cos(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::cos(x); });
    }

    Storage tanh(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::tanh(x); });
    }

    Storage sigmoid(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sigmoid(x); });
    }

    Storage relu(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) {
            return ::mlx::core::maximum(x, ::mlx::core::zeros_like(x));
        });
    }

    // ---- Reduction ----------------------------------------------------

    Storage reduce_sum(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
            return ::mlx::core::sum(x, axes, keepdims);
        });
    }

    Storage reduce_mean(const Storage& a,
                        const Shape& in_shape,
                        const ReduceOpts& opts,
                        Dtype dt) override {
        return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
            return ::mlx::core::mean(x, axes, keepdims);
        });
    }

    Storage reduce_max(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
            return ::mlx::core::max(x, axes, keepdims);
        });
    }

    Storage reduce_min(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
            return ::mlx::core::min(x, axes, keepdims);
        });
    }

    // ---- Linear algebra -----------------------------------------------

    Storage matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        ::mlx::core::array a_arr = *ga.arr;
        ::mlx::core::array b_arr = *gb.arr;
        if (opts.transA)
            a_arr = ::mlx::core::transpose(a_arr);
        if (opts.transB)
            b_arr = ::mlx::core::transpose(b_arr);
        auto result = ::mlx::core::matmul(a_arr, b_arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    // ---- Broadcast / cast -------------------------------------------

    Storage broadcast(const Storage& a,
                      const Shape& src_shape,
                      const Shape& dst_shape,
                      Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::broadcast_to(*gs.arr, gpu::to_mlx_shape(dst_shape));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage cast(const Storage& a,
                 const Shape& /*shape*/,
                 Dtype /*src_dt*/,
                 Dtype dst_dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::astype(*gs.arr, gpu::to_mlx_dtype(dst_dt));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dst_dt)};
    }

private:
    // ---- Helpers -------------------------------------------------------

    template <class Fn>
    Storage mlx_unary(const Storage& a, const Shape& /*shape*/, Dtype dt, Fn fn) {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = fn(*gs.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    template <class Fn>
    Storage mlx_binary(
        const Storage& a, const Storage& b, const Shape& /*shape*/, Dtype dt, Fn fn) {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        auto result = fn(*ga.arr, *gb.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    template <class Fn>
    Storage mlx_reduce(
        const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, Fn fn) {
        const auto& gs = std::get<GpuStorage>(a);
        std::vector<int> axes = opts.axes;
        if (axes.empty()) {
            for (int i = 0; i < static_cast<int>(in_shape.size()); ++i)
                axes.push_back(i);
        }
        auto result = fn(*gs.arr, axes, opts.keepdims);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }
};

}  // namespace backend
}  // namespace lucid

// ---------------------------------------------------------------------------
// Auto-registration at static init.
// ---------------------------------------------------------------------------
namespace {
struct GpuBackendRegistrar {
    GpuBackendRegistrar() {
        lucid::backend::Dispatcher::register_backend(
            lucid::Device::GPU, std::make_unique<lucid::backend::GpuBackend>());
    }
} g_gpu_registrar;
}  // namespace
