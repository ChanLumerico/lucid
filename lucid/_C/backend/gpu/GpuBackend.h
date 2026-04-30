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

    // ---- Additional unary (Phase 4.5) ---------------------------------

    Storage log2(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::log2(x); });
    }

    Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::reciprocal(x); });
    }

    Storage square(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::square(x); });
    }

    Storage cube(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt,
                         [](auto& x) { return ::mlx::core::multiply(::mlx::core::square(x), x); });
    }

    Storage cube_root(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array exponent(1.0 / 3.0, gpu::to_mlx_dtype(dt));
            return ::mlx::core::multiply(::mlx::core::sign(x),
                                         ::mlx::core::power(::mlx::core::abs(x), exponent));
        });
    }

    Storage tan(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::tan(x); });
    }

    Storage asin(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arcsin(x); });
    }

    Storage acos(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arccos(x); });
    }

    Storage atan(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arctan(x); });
    }

    Storage sinh(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sinh(x); });
    }

    Storage cosh(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::cosh(x); });
    }

    Storage invert(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) -> ::mlx::core::array {
            if (dt == Dtype::Bool)
                return ::mlx::core::logical_not(x);
            return ::mlx::core::bitwise_invert(x);
        });
    }

    Storage silu(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt,
                         [](auto& x) { return ::mlx::core::multiply(x, ::mlx::core::sigmoid(x)); });
    }

    Storage gelu(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            const double k0 = std::sqrt(2.0 / M_PI);
            ::mlx::core::array half(0.5, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array a044(0.044715, gpu::to_mlx_dtype(dt));
            ::mlx::core::array kk(k0, gpu::to_mlx_dtype(dt));
            auto x3 = ::mlx::core::multiply(::mlx::core::square(x), x);
            auto inner = ::mlx::core::multiply(
                kk, ::mlx::core::add(x, ::mlx::core::multiply(a044, x3)));
            auto t = ::mlx::core::tanh(inner);
            return ::mlx::core::multiply(::mlx::core::multiply(half, x),
                                         ::mlx::core::add(one, t));
        });
    }

    Storage softplus(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto log1p = ::mlx::core::log1p(::mlx::core::exp(neg_abs));
            return ::mlx::core::add(pos, log1p);
        });
    }

    Storage selu(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            constexpr double kS = 1.0507009873554805;
            constexpr double kA = 1.6732632423543772;
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array s(kS, gpu::to_mlx_dtype(dt));
            ::mlx::core::array sa(kS * kA, gpu::to_mlx_dtype(dt));
            auto pos_branch = ::mlx::core::multiply(s, x);
            auto neg_branch = ::mlx::core::multiply(
                sa, ::mlx::core::subtract(::mlx::core::exp(x), one));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            return ::mlx::core::where(pos_mask, pos_branch, neg_branch);
        });
    }

    Storage mish(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
            return ::mlx::core::multiply(x, ::mlx::core::tanh(sp));
        });
    }

    Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
            return ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
        });
    }

    Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
            auto h = ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
            return ::mlx::core::multiply(x, h);
        });
    }

    Storage relu6(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [dt](auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            return ::mlx::core::minimum(::mlx::core::maximum(x, zero), six);
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

    Storage variance(const Storage& a,
                     const Shape& /*in_shape*/,
                     const ReduceOpts& opts,
                     Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::var(*gs.arr, opts.axes, opts.keepdims, /*ddof=*/0);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
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

    Storage cumsum(const Storage& a, const Shape& /*shape*/, int axis, Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::cumsum(*gs.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage cumprod(const Storage& a, const Shape& /*shape*/, int axis, Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::cumprod(*gs.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage softmax(const Storage& a, const Shape& /*shape*/, int axis, Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::softmax(*gs.arr, axis, /*precise=*/true);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage softmax_backward(const Storage& z,
                             const Storage& grad_out,
                             const Shape& /*shape*/,
                             int axis,
                             Dtype dt) override {
        const auto& z_gpu = std::get<GpuStorage>(z);
        const auto& g_gpu = std::get<GpuStorage>(grad_out);
        auto gz = ::mlx::core::multiply(*g_gpu.arr, *z_gpu.arr);
        auto sum_gz = ::mlx::core::sum(gz, std::vector<int>{axis}, /*keepdims=*/true);
        auto diff = ::mlx::core::subtract(*g_gpu.arr, sum_gz);
        auto result = ::mlx::core::multiply(*z_gpu.arr, diff);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage reverse_along_axis(const Storage& a,
                               const Shape& shape,
                               int axis,
                               Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        std::vector<std::int32_t> idx(shape[static_cast<std::size_t>(axis)]);
        for (std::int64_t i = 0; i < shape[static_cast<std::size_t>(axis)]; ++i)
            idx[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(shape[static_cast<std::size_t>(axis)] - 1 - i);
        ::mlx::core::Shape idx_shape(shape.size(), 1);
        idx_shape[static_cast<std::size_t>(axis)] = shape[static_cast<std::size_t>(axis)];
        ::mlx::core::array idx_arr(idx.data(), idx_shape, ::mlx::core::int32);
        idx_arr = ::mlx::core::broadcast_to(idx_arr, gpu::to_mlx_shape(shape));
        auto result = ::mlx::core::take_along_axis(*gs.arr, idx_arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage trace(const Storage& a, const Shape& /*shape*/, Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::trace(*gs.arr, /*offset=*/0, /*axis1=*/0, /*axis2=*/1);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) override {
        const auto& gg = std::get<GpuStorage>(grad_out);
        const std::int64_t M = input_shape[0];
        const std::int64_t N = input_shape[1];
        auto eye = ::mlx::core::eye(static_cast<int>(M), static_cast<int>(N), 0,
                                    gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::multiply(eye, *gg.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                  const Shape& /*out_shape*/,
                                  Dtype dt,
                                  bool indexing_xy) override {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (const auto& x : xs)
            arrays.push_back(*std::get<GpuStorage>(x).arr);
        auto out = ::mlx::core::meshgrid(arrays, false, indexing_xy ? "xy" : "ij");
        std::vector<Storage> result;
        result.reserve(out.size());
        for (auto& arr : out) {
            auto contiguous = ::mlx::core::contiguous(arr);
            result.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
        }
        return result;
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

    Storage repeat(const Storage& a,
                   const Shape& /*shape*/,
                   Dtype dt,
                   std::int64_t repeats,
                   int axis) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::repeat(*gs.arr, static_cast<int>(repeats), axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage tile(const Storage& a,
                 const Shape& /*shape*/,
                 Dtype dt,
                 const std::vector<std::int64_t>& reps) override {
        const auto& gs = std::get<GpuStorage>(a);
        std::vector<int> reps_int(reps.begin(), reps.end());
        auto result = ::mlx::core::tile(*gs.arr, std::move(reps_int));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage permute(const Storage& a,
                    const Shape& /*shape*/,
                    const std::vector<int>& perm,
                    Dtype dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::transpose(*gs.arr, perm);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage pad(const Storage& a,
                const Shape& /*shape*/,
                Dtype dt,
                const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                double constant) override {
        const auto& gs = std::get<GpuStorage>(a);
        std::vector<std::pair<int, int>> mlx_pad;
        mlx_pad.reserve(pad_width.size());
        for (const auto& [lo, hi] : pad_width)
            mlx_pad.emplace_back(static_cast<int>(lo), static_cast<int>(hi));
        ::mlx::core::array pad_value(static_cast<float>(constant), gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::pad(*gs.arr, mlx_pad, pad_value);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage pow_scalar(const Storage& a, const Shape& /*shape*/, Dtype dt, double exp) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::power(*gs.arr, gpu::mlx_scalar(exp, gpu::to_mlx_dtype(dt)));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage rpow_scalar(const Storage& a, const Shape& /*shape*/, Dtype dt, double base) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = ::mlx::core::power(gpu::mlx_scalar(base, gpu::to_mlx_dtype(dt)), *gs.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage clip(const Storage& a,
                 const Shape& /*shape*/,
                 Dtype dt,
                 double min_v,
                 double max_v) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto lo = gpu::mlx_scalar(min_v, gpu::to_mlx_dtype(dt));
        auto hi = gpu::mlx_scalar(max_v, gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::clip(*gs.arr, lo, hi);
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
