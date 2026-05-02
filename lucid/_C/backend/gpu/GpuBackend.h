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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <mlx/array.h>
#include <mlx/linalg.h>
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

    Storage contiguous(const Storage& src,
                       const Shape& shape,
                       const Stride& /*stride*/,
                       std::size_t /*storage_offset*/,
                       bool /*already_contiguous*/,
                       Dtype dt) override {
        return clone(src, shape, dt);
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

    Storage bitwise_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override {
        return mlx_binary(a, b, shape, dt, [op](auto& x, auto& y) {
            if (op == 0)
                return ::mlx::core::bitwise_and(x, y);
            if (op == 1)
                return ::mlx::core::bitwise_or(x, y);
            if (op == 2)
                return ::mlx::core::bitwise_xor(x, y);
            ErrorBuilder("gpu_backend::bitwise_binary").fail("unknown op");
        });
    }

    Storage compare_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype /*dt*/, int op) override {
        return mlx_binary(a, b, shape, Dtype::Bool, [op](auto& x, auto& y) {
            if (op == 0)
                return ::mlx::core::equal(x, y);
            if (op == 1)
                return ::mlx::core::not_equal(x, y);
            if (op == 2)
                return ::mlx::core::greater(x, y);
            if (op == 3)
                return ::mlx::core::greater_equal(x, y);
            if (op == 4)
                return ::mlx::core::less(x, y);
            if (op == 5)
                return ::mlx::core::less_equal(x, y);
            ErrorBuilder("gpu_backend::compare_binary").fail("unknown op");
        });
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
            auto inner =
                ::mlx::core::multiply(kk, ::mlx::core::add(x, ::mlx::core::multiply(a044, x3)));
            auto t = ::mlx::core::tanh(inner);
            return ::mlx::core::multiply(::mlx::core::multiply(half, x), ::mlx::core::add(one, t));
        });
    }

    Storage gelu_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
            constexpr double kC1 = 0.7978845608028654;
            constexpr double kC2 = 0.044715;
            ::mlx::core::array c1(kC1, gpu::to_mlx_dtype(dt));
            ::mlx::core::array c2(kC2, gpu::to_mlx_dtype(dt));
            ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array half(0.5, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto x2 = ::mlx::core::multiply(x, x);
            auto x3 = ::mlx::core::multiply(x2, x);
            auto inner =
                ::mlx::core::multiply(c1, ::mlx::core::add(x, ::mlx::core::multiply(c2, x3)));
            auto t = ::mlx::core::tanh(inner);
            auto dinner = ::mlx::core::multiply(
                c1,
                ::mlx::core::add(one, ::mlx::core::multiply(three, ::mlx::core::multiply(c2, x2))));
            auto t2 = ::mlx::core::multiply(t, t);
            auto term1 = ::mlx::core::multiply(half, ::mlx::core::add(one, t));
            auto term2 = ::mlx::core::multiply(
                half, ::mlx::core::multiply(
                          x, ::mlx::core::multiply(::mlx::core::subtract(one, t2), dinner)));
            auto dx = ::mlx::core::add(term1, term2);
            return ::mlx::core::multiply(dx, g);
        });
    }

    Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) override {
        return mlx_unary(a, shape, dt, [dt, slope](auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array slope_arr(slope, gpu::to_mlx_dtype(dt));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            auto neg_branch = ::mlx::core::multiply(slope_arr, x);
            return ::mlx::core::where(pos_mask, x, neg_branch);
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

    Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) override {
        return mlx_unary(a, shape, dt, [dt, alpha](auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array alpha_arr(alpha, gpu::to_mlx_dtype(dt));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            auto neg =
                ::mlx::core::multiply(alpha_arr, ::mlx::core::subtract(::mlx::core::exp(x), one));
            return ::mlx::core::where(pos_mask, x, neg);
        });
    }

    Storage elu_backward(const Storage& a,
                         const Storage& grad,
                         const Shape& shape,
                         Dtype dt,
                         double alpha) override {
        return mlx_binary(a, grad, shape, dt, [dt, alpha](auto& x, auto& g) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array alpha_arr(alpha, gpu::to_mlx_dtype(dt));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            auto neg_branch = ::mlx::core::multiply(alpha_arr, ::mlx::core::exp(x));
            auto ones_arr = ::mlx::core::ones_like(x);
            auto deriv = ::mlx::core::where(pos_mask, ones_arr, neg_branch);
            return ::mlx::core::multiply(deriv, g);
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
            auto neg_branch =
                ::mlx::core::multiply(sa, ::mlx::core::subtract(::mlx::core::exp(x), one));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            return ::mlx::core::where(pos_mask, pos_branch, neg_branch);
        });
    }

    Storage selu_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
            constexpr double kS = 1.0507009873554805;
            constexpr double kA = 1.6732632423543772;
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array s_arr(kS, gpu::to_mlx_dtype(dt));
            ::mlx::core::array sa_arr(kS * kA, gpu::to_mlx_dtype(dt));
            auto pos_mask = ::mlx::core::greater_equal(x, zero);
            auto pos_branch = ::mlx::core::broadcast_to(s_arr, x.shape());
            auto neg_branch = ::mlx::core::multiply(sa_arr, ::mlx::core::exp(x));
            auto deriv = ::mlx::core::where(pos_mask, pos_branch, neg_branch);
            return ::mlx::core::multiply(deriv, g);
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

    Storage mish_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
            auto t = ::mlx::core::tanh(sp);
            auto sig = ::mlx::core::sigmoid(x);
            auto one_minus_t2 = ::mlx::core::subtract(one, ::mlx::core::square(t));
            auto deriv = ::mlx::core::add(
                t, ::mlx::core::multiply(x, ::mlx::core::multiply(one_minus_t2, sig)));
            return ::mlx::core::multiply(deriv, g);
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

    Storage hard_sigmoid_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) override {
        return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
            ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array s(1.0 / 6.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            auto in_range =
                ::mlx::core::logical_and(::mlx::core::greater(x, m3), ::mlx::core::less(x, p3));
            auto s_b = ::mlx::core::broadcast_to(s, x.shape());
            auto z_b = ::mlx::core::broadcast_to(zero, x.shape());
            auto deriv = ::mlx::core::where(in_range, s_b, z_b);
            return ::mlx::core::multiply(deriv, g);
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

    Storage hard_swish_backward(const Storage& a,
                                const Storage& grad,
                                const Shape& shape,
                                Dtype dt) override {
        return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
            ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array half(0.5, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array third(1.0 / 3.0, gpu::to_mlx_dtype(dt));
            auto mid_branch = ::mlx::core::add(::mlx::core::multiply(x, third), half);
            auto le_m3 = ::mlx::core::less_equal(x, m3);
            auto ge_p3 = ::mlx::core::greater_equal(x, p3);
            auto z_b = ::mlx::core::broadcast_to(zero, x.shape());
            auto o_b = ::mlx::core::broadcast_to(one, x.shape());
            auto step1 = ::mlx::core::where(ge_p3, o_b, mid_branch);
            auto deriv = ::mlx::core::where(le_m3, z_b, step1);
            return ::mlx::core::multiply(deriv, g);
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

    Storage reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
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
        auto eye =
            ::mlx::core::eye(static_cast<int>(M), static_cast<int>(N), 0, gpu::to_mlx_dtype(dt));
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

    Storage where_branch(const Storage& grad,
                         const Storage& cond,
                         const Shape& shape,
                         Dtype dt,
                         bool true_branch) override {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gc = std::get<GpuStorage>(cond);
        auto zero = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
        auto result = true_branch ? ::mlx::core::where(*gc.arr, *gg.arr, zero)
                                  : ::mlx::core::where(*gc.arr, zero, *gg.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage masked_fill(const Storage& a,
                        const Storage& mask,
                        const Shape& /*shape*/,
                        Dtype dt,
                        double value) override {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gm = std::get<GpuStorage>(mask);
        ::mlx::core::array v(static_cast<float>(value), gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::where(*gm.arr, v, *ga.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage gather(const Storage& a,
                   const Storage& indices,
                   const Shape& /*input_shape*/,
                   const Shape& /*output_shape*/,
                   int axis,
                   Dtype /*index_dtype*/,
                   Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gi = std::get<GpuStorage>(indices);
        auto result = ::mlx::core::take_along_axis(*ga.arr, *gi.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage gather_backward(const Storage& grad,
                            const Storage& indices,
                            const Shape& input_shape,
                            const Shape& /*output_shape*/,
                            int axis,
                            Dtype /*index_dtype*/,
                            Dtype dt) override {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gi = std::get<GpuStorage>(indices);
        auto idx = *gi.arr;
        auto axis_len = ::mlx::core::array(
            static_cast<std::int32_t>(input_shape[static_cast<std::size_t>(axis)]), idx.dtype());
        auto zero = ::mlx::core::array(static_cast<std::int32_t>(0), idx.dtype());
        auto fixed =
            ::mlx::core::where(::mlx::core::less(idx, zero), ::mlx::core::add(idx, axis_len), idx);
        auto base = ::mlx::core::zeros(gpu::to_mlx_shape(input_shape), gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::scatter_add_axis(base, fixed, *gg.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage diagonal(const Storage& a,
                     const Shape& /*input_shape*/,
                     int offset,
                     int axis1,
                     int axis2,
                     Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto result = ::mlx::core::contiguous(::mlx::core::diagonal(*ga.arr, offset, axis1, axis2));
        return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
    }

    Storage diagonal_backward(const Storage& grad,
                              const Shape& input_shape,
                              const Shape& output_shape,
                              int offset,
                              int axis1,
                              int axis2,
                              Dtype dt) override {
        const auto& gg = std::get<GpuStorage>(grad);
        const std::size_t ndim = input_shape.size();
        const int a1n = axis1 < 0 ? axis1 + static_cast<int>(ndim) : axis1;
        const int a2n = axis2 < 0 ? axis2 + static_cast<int>(ndim) : axis2;
        const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
        const std::int64_t c0 = (offset >= 0) ? offset : 0;
        const std::int64_t L = output_shape.empty() ? 0 : output_shape.back();

        ::mlx::core::Shape mlx_in_shape = gpu::to_mlx_shape(input_shape);
        ::mlx::core::Shape mlx_out_shape = gpu::to_mlx_shape(output_shape);
        auto base = ::mlx::core::zeros(mlx_in_shape, gpu::to_mlx_dtype(dt));

        auto build_index = [&](int axis_in_input, std::int64_t start) {
            int out_axis;
            if (axis_in_input == a1n || axis_in_input == a2n) {
                out_axis = static_cast<int>(output_shape.size()) - 1;
            } else {
                int rel = 0;
                for (int d = 0; d < axis_in_input; ++d)
                    if (d != a1n && d != a2n)
                        ++rel;
                out_axis = rel;
            }
            const std::int64_t span = (axis_in_input == a1n || axis_in_input == a2n)
                                          ? L
                                          : input_shape[static_cast<std::size_t>(axis_in_input)];
            auto arr = ::mlx::core::arange(static_cast<int>(start), static_cast<int>(start + span),
                                           ::mlx::core::int32);
            ::mlx::core::Shape bc(output_shape.size(), 1);
            bc[static_cast<std::size_t>(out_axis)] = static_cast<int>(span);
            arr = ::mlx::core::reshape(arr, bc);
            return ::mlx::core::broadcast_to(arr, mlx_out_shape);
        };

        std::vector<::mlx::core::array> indices;
        std::vector<int> axes_v;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::int64_t start = 0;
            if (static_cast<int>(d) == a1n)
                start = r0;
            else if (static_cast<int>(d) == a2n)
                start = c0;
            indices.push_back(build_index(static_cast<int>(d), start));
            axes_v.push_back(static_cast<int>(d));
        }
        ::mlx::core::Shape upd_shape = mlx_out_shape;
        for (std::size_t d = 0; d < ndim; ++d)
            upd_shape.push_back(1);
        auto updates = ::mlx::core::reshape(*gg.arr, upd_shape);
        auto result = ::mlx::core::scatter_add(base, indices, updates, axes_v);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage roll(const Storage& a,
                 const Shape& /*shape*/,
                 Dtype dt,
                 const std::vector<std::int64_t>& shifts,
                 const std::vector<int>& axes) override {
        const auto& ga = std::get<GpuStorage>(a);
        ::mlx::core::Shape mshifts(shifts.begin(), shifts.end());
        auto result = ::mlx::core::roll(*ga.arr, mshifts, axes);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage reshape(const Storage& a,
                    const Shape& /*src_shape*/,
                    const Shape& dst_shape,
                    Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto result = ::mlx::core::reshape(*ga.arr, gpu::to_mlx_shape(dst_shape));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage slice_axis(const Storage& a,
                       const Shape& src_shape,
                       const Shape& slice_shape,
                       int axis,
                       std::int64_t offset,
                       Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        ::mlx::core::Shape lo(src_shape.size(), 0);
        ::mlx::core::Shape hi = gpu::to_mlx_shape(src_shape);
        lo[static_cast<std::size_t>(axis)] = static_cast<::mlx::core::ShapeElem>(offset);
        hi[static_cast<std::size_t>(axis)] = static_cast<::mlx::core::ShapeElem>(
            offset + slice_shape[static_cast<std::size_t>(axis)]);
        auto result = ::mlx::core::slice(*ga.arr, lo, hi);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage insert_axis_slice(const Storage& a,
                              const Shape& src_shape,
                              const Shape& dst_shape,
                              int axis,
                              std::int64_t offset,
                              Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        std::vector<std::pair<int, int>> pad;
        pad.reserve(dst_shape.size());
        for (std::size_t d = 0; d < dst_shape.size(); ++d) {
            if (static_cast<int>(d) == axis) {
                const auto before = static_cast<int>(offset);
                const auto after = static_cast<int>(dst_shape[d] - offset - src_shape[d]);
                pad.emplace_back(before, after);
            } else {
                pad.emplace_back(0, 0);
            }
        }
        ::mlx::core::array zero(static_cast<float>(0.0), gpu::to_mlx_dtype(dt));
        auto result = ::mlx::core::pad(*ga.arr, pad, zero);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage concatenate(const std::vector<Storage>& xs,
                        const std::vector<Shape>& /*shapes*/,
                        int axis,
                        Dtype dt) override {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (const auto& x : xs)
            arrays.push_back(*std::get<GpuStorage>(x).arr);
        auto result = ::mlx::core::concatenate(std::move(arrays), axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage stack(const std::vector<Storage>& xs,
                  const Shape& /*input_shape*/,
                  int axis,
                  Dtype dt) override {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (const auto& x : xs)
            arrays.push_back(*std::get<GpuStorage>(x).arr);
        auto result = ::mlx::core::stack(arrays, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    std::vector<Storage> split_equal(const Storage& a,
                                     const Shape& /*shape*/,
                                     int axis,
                                     std::int64_t num_splits,
                                     Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto pieces = ::mlx::core::split(*ga.arr, static_cast<int>(num_splits), axis);
        std::vector<Storage> out;
        out.reserve(pieces.size());
        for (auto& p : pieces) {
            auto contiguous = ::mlx::core::contiguous(p);
            out.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
        }
        return out;
    }

    std::vector<Storage> split_at(const Storage& a,
                                  const Shape& /*shape*/,
                                  int axis,
                                  const std::vector<std::int64_t>& indices,
                                  Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        ::mlx::core::Shape mlx_idx(indices.begin(), indices.end());
        auto pieces = ::mlx::core::split(*ga.arr, mlx_idx, axis);
        std::vector<Storage> out;
        out.reserve(pieces.size());
        for (auto& p : pieces) {
            auto contiguous = ::mlx::core::contiguous(p);
            out.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
        }
        return out;
    }

    Storage repeat_backward(const Storage& grad_out,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            int axis,
                            std::int64_t repeats,
                            Dtype dt) override {
        const auto& g = std::get<GpuStorage>(grad_out);
        ::mlx::core::Shape reshape_shape;
        reshape_shape.reserve(output_shape.size() + 1);
        for (std::size_t d = 0; d < output_shape.size(); ++d) {
            if (static_cast<int>(d) == axis) {
                reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(input_shape[d]));
                reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(repeats));
            } else {
                reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(output_shape[d]));
            }
        }
        auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
        auto summed = ::mlx::core::sum(reshaped, std::vector<int>{axis + 1}, /*keepdims=*/false);
        auto result = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shape));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage tile_backward(const Storage& grad_out,
                          const Shape& input_shape,
                          const Shape& padded_shape,
                          const Shape& /*output_shape*/,
                          const std::vector<std::int64_t>& reps,
                          Dtype dt) override {
        const auto& g = std::get<GpuStorage>(grad_out);
        ::mlx::core::Shape reshape_shape;
        reshape_shape.reserve(reps.size() * 2);
        std::vector<int> sum_axes;
        sum_axes.reserve(reps.size());
        for (std::size_t d = 0; d < reps.size(); ++d) {
            sum_axes.push_back(static_cast<int>(reshape_shape.size()));
            reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(reps[d]));
            reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(padded_shape[d]));
        }
        auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
        auto summed = sum_axes.empty() ? reshaped
                                       : ::mlx::core::sum(reshaped, sum_axes,
                                                          /*keepdims=*/false);
        auto result = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shape));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    std::pair<Storage, Storage> sort_select(const Storage& a,
                                            const Shape& /*input_shape*/,
                                            const Shape& output_shape,
                                            int axis,
                                            Dtype dt,
                                            bool descending) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto idx = ::mlx::core::argsort(*ga.arr, axis);
        if (descending)
            idx = take_descending_top_indices(idx, axis, output_shape);
        auto values = ::mlx::core::take_along_axis(*ga.arr, idx, axis);
        return {
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(values), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(idx), Dtype::I32)},
        };
    }

    Storage argsort(const Storage& a, const Shape& /*shape*/, int axis, Dtype /*dt*/) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto out = ::mlx::core::argsort(*ga.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), Dtype::I32)};
    }

    Storage arg_reduce_index(const Storage& a,
                             const Shape& /*shape*/,
                             int axis,
                             bool keepdims,
                             Dtype /*dt*/,
                             bool is_min) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto out = is_min ? ::mlx::core::argmin(*ga.arr, axis, keepdims)
                          : ::mlx::core::argmax(*ga.arr, axis, keepdims);
        out = ::mlx::core::astype(out, ::mlx::core::int64);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), Dtype::I64)};
    }

    Storage scatter_add_axis(const Storage& grad,
                             const Storage& indices,
                             const Shape& output_shape,
                             const Shape& /*grad_shape*/,
                             int axis,
                             Dtype dt) override {
        const auto& g = std::get<GpuStorage>(grad);
        const auto& idx = std::get<GpuStorage>(indices);
        auto base = ::mlx::core::zeros(gpu::to_mlx_shape(output_shape), gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::scatter_add_axis(base, *idx.arr, *g.arr, axis);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    // ---- Linear algebra -----------------------------------------------

    Storage matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        ::mlx::core::array a_arr = *ga.arr;
        ::mlx::core::array b_arr = *gb.arr;
        if (opts.transA)
            a_arr = ::mlx::core::swapaxes(a_arr, -2, -1);
        if (opts.transB)
            b_arr = ::mlx::core::swapaxes(b_arr, -2, -1);
        auto result = ::mlx::core::matmul(a_arr, b_arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage linear(const Storage& x,
                   const Storage& weight,
                   const Storage& bias,
                   const Shape& /*x_shape*/,
                   const Shape& /*weight_shape*/,
                   const Shape& out_shape,
                   Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gw = std::get<GpuStorage>(weight);
        const auto& gb = std::get<GpuStorage>(bias);
        if (shape_numel(out_shape) == 0) {
            auto out = ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), gpu::to_mlx_dtype(dt));
            return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
        }
        auto out = ::mlx::core::add(::mlx::core::matmul(*gx.arr, ::mlx::core::transpose(*gw.arr)),
                                    *gb.arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    std::vector<Storage> linear_backward(const Storage& grad,
                                         const Storage& x,
                                         const Storage& weight,
                                         const Shape& x_shape,
                                         const Shape& weight_shape,
                                         const Shape& bias_shape,
                                         Dtype dt) override {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gw = std::get<GpuStorage>(weight);
        const std::size_t M = linear_batch(x_shape);
        const std::size_t K = static_cast<std::size_t>(x_shape.back());
        const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
        if (M == 0 || N == 0 || K == 0) {
            auto dx = ::mlx::core::zeros(gpu::to_mlx_shape(x_shape), gpu::to_mlx_dtype(dt));
            auto dW = ::mlx::core::zeros(gpu::to_mlx_shape(weight_shape), gpu::to_mlx_dtype(dt));
            auto db = ::mlx::core::zeros(gpu::to_mlx_shape(bias_shape), gpu::to_mlx_dtype(dt));
            return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                    Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dW), dt)},
                    Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(db), dt)}};
        }
        ::mlx::core::Shape flat_x{static_cast<int>(M), static_cast<int>(K)};
        ::mlx::core::Shape flat_g{static_cast<int>(M), static_cast<int>(N)};
        auto g_2d = ::mlx::core::reshape(*gg.arr, flat_g);
        auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
        auto dx_flat = ::mlx::core::matmul(g_2d, *gw.arr);
        auto dx = ::mlx::core::reshape(dx_flat, gpu::to_mlx_shape(x_shape));
        auto dW = ::mlx::core::matmul(::mlx::core::transpose(g_2d), x_2d);
        auto db = ::mlx::core::sum(g_2d, std::vector<int>{0}, /*keepdims=*/false);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dW), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(db), dt)}};
    }

    StoragePair rms_norm_forward(const Storage& x,
                                 const Storage& gamma,
                                 std::size_t outer,
                                 std::size_t normalized_size,
                                 double eps,
                                 const Shape& x_shape,
                                 Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
        ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
        auto g_2d = ::mlx::core::reshape(*gg.arr, flat_g);
        auto ms = ::mlx::core::mean(::mlx::core::square(x_2d), std::vector<int>{1}, true);
        auto rstd =
            ::mlx::core::rsqrt(::mlx::core::add(ms, gpu::mlx_scalar(eps, gpu::to_mlx_dtype(dt))));
        auto y_2d = ::mlx::core::multiply(::mlx::core::multiply(x_2d, rstd), g_2d);
        auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x_shape));
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
    }

    StoragePair rms_norm_backward(const Storage& x,
                                  const Storage& gamma,
                                  const Storage& saved_rstd,
                                  const Storage& grad,
                                  std::size_t outer,
                                  std::size_t normalized_size,
                                  const Shape& x_shape,
                                  const Shape& gamma_shape,
                                  Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gr = std::get<GpuStorage>(saved_rstd);
        const auto& ggrad = std::get<GpuStorage>(grad);
        ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
        ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
        auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
        auto gamma_2d = ::mlx::core::reshape(*gg.arr, flat_g);
        auto xnorm = ::mlx::core::multiply(x_2d, *gr.arr);
        auto dgamma_2d = ::mlx::core::sum(::mlx::core::multiply(grad_2d, xnorm),
                                          std::vector<int>{0}, /*keepdims=*/false);
        auto gx_scaled = ::mlx::core::multiply(grad_2d, gamma_2d);
        auto m = ::mlx::core::mean(::mlx::core::multiply(gx_scaled, xnorm), std::vector<int>{1},
                                   /*keepdims=*/true);
        auto dx_2d = ::mlx::core::multiply(
            *gr.arr, ::mlx::core::subtract(gx_scaled, ::mlx::core::multiply(xnorm, m)));
        auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape));
        auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape));
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dgamma), dt)}};
    }

    std::vector<Storage> layer_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            std::size_t outer,
                                            std::size_t normalized_size,
                                            double eps,
                                            const Shape& x_shape,
                                            Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gb = std::get<GpuStorage>(beta);
        ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
        ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
        auto g_2d = ::mlx::core::reshape(*gg.arr, flat_g);
        auto b_2d = ::mlx::core::reshape(*gb.arr, flat_g);
        auto mean = ::mlx::core::mean(x_2d, std::vector<int>{1}, /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(x_2d, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), std::vector<int>{1},
                                     /*keepdims=*/true);
        auto rstd =
            ::mlx::core::rsqrt(::mlx::core::add(var, gpu::mlx_scalar(eps, gpu::to_mlx_dtype(dt))));
        auto xnorm = ::mlx::core::multiply(centered, rstd);
        auto y_2d = ::mlx::core::add(::mlx::core::multiply(xnorm, g_2d), b_2d);
        auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x_shape));
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(mean), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
    }

    std::vector<Storage> layer_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             std::size_t outer,
                                             std::size_t normalized_size,
                                             const Shape& x_shape,
                                             const Shape& gamma_shape,
                                             const Shape& beta_shape,
                                             Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gm = std::get<GpuStorage>(saved_mean);
        const auto& gr = std::get<GpuStorage>(saved_rstd);
        const auto& ggrad = std::get<GpuStorage>(grad);
        ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
        ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
        auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
        auto gamma_2d = ::mlx::core::reshape(*gg.arr, flat_g);
        auto centered = ::mlx::core::subtract(x_2d, *gm.arr);
        auto xnorm = ::mlx::core::multiply(centered, *gr.arr);
        auto dbeta_2d = ::mlx::core::sum(grad_2d, std::vector<int>{0}, /*keepdims=*/false);
        auto dgamma_2d = ::mlx::core::sum(::mlx::core::multiply(grad_2d, xnorm),
                                          std::vector<int>{0}, /*keepdims=*/false);
        auto gx_scaled = ::mlx::core::multiply(grad_2d, gamma_2d);
        auto mean1 = ::mlx::core::mean(gx_scaled, std::vector<int>{1}, /*keepdims=*/true);
        auto mean2 = ::mlx::core::mean(::mlx::core::multiply(gx_scaled, xnorm), std::vector<int>{1},
                                       /*keepdims=*/true);
        auto dx_2d = ::mlx::core::multiply(
            *gr.arr, ::mlx::core::subtract(::mlx::core::subtract(gx_scaled, mean1),
                                           ::mlx::core::multiply(xnorm, mean2)));
        auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape));
        auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape));
        auto dbeta = ::mlx::core::reshape(dbeta_2d, gpu::to_mlx_shape(beta_shape));
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dgamma), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dbeta), dt)}};
    }

    std::vector<Storage> batch_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int /*batch*/,
                                            int channels,
                                            int /*spatial*/,
                                            int ndim,
                                            double eps,
                                            const Shape& /*x_shape*/,
                                            Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gb = std::get<GpuStorage>(beta);
        ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
        br_c[1] = static_cast<::mlx::core::ShapeElem>(channels);
        auto g_view = ::mlx::core::reshape(*gg.arr, br_c);
        auto b_view = ::mlx::core::reshape(*gb.arr, br_c);
        std::vector<int> axes;
        axes.reserve(static_cast<std::size_t>(ndim) + 1);
        axes.push_back(0);
        for (int i = 0; i < ndim; ++i)
            axes.push_back(2 + i);

        auto mean = ::mlx::core::mean(*gx.arr, axes, /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(*gx.arr, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), axes, /*keepdims=*/true);
        auto rstd =
            ::mlx::core::rsqrt(::mlx::core::add(var, gpu::mlx_scalar(eps, gpu::to_mlx_dtype(dt))));
        auto xnorm = ::mlx::core::multiply(centered, rstd);
        auto y = ::mlx::core::add(::mlx::core::multiply(xnorm, g_view), b_view);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(mean), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
    }

    std::vector<Storage> batch_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int /*batch*/,
                                             int channels,
                                             int /*spatial*/,
                                             int ndim,
                                             const Shape& /*x_shape*/,
                                             Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gm = std::get<GpuStorage>(saved_mean);
        const auto& gr = std::get<GpuStorage>(saved_rstd);
        const auto& ggrad = std::get<GpuStorage>(grad);
        ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
        br_c[1] = static_cast<::mlx::core::ShapeElem>(channels);
        auto gamma_view = ::mlx::core::reshape(*gg.arr, br_c);
        auto centered = ::mlx::core::subtract(*gx.arr, *gm.arr);
        auto xnorm = ::mlx::core::multiply(centered, *gr.arr);
        std::vector<int> axes;
        axes.reserve(static_cast<std::size_t>(ndim) + 1);
        axes.push_back(0);
        for (int i = 0; i < ndim; ++i)
            axes.push_back(2 + i);
        auto dgamma =
            ::mlx::core::sum(::mlx::core::multiply(*ggrad.arr, xnorm), axes, /*keepdims=*/false);
        auto dbeta = ::mlx::core::sum(*ggrad.arr, axes, /*keepdims=*/false);
        auto mean_g = ::mlx::core::mean(*ggrad.arr, axes, /*keepdims=*/true);
        auto mean_g_xn =
            ::mlx::core::mean(::mlx::core::multiply(*ggrad.arr, xnorm), axes, /*keepdims=*/true);
        auto inner = ::mlx::core::subtract(::mlx::core::subtract(*ggrad.arr, mean_g),
                                           ::mlx::core::multiply(xnorm, mean_g_xn));
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(gamma_view, *gr.arr), inner);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dgamma), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dbeta), dt)}};
    }

    std::vector<Storage> group_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int /*spatial*/,
                                            int groups,
                                            const std::vector<int>& spatial_dims,
                                            double eps,
                                            const Shape& x_shape,
                                            Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gb = std::get<GpuStorage>(beta);
        const int ndim = static_cast<int>(spatial_dims.size());
        const int channels_per_group = channels / groups;
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape grouped;
        grouped.reserve(static_cast<std::size_t>(ndim) + 3);
        grouped.push_back(static_cast<SE>(batch));
        grouped.push_back(static_cast<SE>(groups));
        grouped.push_back(static_cast<SE>(channels_per_group));
        for (int s : spatial_dims)
            grouped.push_back(static_cast<SE>(s));
        auto x_g = ::mlx::core::reshape(*gx.arr, grouped);

        std::vector<int> reduce_axes;
        reduce_axes.reserve(static_cast<std::size_t>(ndim) + 1);
        reduce_axes.push_back(2);
        for (int i = 0; i < ndim; ++i)
            reduce_axes.push_back(3 + i);
        auto mean = ::mlx::core::mean(x_g, reduce_axes, /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(x_g, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), reduce_axes,
                                     /*keepdims=*/true);
        auto rstd =
            ::mlx::core::rsqrt(::mlx::core::add(var, gpu::mlx_scalar(eps, gpu::to_mlx_dtype(dt))));
        auto xnorm_g = ::mlx::core::multiply(centered, rstd);
        auto xnorm = ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(x_shape));
        ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
        br_c[1] = static_cast<SE>(channels);
        auto g_view = ::mlx::core::reshape(*gg.arr, br_c);
        auto b_view = ::mlx::core::reshape(*gb.arr, br_c);
        auto y = ::mlx::core::add(::mlx::core::multiply(xnorm, g_view), b_view);
        ::mlx::core::Shape mr{static_cast<SE>(batch), static_cast<SE>(groups)};
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(::mlx::core::reshape(mean, mr)),
                                            dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(::mlx::core::reshape(rstd, mr)),
                                            dt)}};
    }

    std::vector<Storage> group_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int /*spatial*/,
                                             int groups,
                                             const std::vector<int>& spatial_dims,
                                             const Shape& x_shape,
                                             Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gm = std::get<GpuStorage>(saved_mean);
        const auto& gr = std::get<GpuStorage>(saved_rstd);
        const auto& ggrad = std::get<GpuStorage>(grad);
        const int ndim = static_cast<int>(spatial_dims.size());
        const int channels_per_group = channels / groups;
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape grouped;
        grouped.reserve(static_cast<std::size_t>(ndim) + 3);
        grouped.push_back(static_cast<SE>(batch));
        grouped.push_back(static_cast<SE>(groups));
        grouped.push_back(static_cast<SE>(channels_per_group));
        for (int s : spatial_dims)
            grouped.push_back(static_cast<SE>(s));
        ::mlx::core::Shape mr_shape;
        mr_shape.reserve(static_cast<std::size_t>(ndim) + 3);
        mr_shape.push_back(static_cast<SE>(batch));
        mr_shape.push_back(static_cast<SE>(groups));
        for (int i = 0; i < ndim + 1; ++i)
            mr_shape.push_back(1);
        ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
        br_c[1] = static_cast<SE>(channels);

        auto x_g = ::mlx::core::reshape(*gx.arr, grouped);
        auto mean_g = ::mlx::core::reshape(*gm.arr, mr_shape);
        auto rstd_g = ::mlx::core::reshape(*gr.arr, mr_shape);
        auto centered = ::mlx::core::subtract(x_g, mean_g);
        auto xnorm_g = ::mlx::core::multiply(centered, rstd_g);
        auto xnorm = ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(x_shape));

        std::vector<int> ch_axes;
        ch_axes.reserve(static_cast<std::size_t>(ndim) + 1);
        ch_axes.push_back(0);
        for (int i = 0; i < ndim; ++i)
            ch_axes.push_back(2 + i);
        auto dgamma = ::mlx::core::sum(::mlx::core::multiply(*ggrad.arr, xnorm), ch_axes, false);
        auto dbeta = ::mlx::core::sum(*ggrad.arr, ch_axes, false);

        std::vector<int> red_axes;
        red_axes.reserve(static_cast<std::size_t>(ndim) + 1);
        red_axes.push_back(2);
        for (int i = 0; i < ndim; ++i)
            red_axes.push_back(3 + i);
        auto gamma_view = ::mlx::core::reshape(*gg.arr, br_c);
        auto gx_scaled4 = ::mlx::core::multiply(gamma_view, *ggrad.arr);
        auto gx_scaled_g = ::mlx::core::reshape(gx_scaled4, grouped);
        auto mean1 = ::mlx::core::mean(gx_scaled_g, red_axes, true);
        auto mean2 = ::mlx::core::mean(::mlx::core::multiply(gx_scaled_g, xnorm_g), red_axes, true);
        auto inner = ::mlx::core::subtract(::mlx::core::subtract(gx_scaled_g, mean1),
                                           ::mlx::core::multiply(xnorm_g, mean2));
        auto dx_g = ::mlx::core::multiply(rstd_g, inner);
        auto dx = ::mlx::core::reshape(dx_g, gpu::to_mlx_shape(x_shape));
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dgamma), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dbeta), dt)}};
    }

    Storage linalg_norm(const Storage& a,
                        const Shape& /*shape*/,
                        double ord,
                        const std::vector<int>& axes,
                        bool keepdims,
                        Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        std::optional<std::vector<int>> axis_opt;
        if (!axes.empty())
            axis_opt = axes;
        auto raw = ::mlx::core::linalg::norm(*ga.arr, ord, axis_opt, keepdims, k_linalg_stream);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(raw), dt)};
    }

    Storage linalg_cholesky(const Storage& a,
                            const Shape& /*shape*/,
                            bool upper,
                            Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto out = ::mlx::core::linalg::cholesky(*ga.arr, upper, k_linalg_stream);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    Storage linalg_inv(const Storage& a, const Shape& /*shape*/, Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto out = ::mlx::core::linalg::inv(*ga.arr, k_linalg_stream);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    Storage linalg_solve(const Storage& a,
                         const Storage& b,
                         const Shape& /*a_shape*/,
                         const Shape& /*b_shape*/,
                         Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        auto out = ::mlx::core::linalg::solve(*ga.arr, *gb.arr, k_linalg_stream);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    Storage linalg_matrix_power(const Storage& a,
                                const Shape& shape,
                                int power,
                                Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        if (power == 0) {
            auto eye = ::mlx::core::eye(n, n, 0, gpu::to_mlx_dtype(dt));
            if (shape.size() > 2) {
                eye = ::mlx::core::broadcast_to(eye, gpu::to_mlx_shape(shape));
            }
            return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(eye), dt)};
        }
        const int reps = std::abs(power);
        auto base = (power < 0) ? ::mlx::core::linalg::inv(*ga.arr, k_linalg_stream) : *ga.arr;
        ::mlx::core::array result = base;
        for (int i = 1; i < reps; ++i)
            result = ::mlx::core::matmul(result, base);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
    }

    Storage linalg_pinv(const Storage& a, const Shape& /*shape*/, Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto out = ::mlx::core::linalg::pinv(*ga.arr, k_linalg_stream);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        auto factors = ::mlx::core::linalg::lu(*ga.arr, k_linalg_stream);
        if (factors.size() < 3)
            ErrorBuilder("gpu_backend::linalg_det").fail("lu returned fewer than 3 factors");

        const auto& p = factors[0];
        const auto& u = factors[2];
        auto diag = ::mlx::core::diagonal(u, 0, -2, -1);
        auto det_u = ::mlx::core::prod(diag, /*keepdims=*/false);

        Shape out_shape(shape.begin(), shape.end() - 2);
        std::int64_t batch = 1;
        for (std::size_t i = 0; i + 2 < shape.size(); ++i)
            batch *= shape[i];

        auto p_eval = p;
        p_eval.eval();
        const std::size_t matrix_n = static_cast<std::size_t>(n);
        std::vector<float> signs(static_cast<std::size_t>(batch), 1.0f);
        const auto* p_data = p_eval.data<std::uint32_t>();
        for (std::int64_t b = 0; b < batch; ++b)
            signs[static_cast<std::size_t>(b)] = perm_index_sign(p_data + b * matrix_n, matrix_n);

        ::mlx::core::array sign_arr(signs.data(), gpu::to_mlx_shape(out_shape),
                                    ::mlx::core::float32);
        if (dt != Dtype::F32)
            sign_arr = ::mlx::core::astype(sign_arr, gpu::to_mlx_dtype(dt));
        auto det_a = ::mlx::core::multiply(det_u, sign_arr);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(det_a), dt)};
    }

    StoragePair linalg_qr(const Storage& a,
                          const Shape& /*shape*/,
                          const Shape& /*q_shape*/,
                          const Shape& /*r_shape*/,
                          Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto [q, r] = ::mlx::core::linalg::qr(*ga.arr, k_linalg_stream);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(q), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(r), dt)}};
    }

    StoragePair linalg_eig(const Storage& a,
                           const Shape& /*shape*/,
                           const Shape& /*values_shape*/,
                           const Shape& /*vectors_shape*/,
                           Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto [w, v] = ::mlx::core::linalg::eig(*ga.arr, k_linalg_stream);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(w), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(v), dt)}};
    }

    std::vector<Storage> linalg_svd(const Storage& a,
                                    const Shape& /*shape*/,
                                    bool compute_uv,
                                    const Shape& /*u_shape*/,
                                    const Shape& /*s_shape*/,
                                    const Shape& /*vt_shape*/,
                                    Dtype dt) override {
        const auto& ga = std::get<GpuStorage>(a);
        auto pieces = ::mlx::core::linalg::svd(*ga.arr, compute_uv, k_linalg_stream);
        std::vector<Storage> out;
        out.reserve(pieces.size());
        for (auto& p : pieces)
            out.push_back(Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(p), dt)});
        return out;
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

    Storage clip(
        const Storage& a, const Shape& /*shape*/, Dtype dt, double min_v, double max_v) override {
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

    Storage mse_loss(const Storage& input,
                     const Storage& target,
                     const Shape& /*shape*/,
                     Dtype dt,
                     int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        auto diff = ::mlx::core::subtract(*x.arr, *t.arr);
        auto sq = ::mlx::core::multiply(diff, diff);
        auto reduced = apply_loss_reduction(sq, reduction);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
    }

    std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
        if (reduction != 0)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
        auto diff = ::mlx::core::subtract(*x.arr, *t.arr);
        auto two = gpu::mlx_scalar(2.0, mlx_dt);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(two, diff), scaled);
        auto dtarget = ::mlx::core::negative(dx);
        return {
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dtarget), dt)},
        };
    }

    Storage huber_loss(const Storage& input,
                       const Storage& target,
                       const Shape& /*shape*/,
                       Dtype dt,
                       double delta,
                       int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto d = gpu::mlx_scalar(delta, mlx_dt);
        auto half_d_sq = gpu::mlx_scalar(0.5 * delta * delta, mlx_dt);
        auto half = gpu::mlx_scalar(0.5, mlx_dt);
        auto r = ::mlx::core::subtract(*x.arr, *t.arr);
        auto ar = ::mlx::core::abs(r);
        auto sq_term = ::mlx::core::multiply(half, ::mlx::core::multiply(r, r));
        auto lin_term = ::mlx::core::subtract(::mlx::core::multiply(d, ar), half_d_sq);
        auto cond = ::mlx::core::less(ar, d);
        auto values = ::mlx::core::where(cond, sq_term, lin_term);
        auto reduced = apply_loss_reduction(values, reduction);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
    }

    std::pair<Storage, Storage> huber_loss_backward(const Storage& input,
                                                    const Storage& target,
                                                    const Storage& grad,
                                                    const Shape& shape,
                                                    Dtype dt,
                                                    double delta,
                                                    int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto d = gpu::mlx_scalar(delta, mlx_dt);
        auto neg_d = gpu::mlx_scalar(-delta, mlx_dt);
        auto r = ::mlx::core::subtract(*x.arr, *t.arr);
        auto ar = ::mlx::core::abs(r);
        auto cond = ::mlx::core::less(ar, d);
        auto sgn_d =
            ::mlx::core::where(::mlx::core::greater(r, gpu::mlx_scalar(0.0, mlx_dt)), d, neg_d);
        auto dr = ::mlx::core::where(cond, r, sgn_d);
        auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
        if (reduction != 0)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
        auto dx = ::mlx::core::multiply(dr, scaled);
        auto dtarget = ::mlx::core::negative(dx);
        return {
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dtarget), dt)},
        };
    }

    Storage bce_loss(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Shape& /*shape*/,
                     Dtype dt,
                     double eps,
                     int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& w = std::get<GpuStorage>(weight);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto e_lo = gpu::mlx_scalar(eps, mlx_dt);
        auto one = gpu::mlx_scalar(1.0, mlx_dt);
        auto e_hi = ::mlx::core::subtract(one, e_lo);
        auto p = ::mlx::core::clip(*x.arr, std::optional<::mlx::core::array>(e_lo),
                                   std::optional<::mlx::core::array>(e_hi));
        auto one_mt = ::mlx::core::subtract(one, *t.arr);
        auto one_mp = ::mlx::core::subtract(one, p);
        auto term1 = ::mlx::core::multiply(*t.arr, ::mlx::core::log(p));
        auto term2 = ::mlx::core::multiply(one_mt, ::mlx::core::log(one_mp));
        auto values =
            ::mlx::core::multiply(*w.arr, ::mlx::core::negative(::mlx::core::add(term1, term2)));
        auto reduced = apply_loss_reduction(values, reduction);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
    }

    std::vector<Storage> bce_loss_backward(const Storage& input,
                                           const Storage& target,
                                           const Storage& weight,
                                           const Storage& grad,
                                           const Shape& shape,
                                           Dtype dt,
                                           double eps,
                                           int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& w = std::get<GpuStorage>(weight);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto e_lo = gpu::mlx_scalar(eps, mlx_dt);
        auto one = gpu::mlx_scalar(1.0, mlx_dt);
        auto e_hi = ::mlx::core::subtract(one, e_lo);
        auto p = ::mlx::core::clip(*x.arr, std::optional<::mlx::core::array>(e_lo),
                                   std::optional<::mlx::core::array>(e_hi));
        auto one_mp = ::mlx::core::subtract(one, p);
        auto one_mt = ::mlx::core::subtract(one, *t.arr);
        auto log_p = ::mlx::core::log(p);
        auto log_1mp = ::mlx::core::log(one_mp);
        auto dlp = ::mlx::core::add(::mlx::core::negative(::mlx::core::divide(*t.arr, p)),
                                    ::mlx::core::divide(one_mt, one_mp));
        auto dtarget_term = ::mlx::core::add(::mlx::core::negative(log_p), log_1mp);
        auto values = ::mlx::core::negative(::mlx::core::add(
            ::mlx::core::multiply(*t.arr, log_p), ::mlx::core::multiply(one_mt, log_1mp)));
        auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
        if (reduction != 0)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
        auto dx = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dlp, scaled));
        auto dtarget = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dtarget_term, scaled));
        auto dweight = ::mlx::core::multiply(values, scaled);
        return {
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dtarget), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dweight), dt)},
        };
    }

    Storage bce_with_logits_loss(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 const Storage& pos_weight,
                                 const Shape& /*shape*/,
                                 const Shape& /*weight_shape*/,
                                 const Shape& /*pos_weight_shape*/,
                                 Dtype dt,
                                 int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& w = std::get<GpuStorage>(weight);
        const auto& pw = std::get<GpuStorage>(pos_weight);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto one = gpu::mlx_scalar(1.0, mlx_dt);
        auto zero = gpu::mlx_scalar(0.0, mlx_dt);
        auto pw_m1 = ::mlx::core::subtract(*pw.arr, one);
        auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *t.arr), one);
        auto log1pexp =
            ::mlx::core::log1p(::mlx::core::exp(::mlx::core::negative(::mlx::core::abs(*x.arr))));
        auto max0 = ::mlx::core::maximum(*x.arr, zero);
        auto loss =
            ::mlx::core::add(::mlx::core::subtract(max0, ::mlx::core::multiply(*x.arr, *t.arr)),
                             ::mlx::core::multiply(log_weight, log1pexp));
        auto values = ::mlx::core::multiply(*w.arr, loss);
        auto reduced = apply_loss_reduction(values, reduction);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
    }

    std::vector<Storage> bce_with_logits_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& weight,
                                                  const Storage& pos_weight,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto& w = std::get<GpuStorage>(weight);
        const auto& pw = std::get<GpuStorage>(pos_weight);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto one = gpu::mlx_scalar(1.0, mlx_dt);
        auto zero = gpu::mlx_scalar(0.0, mlx_dt);
        auto pw_m1 = ::mlx::core::subtract(*pw.arr, one);
        auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *t.arr), one);
        auto sigm = ::mlx::core::sigmoid(*x.arr);
        auto log1pexp =
            ::mlx::core::log1p(::mlx::core::exp(::mlx::core::negative(::mlx::core::abs(*x.arr))));
        auto dlx = ::mlx::core::subtract(::mlx::core::multiply(log_weight, sigm), *t.arr);
        auto dtarget_term =
            ::mlx::core::add(::mlx::core::negative(*x.arr), ::mlx::core::multiply(pw_m1, log1pexp));
        auto loss = ::mlx::core::add(::mlx::core::subtract(::mlx::core::maximum(*x.arr, zero),
                                                           ::mlx::core::multiply(*x.arr, *t.arr)),
                                     ::mlx::core::multiply(log_weight, log1pexp));
        auto dpos_weight = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(*t.arr, log1pexp));
        auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
        if (reduction != 0)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
        auto dx = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dlx, scaled));
        auto dtarget = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dtarget_term, scaled));
        auto dweight = ::mlx::core::multiply(loss, scaled);
        auto dpos_weight_scaled = ::mlx::core::multiply(dpos_weight, scaled);
        return {
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dtarget), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dweight), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dpos_weight_scaled), dt)},
        };
    }

    ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                              const Storage& target,
                                              const Storage* weight,
                                              const Shape& /*input_shape*/,
                                              const Shape& target_shape,
                                              Dtype dt,
                                              double eps,
                                              int ignore_index,
                                              int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto softmax = ::mlx::core::softmax(*x.arr, std::vector<int>{1}, /*precise=*/true);
        auto t_idx = class_target_indices(*t.arr, target_shape);
        auto ig_mask = class_ignore_mask(t_idx, ignore_index);
        auto safe_t = safe_class_indices(t_idx, ig_mask);

        auto pred = ::mlx::core::take_along_axis(softmax, safe_t, 1);
        auto neg_log_pred = ::mlx::core::negative(
            ::mlx::core::log(::mlx::core::add(pred, gpu::mlx_scalar(eps, mlx_dt))));
        auto w_gather = class_weight_gather(weight, safe_t, neg_log_pred.shape(), mlx_dt);
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        auto loss =
            ::mlx::core::multiply(::mlx::core::multiply(w_gather, neg_log_pred), ig_mask_dt);
        auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});
        auto valid_count = class_valid_count(ig_mask_dt, mlx_dt);
        auto output = reduce_class_loss(loss_squeezed, valid_count, dt, reduction);
        return {std::move(output),
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(softmax), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(valid_count), dt)}};
    }

    Storage cross_entropy_backward(const Storage& saved_softmax,
                                   const Storage& target,
                                   const Storage* weight,
                                   const Storage& valid_count,
                                   const Storage& grad,
                                   const Shape& input_shape,
                                   Dtype dt,
                                   int ignore_index,
                                   int reduction) override {
        const auto& sm = std::get<GpuStorage>(saved_softmax);
        const auto& t = std::get<GpuStorage>(target);
        const auto& vc = std::get<GpuStorage>(valid_count);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto full_shape = gpu::to_mlx_shape(input_shape);
        auto t_shape = full_shape;
        t_shape[1] = 1;
        auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*t.arr, ::mlx::core::int64), t_shape);
        auto ig_mask = class_ignore_mask(t_idx, ignore_index);
        auto safe_t = safe_class_indices(t_idx, ig_mask);

        auto c_range = ::mlx::core::astype(
            ::mlx::core::arange(0, static_cast<int>(input_shape[1]), 1), ::mlx::core::int64);
        ::mlx::core::Shape c_shape(full_shape.size(), 1);
        c_shape[1] = static_cast<int>(input_shape[1]);
        auto c_idx = ::mlx::core::reshape(c_range, c_shape);
        auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
        auto base = ::mlx::core::subtract(*sm.arr, onehot);

        auto w_gather = class_weight_gather(weight, safe_t, t_shape, mlx_dt);
        w_gather = ::mlx::core::multiply(w_gather, ::mlx::core::astype(ig_mask, mlx_dt));
        auto w_full = ::mlx::core::broadcast_to(w_gather, full_shape);
        auto scaled = class_scaled_grad(*g.arr, *vc.arr, t_shape, reduction);
        auto scaled_full = ::mlx::core::broadcast_to(scaled, full_shape);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(base, w_full), scaled_full);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)};
    }

    ClassLossForwardResult nll_loss(const Storage& input,
                                    const Storage& target,
                                    const Storage* weight,
                                    const Shape& /*input_shape*/,
                                    const Shape& target_shape,
                                    Dtype dt,
                                    int ignore_index,
                                    int reduction) override {
        const auto& x = std::get<GpuStorage>(input);
        const auto& t = std::get<GpuStorage>(target);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto t_idx = class_target_indices(*t.arr, target_shape);
        auto ig_mask = class_ignore_mask(t_idx, ignore_index);
        auto safe_t = safe_class_indices(t_idx, ig_mask);
        auto pred = ::mlx::core::take_along_axis(*x.arr, safe_t, 1);
        auto neg = ::mlx::core::negative(pred);
        auto w_gather = class_weight_gather(weight, safe_t, neg.shape(), mlx_dt);
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        auto loss = ::mlx::core::multiply(::mlx::core::multiply(w_gather, neg), ig_mask_dt);
        auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});
        auto valid_count = class_valid_count(ig_mask_dt, mlx_dt);
        auto output = reduce_class_loss(loss_squeezed, valid_count, dt, reduction);
        return {std::move(output), Storage{},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(valid_count), dt)}};
    }

    Storage nll_loss_backward(const Storage& target,
                              const Storage* weight,
                              const Storage& valid_count,
                              const Storage& grad,
                              const Shape& input_shape,
                              Dtype dt,
                              int ignore_index,
                              int reduction) override {
        const auto& t = std::get<GpuStorage>(target);
        const auto& vc = std::get<GpuStorage>(valid_count);
        const auto& g = std::get<GpuStorage>(grad);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto full_shape = gpu::to_mlx_shape(input_shape);
        auto t_shape = full_shape;
        t_shape[1] = 1;
        auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*t.arr, ::mlx::core::int64), t_shape);
        auto ig_mask = class_ignore_mask(t_idx, ignore_index);
        auto safe_t = safe_class_indices(t_idx, ig_mask);

        auto c_range = ::mlx::core::astype(
            ::mlx::core::arange(0, static_cast<int>(input_shape[1]), 1), ::mlx::core::int64);
        ::mlx::core::Shape c_shape(full_shape.size(), 1);
        c_shape[1] = static_cast<int>(input_shape[1]);
        auto c_idx = ::mlx::core::reshape(c_range, c_shape);
        auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
        auto neg_onehot = ::mlx::core::negative(onehot);
        auto w_gather = class_weight_gather(weight, safe_t, t_shape, mlx_dt);
        w_gather = ::mlx::core::multiply(w_gather, ::mlx::core::astype(ig_mask, mlx_dt));
        auto w_full = ::mlx::core::broadcast_to(w_gather, full_shape);
        auto scaled = class_scaled_grad(*g.arr, *vc.arr, t_shape, reduction);
        auto scaled_full = ::mlx::core::broadcast_to(scaled, full_shape);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(neg_onehot, w_full), scaled_full);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)};
    }

    CpuStorage to_cpu(const Storage& a, const Shape& shape) override {
        return gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), shape);
    }

    // ---- Attention --------------------------------------------------------

    std::vector<Storage> sdpa_forward(
        const Storage& q,
        const Storage& k,
        const Storage& v,
        const Storage* attn_mask,
        const Shape& q_shape,
        const Shape& k_shape,
        const Shape& v_shape,
        Dtype mask_dtype,
        std::size_t mask_numel,
        double scale,
        bool is_causal,
        Dtype dt) override {
        (void)k_shape; (void)v_shape; (void)mask_numel;

        const auto& gQ = std::get<GpuStorage>(q);
        const auto& gK = std::get<GpuStorage>(k);
        const auto& gV = std::get<GpuStorage>(v);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);

        // Flatten leading dims to get Lq, Lk from shape args.
        const std::size_t Lq = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
        const std::size_t Lk = static_cast<std::size_t>(k_shape[k_shape.size() - 2]);
        (void)Lq; (void)Lk;

        auto k_t = ::mlx::core::swapaxes(*gK.arr, -2, -1);
        auto scores = ::mlx::core::matmul(*gQ.arr, k_t);
        auto scale_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(scale)), mlx_dt);
        scores = ::mlx::core::multiply(scores, scale_arr);

        auto neg_inf = ::mlx::core::astype(
            ::mlx::core::array(-std::numeric_limits<float>::infinity()), mlx_dt);
        if (attn_mask) {
            const auto& gM = std::get<GpuStorage>(*attn_mask);
            if (mask_dtype == Dtype::Bool) {
                scores = ::mlx::core::where(*gM.arr, neg_inf, scores);
            } else {
                scores = ::mlx::core::add(scores, *gM.arr);
            }
        }
        if (is_causal) {
            const int lq = static_cast<int>(q_shape[q_shape.size() - 2]);
            const int lk = static_cast<int>(k_shape[k_shape.size() - 2]);
            auto mask = ::mlx::core::triu(
                ::mlx::core::ones({lq, lk}, ::mlx::core::bool_), /*k=*/1);
            scores = ::mlx::core::where(mask, neg_inf, scores);
        }
        auto weights = ::mlx::core::softmax(scores, std::vector<int>{-1}, /*precise=*/true);
        auto output  = ::mlx::core::matmul(weights, *gV.arr);

        return {Storage{gpu::wrap_mlx_array(std::move(weights), dt)},
                Storage{gpu::wrap_mlx_array(std::move(output),  dt)}};
    }

    std::vector<Storage> sdpa_backward(
        const Storage& grad_out,
        const Storage& q,
        const Storage& k,
        const Storage& v,
        const Storage& saved_weights,
        const Shape& /*q_shape*/,
        const Shape& /*k_shape*/,
        const Shape& /*v_shape*/,
        double scale,
        Dtype dt) override {

        const auto& gQ = std::get<GpuStorage>(q);
        const auto& gK = std::get<GpuStorage>(k);
        const auto& gV = std::get<GpuStorage>(v);
        const auto& gW = std::get<GpuStorage>(saved_weights);
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);

        auto scale_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(scale)), mlx_dt);

        // dV = W^T @ dout
        auto W_t = ::mlx::core::swapaxes(*gW.arr, -2, -1);
        auto dV  = ::mlx::core::matmul(W_t, *gG.arr);

        // dweights = dout @ V^T
        auto V_t = ::mlx::core::swapaxes(*gV.arr, -2, -1);
        auto dW  = ::mlx::core::matmul(*gG.arr, V_t);

        // softmax backward
        auto wdw      = ::mlx::core::multiply(*gW.arr, dW);
        auto sum_wdw  = ::mlx::core::sum(wdw, std::vector<int>{-1}, /*keepdims=*/true);
        auto dscores  = ::mlx::core::multiply(*gW.arr, ::mlx::core::subtract(dW, sum_wdw));

        // dQ = scale * dscores @ K
        auto dQ = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores, *gK.arr));

        // dK = scale * dscores^T @ Q
        auto dscores_t = ::mlx::core::swapaxes(dscores, -2, -1);
        auto dK = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores_t, *gQ.arr));

        return {Storage{gpu::wrap_mlx_array(std::move(dQ), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dK), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dV), dt)}};
    }

    // ---- Transposed convolution ------------------------------------------

    Storage conv_transpose_nd_forward(
        const Storage& x,
        const Storage& W,
        const Storage& b,
        int /*B*/,
        int /*Cin*/,
        int Cout,
        const int* /*S*/,
        const int* /*K*/,
        const int* /*O*/,
        const int* stride,
        const int* pad,
        const int* opad,
        int N,
        const Shape& /*out_shape*/,
        Dtype dt) override {

        const auto& gx = std::get<GpuStorage>(x);
        const auto& gW = std::get<GpuStorage>(W);
        const auto& gb = std::get<GpuStorage>(b);

        auto x_nhwc = ::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N));
        auto W_nhwc = ::mlx::core::transpose(*gW.arr, gpu_w_to_mlx_transpose_perm(N));
        auto y_nhwc = gpu_mlx_conv_transpose(x_nhwc, W_nhwc, stride, pad, opad, N);

        ::mlx::core::Shape b_brd(N + 2, 1);
        b_brd[N + 1] = static_cast<::mlx::core::ShapeElem>(Cout);
        auto b_view = ::mlx::core::reshape(*gb.arr, b_brd);
        y_nhwc = ::mlx::core::add(y_nhwc, b_view);
        auto y = ::mlx::core::contiguous(::mlx::core::transpose(y_nhwc, gpu_nhwc_to_nchw_perm(N)));
        return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
    }

    std::vector<Storage> conv_transpose_nd_backward(
        const Storage& grad_out,
        const Storage& x,
        const Storage& W,
        int /*B*/,
        int Cin,
        int Cout,
        const int* S,
        const int* K,
        const int* /*O*/,
        const int* stride,
        const int* pad,
        int N,
        Dtype dt) override {

        const auto& gx = std::get<GpuStorage>(x);
        const auto& gW = std::get<GpuStorage>(W);
        const auto& gG = std::get<GpuStorage>(grad_out);

        // db = sum(grad, axes={0, 2..N+1})
        std::vector<int> db_axes;
        db_axes.reserve(N + 1);
        db_axes.push_back(0);
        for (int i = 0; i < N; ++i) db_axes.push_back(2 + i);
        auto db = ::mlx::core::sum(*gG.arr, db_axes, false);

        // dx = regular conv(grad, W_perm, stride, pad)
        std::vector<int> w_dx_perm;
        w_dx_perm.push_back(0);
        for (int i = 0; i < N; ++i) w_dx_perm.push_back(2 + i);
        w_dx_perm.push_back(1);
        auto grad_nhwc = ::mlx::core::transpose(*gG.arr, gpu_nchw_to_nhwc_perm(N));
        auto W_dx_nhwc = ::mlx::core::transpose(*gW.arr, w_dx_perm);
        auto dx_nhwc   = gpu_mlx_conv(grad_nhwc, W_dx_nhwc, stride, pad, N);
        auto dx = ::mlx::core::contiguous(::mlx::core::transpose(dx_nhwc, gpu_nhwc_to_nchw_perm(N)));

        // dW via conv_general
        std::vector<int> perm_axes;
        perm_axes.push_back(1);
        for (int i = 0; i < N; ++i) perm_axes.push_back(2 + i);
        perm_axes.push_back(0);
        auto g_perm = ::mlx::core::transpose(*gG.arr, perm_axes);
        auto x_perm = ::mlx::core::transpose(*gx.arr, perm_axes);
        std::vector<int> conv_stride(N, 1);
        std::vector<int> conv_pad(N), conv_kdil(N), conv_idil(N, 1);
        for (int i = 0; i < N; ++i) {
            conv_pad[i]  = pad[i];
            conv_kdil[i] = stride[i];
        }
        auto dW_perm = ::mlx::core::conv_general(g_perm, x_perm, conv_stride, conv_pad,
                                                 conv_pad, conv_kdil, conv_idil);
        // Crop to (Co, *K, Ci) and permute to (Cin, Cout, *K).
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.push_back(static_cast<SE>(Cout));
        for (int i = 0; i < N; ++i) crop_hi.push_back(static_cast<SE>(K[i]));
        crop_hi.push_back(static_cast<SE>(Cin));
        dW_perm = ::mlx::core::slice(dW_perm, crop_lo, crop_hi);
        std::vector<int> dW_back;
        dW_back.push_back(N + 1);
        dW_back.push_back(0);
        for (int i = 0; i < N; ++i) dW_back.push_back(1 + i);
        auto dW = ::mlx::core::contiguous(::mlx::core::transpose(dW_perm, dW_back));

        return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
                Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
    }

    // ---- Expand-and-multiply (dropout helpers) ----------------------------

    std::pair<Storage, Storage> expand_and_multiply(
        const Storage& mask, const Storage& x,
        const Shape& mask_shape, const Shape& x_shape, Dtype dt) override {
        (void)mask_shape;
        const auto& gm = std::get<GpuStorage>(mask);
        const auto& gx = std::get<GpuStorage>(x);
        auto full_mask = ::mlx::core::broadcast_to(*gm.arr, gpu::to_mlx_shape(x_shape));
        auto out = ::mlx::core::multiply(full_mask, *gx.arr);
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(full_mask), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)}};
    }

    /// DropBlock keep-mask: seed (flat x_shape) → scale*(1-dilated) mask.
    Storage drop_block_mask(
        const Storage& seed, double drop_prob, std::int64_t block_size,
        const Shape& x_shape, Dtype dt) override {
        const auto& sg = std::get<GpuStorage>(seed);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const int K = static_cast<int>(block_size);
        const int pad_sz = K / 2;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        const int H = static_cast<int>(x_shape[2]);
        const int W = static_cast<int>(x_shape[3]);
        auto seed_4d = ::mlx::core::reshape(*sg.arr, ::mlx::core::Shape{B, C, H, W});
        auto seed_pad = ::mlx::core::pad(
            seed_4d,
            std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {pad_sz, pad_sz}, {pad_sz, pad_sz}},
            ::mlx::core::array(0.0f, mlx_dt));
        ::mlx::core::array dilated = ::mlx::core::zeros({B, C, H, W}, mlx_dt);
        for (int dy = 0; dy < K; ++dy) {
            for (int dx = 0; dx < K; ++dx) {
                auto s = ::mlx::core::slice(seed_pad, {0, 0, dy, dx}, {B, C, dy + H, dx + W});
                dilated = (dy == 0 && dx == 0) ? s : ::mlx::core::maximum(dilated, s);
            }
        }
        auto scale_v = ::mlx::core::array(static_cast<float>(1.0 / (1.0 - drop_prob)), mlx_dt);
        auto keep = ::mlx::core::multiply(
            scale_v, ::mlx::core::subtract(::mlx::core::array(1.0f, mlx_dt), dilated));
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(keep), dt)};
    }

    // ---- Embedding -------------------------------------------------------

    Storage embedding_forward(const Storage& weight,
                              const Storage& indices,
                              const Shape& weight_shape,
                              const Shape& indices_shape,
                              const Shape& out_shape,
                              int padding_idx,
                              Dtype dt) override {
        (void)weight_shape; (void)indices_shape; (void)out_shape;
        const auto& gw = std::get<GpuStorage>(weight);
        const auto& gi = std::get<GpuStorage>(indices);
        auto idx = ::mlx::core::astype(*gi.arr, ::mlx::core::int64);
        auto out = ::mlx::core::take(*gw.arr, idx, /*axis=*/0);
        if (padding_idx >= 0) {
            auto pad_v = ::mlx::core::astype(::mlx::core::array(padding_idx), ::mlx::core::int64);
            auto mask = ::mlx::core::not_equal(idx, pad_v);
            auto mask_dt = ::mlx::core::astype(mask, gpu::to_mlx_dtype(dt));
            auto mask_shape_v = mask_dt.shape();
            mask_shape_v.push_back(1);
            mask_dt = ::mlx::core::reshape(mask_dt, mask_shape_v);
            out = ::mlx::core::multiply(out, mask_dt);
        }
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }

    Storage embedding_backward(const Storage& grad_out,
                               const Storage& indices,
                               const Shape& weight_shape,
                               const Shape& indices_shape,
                               int padding_idx,
                               Dtype dt) override {
        const std::int64_t N = weight_shape[0];
        const std::int64_t D = weight_shape[1];
        const std::size_t M = 1;
        std::size_t M_total = 1;
        for (auto d : indices_shape) M_total *= static_cast<std::size_t>(d);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto& gi = std::get<GpuStorage>(indices);
        auto idx_flat = ::mlx::core::reshape(
            ::mlx::core::astype(*gi.arr, ::mlx::core::int64),
            {static_cast<int>(M_total)});
        auto grad_flat = ::mlx::core::reshape(*gg.arr,
            {static_cast<int>(M_total), static_cast<int>(D)});
        if (padding_idx >= 0) {
            auto pad_v = ::mlx::core::astype(::mlx::core::array(padding_idx), ::mlx::core::int64);
            auto mask = ::mlx::core::not_equal(idx_flat, pad_v);
            auto mask_dt = ::mlx::core::astype(mask, mlx_dt);
            auto mask_b = ::mlx::core::reshape(mask_dt, {static_cast<int>(M_total), 1});
            grad_flat = ::mlx::core::multiply(grad_flat, mask_b);
        }
        auto idx_col = ::mlx::core::reshape(idx_flat, {static_cast<int>(M_total), 1});
        auto arange_n = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(N), 1), ::mlx::core::int64),
            {1, static_cast<int>(N)});
        auto onehot = ::mlx::core::astype(::mlx::core::equal(arange_n, idx_col), mlx_dt);
        auto onehot_t = ::mlx::core::transpose(onehot);
        auto dW = ::mlx::core::matmul(onehot_t, grad_flat);
        return Storage{gpu::wrap_mlx_array(std::move(dW), dt)};
        (void)M;
    }

    Storage sinusoidal_pos_embedding(std::int64_t seq_len,
                                     std::int64_t embed_dim,
                                     Dtype dt) override {
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const int L = static_cast<int>(seq_len);
        const int D = static_cast<int>(embed_dim);
        const int Dh = D / 2;
        Shape out_shape{seq_len, embed_dim};
        if (L * D == 0) {
            auto z = ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), mlx_dt);
            return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(z), dt)};
        }
        auto pos = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, L, 1), mlx_dt), {L, 1});
        auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), mlx_dt);
        const float inv_d = static_cast<float>(-std::log(10000.0) / static_cast<double>(D));
        auto two_inv_d = ::mlx::core::astype(::mlx::core::array(2.0f * inv_d), mlx_dt);
        auto theta = ::mlx::core::exp(::mlx::core::multiply(two_inv_d, k_arr));
        auto theta_row = ::mlx::core::reshape(theta, {1, Dh});
        auto angle = ::mlx::core::matmul(pos, theta_row);
        auto sin_t = ::mlx::core::sin(angle);
        auto cos_t = ::mlx::core::cos(angle);
        auto sin_e = ::mlx::core::expand_dims(sin_t, -1);
        auto cos_e = ::mlx::core::expand_dims(cos_t, -1);
        auto stacked = ::mlx::core::concatenate(
            std::vector<::mlx::core::array>{sin_e, cos_e}, -1);
        auto out = ::mlx::core::reshape(stacked, {L, 2 * Dh});
        if (D % 2 != 0) {
            auto pad_col = ::mlx::core::zeros({L, 1}, mlx_dt);
            out = ::mlx::core::concatenate(
                std::vector<::mlx::core::array>{out, pad_col}, -1);
        }
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
    }

    std::vector<Storage> rope_forward(const Storage& x,
                                      const Storage* position_ids,
                                      const Shape& x_shape,
                                      bool interleaved,
                                      Dtype pos_dtype,
                                      Dtype dt) override {
        (void)pos_dtype;
        const auto& gx = std::get<GpuStorage>(x);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const std::size_t ndim = x_shape.size();
        const int L = static_cast<int>(x_shape[ndim - 2]);
        const int D = static_cast<int>(x_shape.back());
        const int Dh = D / 2;
        auto in_shape = gpu::to_mlx_shape(x_shape);
        ::mlx::core::array pos_arr{0};
        if (position_ids) {
            const auto& gp = std::get<GpuStorage>(*position_ids);
            pos_arr = ::mlx::core::astype(*gp.arr, mlx_dt);
        } else {
            pos_arr = ::mlx::core::astype(::mlx::core::arange(0, L, 1), mlx_dt);
        }
        pos_arr = ::mlx::core::reshape(pos_arr, {L, 1});
        const float coef = static_cast<float>(-2.0 * std::log(10000.0) / static_cast<double>(D));
        auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), mlx_dt);
        auto theta = ::mlx::core::exp(
            ::mlx::core::multiply(::mlx::core::astype(::mlx::core::array(coef), mlx_dt), k_arr));
        auto angle = ::mlx::core::matmul(pos_arr, ::mlx::core::reshape(theta, {1, Dh}));
        auto cos_t = ::mlx::core::cos(angle);
        auto sin_t = ::mlx::core::sin(angle);
        ::mlx::core::Shape cos_b_shape;
        for (std::size_t i = 0; i + 2 < ndim; ++i) cos_b_shape.push_back(1);
        cos_b_shape.push_back(L);
        cos_b_shape.push_back(Dh);
        auto cos_b = ::mlx::core::reshape(cos_t, cos_b_shape);
        auto sin_b = ::mlx::core::reshape(sin_t, cos_b_shape);
        ::mlx::core::array out_arr{0};
        if (interleaved) {
            ::mlx::core::Shape x_shape_p = in_shape;
            x_shape_p.pop_back(); x_shape_p.push_back(Dh); x_shape_p.push_back(2);
            auto x_re = ::mlx::core::reshape(*gx.arr, x_shape_p);
            auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto xe = ::mlx::core::take(x_re, idx0, -1);
            auto xo = ::mlx::core::take(x_re, idx1, -1);
            auto out_e = ::mlx::core::subtract(::mlx::core::multiply(xe, cos_b),
                                               ::mlx::core::multiply(xo, sin_b));
            auto out_o = ::mlx::core::add(::mlx::core::multiply(xo, cos_b),
                                          ::mlx::core::multiply(xe, sin_b));
            auto oe = ::mlx::core::expand_dims(out_e, -1);
            auto oo = ::mlx::core::expand_dims(out_o, -1);
            out_arr = ::mlx::core::reshape(
                ::mlx::core::concatenate(std::vector<::mlx::core::array>{oe, oo}, -1), in_shape);
        } else {
            auto i1 = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), ::mlx::core::int64);
            auto i2 = ::mlx::core::astype(::mlx::core::arange(Dh, D, 1), ::mlx::core::int64);
            auto xa = ::mlx::core::take(*gx.arr, i1, -1);
            auto xb = ::mlx::core::take(*gx.arr, i2, -1);
            auto oa = ::mlx::core::subtract(::mlx::core::multiply(xa, cos_b),
                                            ::mlx::core::multiply(xb, sin_b));
            auto ob = ::mlx::core::add(::mlx::core::multiply(xb, cos_b),
                                       ::mlx::core::multiply(xa, sin_b));
            out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{oa, ob}, -1);
        }
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out_arr), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(cos_t), dt)},
                Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(sin_t), dt)}};
    }

    Storage rope_backward(const Storage& grad_out,
                          const Storage& saved_cos,
                          const Storage& saved_sin,
                          const Shape& x_shape,
                          bool interleaved,
                          Dtype dt) override {
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto& cs = std::get<GpuStorage>(saved_cos);
        const auto& ss = std::get<GpuStorage>(saved_sin);
        const std::size_t ndim = x_shape.size();
        const int L = static_cast<int>(x_shape[ndim - 2]);
        const int D = static_cast<int>(x_shape.back());
        const int Dh = D / 2;
        auto in_shape = gpu::to_mlx_shape(x_shape);
        ::mlx::core::Shape cos_b_shape;
        for (std::size_t i = 0; i + 2 < ndim; ++i) cos_b_shape.push_back(1);
        cos_b_shape.push_back(L); cos_b_shape.push_back(Dh);
        auto cos_b = ::mlx::core::reshape(*cs.arr, cos_b_shape);
        auto sin_b = ::mlx::core::reshape(*ss.arr, cos_b_shape);
        ::mlx::core::array out_arr{0};
        if (interleaved) {
            ::mlx::core::Shape g_shape = in_shape;
            g_shape.pop_back(); g_shape.push_back(Dh); g_shape.push_back(2);
            auto g_re = ::mlx::core::reshape(*gg.arr, g_shape);
            auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto ge = ::mlx::core::take(g_re, idx0, -1);
            auto go = ::mlx::core::take(g_re, idx1, -1);
            auto dxe = ::mlx::core::add(::mlx::core::multiply(cos_b, ge),
                                        ::mlx::core::multiply(sin_b, go));
            auto dxo = ::mlx::core::subtract(::mlx::core::multiply(cos_b, go),
                                             ::mlx::core::multiply(sin_b, ge));
            auto de = ::mlx::core::expand_dims(dxe, -1);
            auto do_ = ::mlx::core::expand_dims(dxo, -1);
            out_arr = ::mlx::core::reshape(
                ::mlx::core::concatenate(std::vector<::mlx::core::array>{de, do_}, -1), in_shape);
        } else {
            auto i1 = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), ::mlx::core::int64);
            auto i2 = ::mlx::core::astype(::mlx::core::arange(Dh, D, 1), ::mlx::core::int64);
            auto ga = ::mlx::core::take(*gg.arr, i1, -1);
            auto gb = ::mlx::core::take(*gg.arr, i2, -1);
            auto dxa = ::mlx::core::add(::mlx::core::multiply(cos_b, ga),
                                        ::mlx::core::multiply(sin_b, gb));
            auto dxb = ::mlx::core::subtract(::mlx::core::multiply(cos_b, gb),
                                             ::mlx::core::multiply(sin_b, ga));
            out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{dxa, dxb}, -1);
        }
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out_arr), dt)};
    }

    // ---- Pooling ---------------------------------------------------------

    std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                             const Shape& x_shape,
                                             const Shape& out_shape,
                                             const PoolOpts& opts,
                                             Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3], K[3], stride[3], Sp[3];
        int K_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            K[i] = opts.K[i]; stride[i] = opts.stride[i];
            Sp[i] = S[i] + 2 * opts.pad[i];
            K_total *= K[i];
        }
        ::mlx::core::array neg_inf(-std::numeric_limits<double>::infinity(),
                                   gpu::to_mlx_dtype(dt));
        std::vector<std::pair<int, int>> pad_widths;
        pad_widths.emplace_back(0, 0); pad_widths.emplace_back(0, 0);
        for (int i = 0; i < N; ++i) pad_widths.emplace_back(opts.pad[i], opts.pad[i]);
        auto x_pad = ::mlx::core::pad(*gx.arr, pad_widths, neg_inf);
        auto wins = gpu_build_window_view(x_pad, B, C, Sp, O, K, stride, N);
        std::vector<int> kernel_axes;
        for (int i = 0; i < N; ++i) kernel_axes.push_back(2 + N + i);
        auto y = ::mlx::core::max(wins, kernel_axes, false);
        ::mlx::core::Shape flat_win;
        flat_win.push_back(B); flat_win.push_back(C);
        for (int i = 0; i < N; ++i) flat_win.push_back(O[i]);
        flat_win.push_back(K_total);
        auto wins_flat = ::mlx::core::reshape(wins, flat_win);
        auto argmax = ::mlx::core::astype(
            ::mlx::core::argmax(wins_flat, 2 + N, false), ::mlx::core::int32);
        return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(argmax), Dtype::I32)}};
    }

    Storage max_pool_nd_backward(const Storage& grad_out,
                                 const Storage& saved_argmax,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override {
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto& gA = std::get<GpuStorage>(saved_argmax);
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3], Sp[3];
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            Sp[i] = S[i] + 2 * opts.pad[i];
        }
        int K_suffix[4]; K_suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i) K_suffix[i] = K_suffix[i + 1] * opts.K[i];
        using SE = ::mlx::core::ShapeElem;
        const auto idt = ::mlx::core::int32;
        auto compute_ih = [&](int i) -> ::mlx::core::array {
            auto div_a = ::mlx::core::array(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
            auto mod_a = ::mlx::core::array(static_cast<std::int32_t>(opts.K[i]), idt);
            auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(*gA.arr, div_a), mod_a);
            ::mlx::core::Shape rs(N + 2, 1); rs[2 + i] = static_cast<SE>(O[i]);
            auto o_r = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), rs);
            return ::mlx::core::add(
                ::mlx::core::multiply(o_r, ::mlx::core::array(static_cast<std::int32_t>(opts.stride[i]), idt)), ki);
        };
        ::mlx::core::array flat_idx = compute_ih(0);
        for (int i = 1; i < N; ++i)
            flat_idx = ::mlx::core::add(
                ::mlx::core::multiply(flat_idx, ::mlx::core::array(static_cast<std::int32_t>(Sp[i]), idt)),
                compute_ih(i));
        ::mlx::core::Shape full_O; full_O.push_back(B); full_O.push_back(C);
        for (int i = 0; i < N; ++i) full_O.push_back(O[i]);
        flat_idx = ::mlx::core::broadcast_to(flat_idx, full_O);
        const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
        std::int64_t Sp_total = 1;
        for (int i = 0; i < N; ++i) Sp_total *= Sp[i];
        auto idx_2d = ::mlx::core::reshape(flat_idx, {BC, static_cast<SE>(x_shape[2])});
        auto g_2d = ::mlx::core::reshape(*gG.arr, {BC, static_cast<SE>(out_shape[2])});
        auto zero_pad = ::mlx::core::zeros({BC, static_cast<SE>(Sp_total)},
                                           gpu::to_mlx_dtype(dt));
        // Recompute O_total for actual reshape.
        std::int64_t O_total = 1;
        for (int i = 0; i < N; ++i) O_total *= O[i];
        auto idx_2d_flat = ::mlx::core::reshape(flat_idx, {BC, static_cast<SE>(O_total)});
        auto g_2d_flat  = ::mlx::core::reshape(*gG.arr, {BC, static_cast<SE>(O_total)});
        auto dx_pad_2d = ::mlx::core::scatter_add_axis(zero_pad, idx_2d_flat, g_2d_flat, 1);
        ::mlx::core::Shape full_Sp; full_Sp.push_back(B); full_Sp.push_back(C);
        for (int i = 0; i < N; ++i) full_Sp.push_back(Sp[i]);
        auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);
        ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
        crop_hi.push_back(B); crop_hi.push_back(C);
        for (int i = 0; i < N; ++i) {
            crop_lo[2 + i] = static_cast<SE>(opts.pad[i]);
            crop_hi.push_back(static_cast<SE>(opts.pad[i] + S[i]));
        }
        auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
        return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
        (void)idx_2d; (void)g_2d;
    }

    Storage avg_pool_nd_forward(const Storage& x,
                                const Shape& x_shape,
                                const Shape& out_shape,
                                const PoolOpts& opts,
                                Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3], K[3], stride[3], Sp[3];
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            K[i] = opts.K[i]; stride[i] = opts.stride[i];
            Sp[i] = S[i] + 2 * opts.pad[i];
        }
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        std::vector<std::pair<int, int>> pad_widths;
        pad_widths.emplace_back(0, 0); pad_widths.emplace_back(0, 0);
        for (int i = 0; i < N; ++i) pad_widths.emplace_back(opts.pad[i], opts.pad[i]);
        auto x_pad = ::mlx::core::pad(*gx.arr, pad_widths, zero);
        auto wins = gpu_build_window_view(x_pad, B, C, Sp, O, K, stride, N);
        std::vector<int> kernel_axes;
        for (int i = 0; i < N; ++i) kernel_axes.push_back(2 + N + i);
        auto y = ::mlx::core::mean(wins, kernel_axes, false);
        return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
    }

    Storage avg_pool_nd_backward(const Storage& grad_out,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override {
        const auto& gG = std::get<GpuStorage>(grad_out);
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3], Sp[3];
        int K_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            Sp[i] = S[i] + 2 * opts.pad[i];
            K_total *= opts.K[i];
        }
        using SE = ::mlx::core::ShapeElem;
        const auto idt = ::mlx::core::int32;
        auto inv_K = ::mlx::core::array(1.0 / static_cast<double>(K_total),
                                        gpu::to_mlx_dtype(dt));
        ::mlx::core::Shape g_with_k; g_with_k.push_back(B); g_with_k.push_back(C);
        for (int i = 0; i < N; ++i) g_with_k.push_back(O[i]);
        g_with_k.push_back(1);
        auto g_exp = ::mlx::core::multiply(::mlx::core::reshape(*gG.arr, g_with_k), inv_K);
        ::mlx::core::Shape full_k = g_with_k; full_k[N + 2] = static_cast<SE>(K_total);
        auto updates = ::mlx::core::broadcast_to(g_exp, full_k);
        ::mlx::core::Shape kr_shape(N + 3, 1); kr_shape[N + 2] = static_cast<SE>(K_total);
        auto k_range = ::mlx::core::reshape(::mlx::core::arange(0, K_total, 1, idt), kr_shape);
        int K_suffix[4]; K_suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i) K_suffix[i] = K_suffix[i + 1] * opts.K[i];
        auto compute_ih = [&](int i) -> ::mlx::core::array {
            auto div_a = ::mlx::core::array(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
            auto mod_a = ::mlx::core::array(static_cast<std::int32_t>(opts.K[i]), idt);
            auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(k_range, div_a), mod_a);
            ::mlx::core::Shape rs(N + 3, 1); rs[2 + i] = static_cast<SE>(O[i]);
            auto o_r = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), rs);
            return ::mlx::core::add(
                ::mlx::core::multiply(o_r, ::mlx::core::array(static_cast<std::int32_t>(opts.stride[i]), idt)), ki);
        };
        ::mlx::core::array flat_idx = compute_ih(0);
        for (int i = 1; i < N; ++i)
            flat_idx = ::mlx::core::add(
                ::mlx::core::multiply(flat_idx, ::mlx::core::array(static_cast<std::int32_t>(Sp[i]), idt)),
                compute_ih(i));
        flat_idx = ::mlx::core::broadcast_to(flat_idx, full_k);
        const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
        std::int64_t Sp_total = 1;
        for (int i = 0; i < N; ++i) Sp_total *= Sp[i];
        std::int64_t O_total = 1;
        for (int i = 0; i < N; ++i) O_total *= O[i];
        auto idx_2d = ::mlx::core::reshape(flat_idx, {BC, static_cast<SE>(O_total * K_total)});
        auto upd_2d = ::mlx::core::reshape(updates, {BC, static_cast<SE>(O_total * K_total)});
        auto zero_pad = ::mlx::core::zeros({BC, static_cast<SE>(Sp_total)}, gpu::to_mlx_dtype(dt));
        auto dx_pad_2d = ::mlx::core::scatter_add_axis(zero_pad, idx_2d, upd_2d, 1);
        ::mlx::core::Shape full_Sp; full_Sp.push_back(B); full_Sp.push_back(C);
        for (int i = 0; i < N; ++i) full_Sp.push_back(Sp[i]);
        auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);
        ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
        crop_hi.push_back(B); crop_hi.push_back(C);
        for (int i = 0; i < N; ++i) {
            crop_lo[2 + i] = static_cast<SE>(opts.pad[i]);
            crop_hi.push_back(static_cast<SE>(opts.pad[i] + S[i]));
        }
        auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
        return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
    }

    // ---- N-D convolution -------------------------------------------------

    Storage conv_nd_forward(
        const Storage& x, const Storage& W, const Storage& b,
        int B, int Cin, int Cout, int Cin_g, int Cout_g,
        const int* S, const int* K, const int* O,
        const ConvNdOpts& opts,
        const Shape& out_shape, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gW = std::get<GpuStorage>(W);
        const auto& gb = std::get<GpuStorage>(b);
        (void)Cin; (void)Cout; (void)Cin_g; (void)Cout_g; (void)S; (void)K; (void)O; (void)B;
        const int N = opts.N;
        std::vector<int> sv(opts.stride, opts.stride + N);
        std::vector<int> pv(opts.pad, opts.pad + N);
        std::vector<int> dv(opts.dilation, opts.dilation + N);
        std::vector<int> idv(N, 1);
        auto x_nhwc = ::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N));
        auto W_nhwc = ::mlx::core::transpose(*gW.arr, gpu_nchw_to_nhwc_perm(N));
        auto y_nhwc = ::mlx::core::conv_general(x_nhwc, W_nhwc, sv, pv, pv, dv, idv,
                                                opts.groups, false);
        int Cout_local = static_cast<int>(out_shape[1]);
        ::mlx::core::Shape b_brd(N + 2, 1); b_brd[N + 1] = Cout_local;
        y_nhwc = ::mlx::core::add(y_nhwc, ::mlx::core::reshape(*gb.arr, b_brd));
        auto y = ::mlx::core::contiguous(::mlx::core::transpose(y_nhwc, gpu_nhwc_to_nchw_perm(N)));
        return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
    }

    std::vector<Storage> conv_nd_backward(
        const Storage& grad_out, const Storage& x, const Storage& W,
        int B, int Cin, int Cout, int Cin_g, int Cout_g,
        const int* S, const int* K, const int* O,
        const ConvNdOpts& opts, Dtype dt) override {
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gW = std::get<GpuStorage>(W);
        (void)B; (void)Cin; (void)Cout; (void)Cin_g; (void)Cout_g;
        const int N = opts.N;
        std::vector<int> sv(opts.stride, opts.stride + N);
        std::vector<int> pv(opts.pad, opts.pad + N);
        std::vector<int> dv(opts.dilation, opts.dilation + N);
        // db = sum over batch + spatial axes
        std::vector<int> db_axes; db_axes.push_back(0);
        for (int i = 0; i < N; ++i) db_axes.push_back(2 + i);
        auto db = ::mlx::core::sum(*gG.arr, db_axes, false);
        // dx via transposed conv
        std::vector<int> opad(N);
        for (int i = 0; i < N; ++i)
            opad[i] = S[i] + (K[i] - 1) * dv[i] - 1 - (O[i] - 1) * sv[i] - pv[i];
        auto grad_nhwc = ::mlx::core::transpose(*gG.arr, gpu_nchw_to_nhwc_perm(N));
        std::vector<int> W_t_perm = gpu_w_to_transpose_perm(N);
        auto W_t_nhwc = ::mlx::core::transpose(*gW.arr, W_t_perm);
        std::vector<int> ones_n(N, 1);
        auto dx_nhwc = ::mlx::core::conv_general(grad_nhwc, W_t_nhwc, ones_n, pv, opad,
                                                 dv, sv, opts.groups, true);
        auto dx = ::mlx::core::contiguous(::mlx::core::transpose(dx_nhwc, gpu_nhwc_to_nchw_perm(N)));
        // dW via strided conv_general trick
        std::vector<int> perm;
        perm.push_back(1);
        for (int i = 0; i < N; ++i) perm.push_back(2 + i);
        perm.push_back(0);
        auto x_perm = ::mlx::core::transpose(*gx.arr, perm);
        auto g_perm = ::mlx::core::transpose(*gG.arr, perm);
        std::vector<int> conv_s(N, 1);
        auto dW_perm = ::mlx::core::conv_general(x_perm, g_perm, conv_s, pv, pv, sv, dv,
                                                  opts.groups, false);
        ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
        crop_hi.push_back(Cin_g);
        for (int i = 0; i < N; ++i) crop_hi.push_back(K[i]);
        crop_hi.push_back(Cout);
        dW_perm = ::mlx::core::slice(dW_perm, crop_lo, crop_hi);
        std::vector<int> dW_back; dW_back.push_back(N + 1); dW_back.push_back(0);
        for (int i = 0; i < N; ++i) dW_back.push_back(1 + i);
        auto dW = ::mlx::core::contiguous(::mlx::core::transpose(dW_perm, dW_back));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
                Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
    }

    Storage unfold_forward(
        const Storage& x,
        int B, int C,
        const std::vector<int>& S, const std::vector<int>& K, const std::vector<int>& O,
        const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation,
        const Shape& out_shape, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const int N = static_cast<int>(S.size());
        int S_total = 1;
        for (int d : S) S_total *= d;
        ::mlx::core::Shape composite;
        composite.push_back(B); composite.push_back(C);
        for (int i = 0; i < N; ++i) composite.push_back(K[i]);
        for (int i = 0; i < N; ++i) composite.push_back(O[i]);
        const auto i32 = ::mlx::core::int32;
        ::mlx::core::Shape b_shape(composite.size(), 1); b_shape[0] = B;
        auto b_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);
        ::mlx::core::Shape c_shape(composite.size(), 1); c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);
        std::optional<::mlx::core::array> valid_opt;
        std::vector<::mlx::core::array> in_d_clipped;
        for (int d = 0; d < N; ++d) {
            auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32);
            auto o_arr = ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32);
            ::mlx::core::Shape ks(composite.size(), 1); ks[2 + d] = K[d];
            ::mlx::core::Shape os(composite.size(), 1); os[2 + N + d] = O[d];
            k_arr = ::mlx::core::reshape(k_arr, ks);
            o_arr = ::mlx::core::reshape(o_arr, os);
            auto sd = ::mlx::core::array(stride[d], i32);
            auto dd = ::mlx::core::array(dilation[d], i32);
            auto pd = ::mlx::core::array(pad[d], i32);
            auto in_d = ::mlx::core::subtract(
                ::mlx::core::add(::mlx::core::multiply(sd, o_arr),
                                 ::mlx::core::multiply(dd, k_arr)), pd);
            auto zero_i = ::mlx::core::array(0, i32);
            auto cap_i = ::mlx::core::array(S[d] - 1, i32);
            auto v = ::mlx::core::logical_and(
                ::mlx::core::greater_equal(in_d, zero_i), ::mlx::core::less_equal(in_d, cap_i));
            valid_opt = valid_opt.has_value() ? ::mlx::core::logical_and(*valid_opt, v) : v;
            in_d_clipped.push_back(
                ::mlx::core::clip(in_d, std::optional<::mlx::core::array>(zero_i),
                                  std::optional<::mlx::core::array>(cap_i)));
        }
        auto valid = *valid_opt;
        auto flat = ::mlx::core::add(
            ::mlx::core::multiply(b_idx, ::mlx::core::array(C * S_total, i32)),
            ::mlx::core::multiply(c_idx, ::mlx::core::array(S_total, i32)));
        for (int d = 0; d < N; ++d) {
            int trailing = 1;
            for (int e = d + 1; e < N; ++e) trailing *= S[e];
            flat = ::mlx::core::add(flat,
                ::mlx::core::multiply(in_d_clipped[d], ::mlx::core::array(trailing, i32)));
        }
        flat = ::mlx::core::broadcast_to(flat, composite);
        auto x_flat = ::mlx::core::reshape(*gx.arr, {B * C * S_total});
        auto sampled = ::mlx::core::take(x_flat, flat);
        int K_total = 1; for (int k : K) K_total *= k;
        int O_total = 1; for (int o : O) O_total *= o;
        auto valid_b = ::mlx::core::astype(
            ::mlx::core::broadcast_to(valid, composite), gpu::to_mlx_dtype(dt));
        auto masked = ::mlx::core::multiply(sampled, valid_b);
        auto reshaped = ::mlx::core::reshape(masked, {B, C * K_total, O_total});
        return Storage{gpu::wrap_mlx_array(std::move(reshaped), dt)};
        (void)out_shape;
    }

    Storage unfold_backward(
        const Storage& grad_out,
        int B, int C,
        const std::vector<int>& S, const std::vector<int>& K, const std::vector<int>& O,
        const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation,
        Dtype dt) override {
        // Route through unfold_forward on the transposed problem: scatter gradients.
        // GPU unfold backward: scatter-add from (B, C*K_total, O_total) → (B, C, *S).
        // Reuse the same index computation from unfold_forward, then scatter_add.
        const auto& gg = std::get<GpuStorage>(grad_out);
        const int N = static_cast<int>(S.size());
        int S_total = 1; for (int d : S) S_total *= d;
        int K_total = 1; for (int k : K) K_total *= k;
        int O_total = 1; for (int o : O) O_total *= o;
        ::mlx::core::Shape composite;
        composite.push_back(B); composite.push_back(C);
        for (int i = 0; i < N; ++i) composite.push_back(K[i]);
        for (int i = 0; i < N; ++i) composite.push_back(O[i]);
        const auto i32 = ::mlx::core::int32;
        ::mlx::core::Shape b_shape(composite.size(), 1); b_shape[0] = B;
        auto b_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);
        ::mlx::core::Shape c_shape(composite.size(), 1); c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);
        std::optional<::mlx::core::array> valid_opt;
        std::vector<::mlx::core::array> in_d_clipped;
        for (int d = 0; d < N; ++d) {
            auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32);
            auto o_arr = ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32);
            ::mlx::core::Shape ks(composite.size(), 1); ks[2 + d] = K[d];
            ::mlx::core::Shape os(composite.size(), 1); os[2 + N + d] = O[d];
            k_arr = ::mlx::core::reshape(k_arr, ks);
            o_arr = ::mlx::core::reshape(o_arr, os);
            auto sd = ::mlx::core::array(stride[d], i32);
            auto dd = ::mlx::core::array(dilation[d], i32);
            auto pd = ::mlx::core::array(pad[d], i32);
            auto in_d = ::mlx::core::subtract(
                ::mlx::core::add(::mlx::core::multiply(sd, o_arr),
                                 ::mlx::core::multiply(dd, k_arr)), pd);
            auto zero_i = ::mlx::core::array(0, i32);
            auto cap_i = ::mlx::core::array(S[d] - 1, i32);
            auto v = ::mlx::core::logical_and(
                ::mlx::core::greater_equal(in_d, zero_i), ::mlx::core::less_equal(in_d, cap_i));
            valid_opt = valid_opt.has_value() ? ::mlx::core::logical_and(*valid_opt, v) : v;
            in_d_clipped.push_back(
                ::mlx::core::clip(in_d, std::optional<::mlx::core::array>(zero_i),
                                  std::optional<::mlx::core::array>(cap_i)));
        }
        auto valid = *valid_opt;
        auto flat = ::mlx::core::add(
            ::mlx::core::multiply(b_idx, ::mlx::core::array(C * S_total, i32)),
            ::mlx::core::multiply(c_idx, ::mlx::core::array(S_total, i32)));
        for (int d = 0; d < N; ++d) {
            int trailing = 1;
            for (int e = d + 1; e < N; ++e) trailing *= S[e];
            flat = ::mlx::core::add(flat,
                ::mlx::core::multiply(in_d_clipped[d], ::mlx::core::array(trailing, i32)));
        }
        flat = ::mlx::core::broadcast_to(flat, composite);
        // Apply valid mask to gradients
        auto valid_b = ::mlx::core::astype(
            ::mlx::core::broadcast_to(valid, composite), gpu::to_mlx_dtype(dt));
        // grad_out is (B, C*K_total, O_total); reshape to composite shape
        auto g_comp = ::mlx::core::reshape(*gg.arr, composite);
        g_comp = ::mlx::core::multiply(g_comp, valid_b);
        // Scatter-add: dx_flat[flat[i]] += g_comp[i]
        auto flat_2d = ::mlx::core::reshape(flat, {B * C * K_total * O_total});
        auto g_flat = ::mlx::core::reshape(g_comp, {B * C * K_total * O_total});
        auto dx_flat = ::mlx::core::zeros({B * C * S_total},
                                          gpu::to_mlx_dtype(dt));
        auto flat_2d_i64 = ::mlx::core::astype(flat_2d, ::mlx::core::int64);
        dx_flat = ::mlx::core::scatter_add_axis(
            dx_flat, ::mlx::core::reshape(flat_2d_i64, {B * C * K_total * O_total, 1}),
            ::mlx::core::reshape(g_flat, {B * C * K_total * O_total, 1}), 0);
        auto dx = ::mlx::core::reshape(dx_flat, {B, C, S_total});
        // Reshape back to (B, C, *S)
        ::mlx::core::Shape s_shape; s_shape.push_back(B); s_shape.push_back(C);
        for (int d : S) s_shape.push_back(d);
        dx = ::mlx::core::reshape(dx_flat, s_shape);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)};
    }

private:
    // ---- Helpers -------------------------------------------------------

    inline static const ::mlx::core::Device k_linalg_stream{::mlx::core::Device::cpu};

    // ---- N-D conv permutation helpers (runtime N = 1/2/3) ---------------

    static std::vector<int> gpu_nchw_to_nhwc_perm(int N) {
        std::vector<int> p; p.reserve(N + 2);
        p.push_back(0);
        for (int i = 0; i < N; ++i) p.push_back(2 + i);
        p.push_back(1);
        return p;
    }

    static std::vector<int> gpu_nhwc_to_nchw_perm(int N) {
        std::vector<int> p; p.reserve(N + 2);
        p.push_back(0); p.push_back(N + 1);
        for (int i = 0; i < N; ++i) p.push_back(1 + i);
        return p;
    }

    // W for ConvTranspose: (Cin, Cout, *K) → (Cout, *K, Cin) = [1, 2..N+1, 0]
    static std::vector<int> gpu_w_to_mlx_transpose_perm(int N) {
        std::vector<int> p; p.push_back(1);
        for (int i = 0; i < N; ++i) p.push_back(2 + i);
        p.push_back(0);
        return p;
    }

    // W for ConvNd backward transpose: (Cout, Cin/g, *K) → (Cin/g, *K, Cout) = [1, 2..N+1, 0]
    static std::vector<int> gpu_w_to_transpose_perm(int N) {
        std::vector<int> p; p.push_back(1);
        for (int i = 0; i < N; ++i) p.push_back(2 + i);
        p.push_back(0);
        return p;
    }

    static ::mlx::core::array gpu_mlx_conv_transpose(
        const ::mlx::core::array& x_nhwc, const ::mlx::core::array& W_nhwc,
        const int* stride, const int* pad, const int* opad, int N) {
        if (N == 1)
            return ::mlx::core::conv_transpose1d(x_nhwc, W_nhwc, stride[0], pad[0], 1, opad[0]);
        if (N == 2)
            return ::mlx::core::conv_transpose2d(
                x_nhwc, W_nhwc,
                std::pair<int, int>{stride[0], stride[1]},
                std::pair<int, int>{pad[0], pad[1]},
                std::pair<int, int>{opad[0], opad[1]});
        return ::mlx::core::conv_transpose3d(
            x_nhwc, W_nhwc,
            std::tuple<int, int, int>{stride[0], stride[1], stride[2]},
            std::tuple<int, int, int>{pad[0], pad[1], pad[2]},
            std::tuple<int, int, int>{opad[0], opad[1], opad[2]});
    }

    static ::mlx::core::array gpu_mlx_conv(
        const ::mlx::core::array& x_nhwc, const ::mlx::core::array& W_nhwc,
        const int* stride, const int* pad, int N) {
        if (N == 1) return ::mlx::core::conv1d(x_nhwc, W_nhwc, stride[0], pad[0]);
        if (N == 2) return ::mlx::core::conv2d(x_nhwc, W_nhwc,
            std::pair<int, int>{stride[0], stride[1]},
            std::pair<int, int>{pad[0], pad[1]});
        return ::mlx::core::conv3d(x_nhwc, W_nhwc,
            std::tuple<int, int, int>{stride[0], stride[1], stride[2]},
            std::tuple<int, int, int>{pad[0], pad[1], pad[2]});
    }

    static ::mlx::core::array gpu_build_window_view(
        const ::mlx::core::array& padded, int B, int C,
        const int* Sp, const int* O, const int* K, const int* stride, int N) {
        using SE = ::mlx::core::ShapeElem;
        using SS = std::int64_t;
        ::mlx::core::Shape windowed;
        windowed.reserve(2 + 2 * N);
        windowed.push_back(B); windowed.push_back(C);
        for (int i = 0; i < N; ++i) windowed.push_back(static_cast<SE>(O[i]));
        for (int i = 0; i < N; ++i) windowed.push_back(static_cast<SE>(K[i]));
        SS suffix[4]; suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i) suffix[i] = suffix[i + 1] * static_cast<SS>(Sp[i]);
        ::mlx::core::Strides strides_v;
        strides_v.reserve(2 + 2 * N);
        strides_v.push_back(static_cast<SS>(C) * suffix[0]);
        strides_v.push_back(suffix[0]);
        for (int i = 0; i < N; ++i) strides_v.push_back(static_cast<SS>(stride[i]) * suffix[i + 1]);
        for (int i = 0; i < N; ++i) strides_v.push_back(suffix[i + 1]);
        return ::mlx::core::as_strided(padded, windowed, strides_v, 0);
    }

    ::mlx::core::array take_descending_top_indices(const ::mlx::core::array& idx,
                                                   int axis,
                                                   const Shape& output_shape) {
        const auto& full_shape = idx.shape();
        const std::int64_t k = output_shape[static_cast<std::size_t>(axis)];
        const std::int64_t L = full_shape[axis];
        std::vector<std::int32_t> selector(static_cast<std::size_t>(k));
        for (std::int64_t i = 0; i < k; ++i)
            selector[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(L - 1 - i);
        ::mlx::core::Shape selector_shape(full_shape.size(), 1);
        selector_shape[axis] = static_cast<int>(k);
        ::mlx::core::array selector_arr(selector.data(), selector_shape, ::mlx::core::int32);
        ::mlx::core::Shape out_shape = full_shape;
        out_shape[axis] = static_cast<int>(k);
        selector_arr = ::mlx::core::broadcast_to(selector_arr, out_shape);
        return ::mlx::core::take_along_axis(idx, selector_arr, axis);
    }

    ::mlx::core::array apply_loss_reduction(const ::mlx::core::array& values, int reduction) {
        if (reduction == 0)
            return values;
        auto sum = ::mlx::core::sum(values, /*keepdims=*/false);
        if (reduction == 1)
            return ::mlx::core::divide(
                sum, gpu::mlx_scalar(static_cast<double>(values.size()), values.dtype()));
        return sum;
    }

    ::mlx::core::array scale_loss_grad(const ::mlx::core::array& grad,
                                       int reduction,
                                       std::size_t numel,
                                       ::mlx::core::Dtype dt) {
        if (reduction == 1)
            return ::mlx::core::divide(grad, gpu::mlx_scalar(static_cast<double>(numel), dt));
        return grad;
    }

    ::mlx::core::array class_target_indices(const ::mlx::core::array& target,
                                            const Shape& target_shape) {
        auto t_shape = gpu::to_mlx_shape(target_shape);
        t_shape.insert(t_shape.begin() + 1, 1);
        return ::mlx::core::reshape(::mlx::core::astype(target, ::mlx::core::int64), t_shape);
    }

    ::mlx::core::array class_ignore_mask(const ::mlx::core::array& target_idx, int ignore_index) {
        auto ignore = ::mlx::core::astype(::mlx::core::array(ignore_index), ::mlx::core::int64);
        return ::mlx::core::not_equal(target_idx, ignore);
    }

    ::mlx::core::array safe_class_indices(const ::mlx::core::array& target_idx,
                                          const ::mlx::core::array& keep_mask) {
        auto zero = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        return ::mlx::core::where(keep_mask, target_idx, zero);
    }

    ::mlx::core::array class_weight_gather(const Storage* weight,
                                           const ::mlx::core::array& safe_t,
                                           const ::mlx::core::Shape& shape,
                                           ::mlx::core::Dtype dt) {
        if (weight) {
            const auto& w = std::get<GpuStorage>(*weight);
            return ::mlx::core::take(*w.arr, safe_t);
        }
        return ::mlx::core::broadcast_to(gpu::mlx_scalar(1.0, dt), shape);
    }

    ::mlx::core::array class_valid_count(const ::mlx::core::array& keep_mask_dt,
                                         ::mlx::core::Dtype dt) {
        auto valid = ::mlx::core::sum(keep_mask_dt, /*keepdims=*/false);
        return ::mlx::core::maximum(valid, gpu::mlx_scalar(1.0, dt));
    }

    Storage reduce_class_loss(const ::mlx::core::array& values,
                              const ::mlx::core::array& valid_count,
                              Dtype dt,
                              int reduction) {
        if (reduction == 0)
            return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(values), dt)};
        auto sum = ::mlx::core::sum(values, /*keepdims=*/false);
        if (reduction == 1)
            sum = ::mlx::core::divide(sum, valid_count);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(sum), dt)};
    }

    ::mlx::core::array class_scaled_grad(const ::mlx::core::array& grad,
                                         const ::mlx::core::array& valid_count,
                                         const ::mlx::core::Shape& target_axis_shape,
                                         int reduction) {
        ::mlx::core::array scaled = grad;
        if (reduction == 1)
            scaled = ::mlx::core::divide(scaled, valid_count);
        if (reduction != 0)
            return ::mlx::core::broadcast_to(scaled, target_axis_shape);
        auto grad_shape = scaled.shape();
        grad_shape.insert(grad_shape.begin() + 1, 1);
        return ::mlx::core::reshape(scaled, grad_shape);
    }

    static std::size_t linear_batch(const Shape& x_shape) {
        std::size_t M = 1;
        for (std::size_t d = 0; d + 1 < x_shape.size(); ++d)
            M *= static_cast<std::size_t>(x_shape[d]);
        return M;
    }

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

    static float perm_index_sign(const std::uint32_t* p, std::size_t n) {
        std::vector<bool> seen(n, false);
        std::size_t cycles = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (seen[i])
                continue;
            ++cycles;
            std::size_t j = i;
            while (!seen[j]) {
                seen[j] = true;
                j = p[j];
            }
        }
        return ((n - cycles) % 2 == 0) ? 1.0f : -1.0f;
    }

    // ---- BatchNorm eval --------------------------------------------------

    std::vector<Storage> batch_norm_eval_forward(
        const Storage& x, const Storage& mean, const Storage& var,
        const Storage& gamma, const Storage& beta,
        const Shape& x_shape, int C, int /*spatial*/, double eps, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gm = std::get<GpuStorage>(mean);
        const auto& gv = std::get<GpuStorage>(var);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gb = std::get<GpuStorage>(beta);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(*gv.arr, eps_arr));
        ::mlx::core::Shape b_shape(x_shape.size(), 1);
        b_shape[1] = C;
        auto m_b = ::mlx::core::reshape(*gm.arr, b_shape);
        auto r_b = ::mlx::core::reshape(rstd, b_shape);
        auto g_b = ::mlx::core::reshape(*gg.arr, b_shape);
        auto bb_b = ::mlx::core::reshape(*gb.arr, b_shape);
        auto y = ::mlx::core::add(
            ::mlx::core::multiply(g_b,
                ::mlx::core::multiply(::mlx::core::subtract(*gx.arr, m_b), r_b)), bb_b);
        return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(rstd), dt)}};
    }

    std::vector<Storage> batch_norm_eval_backward(
        const Storage& x, const Storage& mean, const Storage& gamma,
        const Storage& rstd, const Storage& grad_out,
        const Shape& x_shape, int C, int /*spatial*/, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gm = std::get<GpuStorage>(mean);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& rs = std::get<GpuStorage>(rstd);
        const auto& go = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        ::mlx::core::Shape b_shape(x_shape.size(), 1);
        b_shape[1] = C;
        auto m_b = ::mlx::core::reshape(*gm.arr, b_shape);
        auto r_b = ::mlx::core::reshape(*rs.arr, b_shape);
        auto g_b = ::mlx::core::reshape(*gg.arr, b_shape);
        auto x_minus_m = ::mlx::core::subtract(*gx.arr, m_b);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(g_b, r_b), *go.arr);
        std::vector<int> reduce_axes;
        for (std::size_t i = 0; i < x_shape.size(); ++i)
            if (i != 1) reduce_axes.push_back(static_cast<int>(i));
        auto sum_g = ::mlx::core::sum(*go.arr, reduce_axes, false);
        auto xm_g  = ::mlx::core::multiply(x_minus_m, *go.arr);
        auto sum_xm_g = ::mlx::core::sum(xm_g, reduce_axes, false);
        auto db = sum_g;
        auto dg = ::mlx::core::multiply(sum_xm_g, *rs.arr);
        auto dm = ::mlx::core::negative(
            ::mlx::core::multiply(*gg.arr, ::mlx::core::multiply(*rs.arr, sum_g)));
        auto half = ::mlx::core::astype(::mlx::core::array(-0.5f), mlx_dt);
        auto r3 = ::mlx::core::multiply(*rs.arr, ::mlx::core::multiply(*rs.arr, *rs.arr));
        auto dv = ::mlx::core::multiply(
            half, ::mlx::core::multiply(*gg.arr, ::mlx::core::multiply(r3, sum_xm_g)));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dm), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dv), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dg), dt)},
                Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
    }

    // ---- Lp-normalize ---------------------------------------------------

    std::vector<Storage> lp_normalize_forward(
        const Storage& x, const Shape& x_shape, double ord, int axis,
        double eps, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto abs_x = ::mlx::core::abs(*gx.arr);
        auto ord_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(ord)), mlx_dt);
        auto inv_ord = ::mlx::core::astype(::mlx::core::array(static_cast<float>(1.0/ord)), mlx_dt);
        auto pow_x = ::mlx::core::power(abs_x, ord_arr);
        auto sum_p = ::mlx::core::sum(pow_x, std::vector<int>{axis}, true);
        auto N = ::mlx::core::power(sum_p, inv_ord);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto N_clip = ::mlx::core::maximum(N, eps_arr);
        auto saved_norm = N_clip;  // keep copy for saved_norm
        auto y = ::mlx::core::divide(*gx.arr, N_clip);
        return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(saved_norm), dt)}};
    }

    Storage lp_normalize_backward(
        const Storage& x, const Storage& saved_norm, const Storage& grad_out,
        const Shape& /*x_shape*/, double ord, int axis, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gn = std::get<GpuStorage>(saved_norm);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        auto proj = ::mlx::core::sum(::mlx::core::multiply(*gg.arr, *gx.arr),
                                     std::vector<int>{axis}, true);
        auto first = ::mlx::core::divide(*gg.arr, *gn.arr);
        auto sign_x = ::mlx::core::sign(*gx.arr);
        auto abs_x = ::mlx::core::abs(*gx.arr);
        auto ord_m1 = ::mlx::core::astype(
            ::mlx::core::array(static_cast<float>(ord-1.0)), mlx_dt);
        auto ord_p1 = ::mlx::core::astype(
            ::mlx::core::array(static_cast<float>(ord+1.0)), mlx_dt);
        auto abs_pm1 = ::mlx::core::power(abs_x, ord_m1);
        auto N_pp1   = ::mlx::core::power(*gn.arr, ord_p1);
        auto second  = ::mlx::core::divide(
            ::mlx::core::multiply(::mlx::core::multiply(sign_x, abs_pm1), proj), N_pp1);
        auto dx = ::mlx::core::subtract(first, second);
        return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
    }

    // ---- Global response normalization -----------------------------------

    std::vector<Storage> global_response_norm_forward(
        const Storage& x, const Storage& gamma, const Storage& beta,
        const Shape& x_shape, double eps, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gb = std::get<GpuStorage>(beta);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const int C = static_cast<int>(x_shape[1]);
        const int B = static_cast<int>(x_shape[0]);
        auto x_sq = ::mlx::core::multiply(*gx.arr, *gx.arr);
        auto G_sq = ::mlx::core::sum(x_sq, std::vector<int>{2, 3}, true);
        auto G = ::mlx::core::sqrt(G_sq);
        auto m = ::mlx::core::mean(G, std::vector<int>{1}, true);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto denom = ::mlx::core::add(m, eps_arr);
        auto Nx = ::mlx::core::divide(G, denom);
        auto g_b = ::mlx::core::reshape(*gg.arr, {1, C, 1, 1});
        auto bb_b = ::mlx::core::reshape(*gb.arr, {1, C, 1, 1});
        auto y = ::mlx::core::add(
            ::mlx::core::multiply(g_b, ::mlx::core::multiply(*gx.arr, Nx)),
            ::mlx::core::multiply(bb_b, *gx.arr));
        auto Nx_flat = ::mlx::core::reshape(Nx, {B, C});
        return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(Nx_flat), dt)}};
    }

    std::vector<Storage> global_response_norm_backward(
        const Storage& x, const Storage& gamma, const Storage& beta,
        const Storage& saved_Nx, const Storage& grad_out,
        const Shape& x_shape, double eps, Dtype dt) override {
        const auto& gx = std::get<GpuStorage>(x);
        const auto& gg = std::get<GpuStorage>(gamma);
        const auto& gnx = std::get<GpuStorage>(saved_Nx);
        const auto& go = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dt);
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        auto x_sq = ::mlx::core::multiply(*gx.arr, *gx.arr);
        auto G_sq = ::mlx::core::sum(x_sq, std::vector<int>{2, 3}, true);
        auto G = ::mlx::core::sqrt(G_sq);
        auto m_b = ::mlx::core::mean(G, std::vector<int>{1}, true);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto denom = ::mlx::core::add(m_b, eps_arr);
        auto Nx = ::mlx::core::reshape(*gnx.arr, {B, C, 1, 1});
        auto g_b = ::mlx::core::reshape(*gg.arr, {1, C, 1, 1});
        auto gx_prod = ::mlx::core::multiply(*go.arr, *gx.arr);
        auto db = ::mlx::core::sum(gx_prod, std::vector<int>{0, 2, 3}, false);
        auto gxN = ::mlx::core::multiply(gx_prod, Nx);
        auto dg = ::mlx::core::sum(gxN, std::vector<int>{0, 2, 3}, false);
        auto sum_gx = ::mlx::core::sum(gx_prod, std::vector<int>{2, 3}, true);
        auto A = ::mlx::core::multiply(g_b, sum_gx);
        auto AG = ::mlx::core::multiply(A, G);
        auto inner_sum = ::mlx::core::sum(AG, std::vector<int>{1}, true);
        auto C_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(C)), mlx_dt);
        auto denom_sq = ::mlx::core::multiply(denom, denom);
        auto second = ::mlx::core::divide(inner_sum, ::mlx::core::multiply(denom_sq, C_arr));
        auto dG = ::mlx::core::subtract(::mlx::core::divide(A, denom), second);
        auto eps_g = ::mlx::core::astype(::mlx::core::array(1e-12f), mlx_dt);
        auto G_safe = ::mlx::core::maximum(G, eps_g);
        auto dG_term = ::mlx::core::divide(::mlx::core::multiply(dG, *gx.arr), G_safe);
        // dx = γ·N·g + β·g + dG·x/G
        const auto& gb = std::get<GpuStorage>(beta);
        auto bb_b = ::mlx::core::reshape(*gb.arr, {1, C, 1, 1});
        auto dx = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(::mlx::core::multiply(g_b, Nx), *go.arr),
                             ::mlx::core::multiply(bb_b, *go.arr)),
            dG_term);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dg), dt)},
                Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
    }

    // ---- Interpolation stubs (forward-only for now) ----------------------

    Storage interpolate_bilinear_forward(
        const Storage& input, const Shape& in_shape,
        int H_out, int W_out, bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::interpolate_bilinear_forward").not_implemented(
            "not yet routed through GpuBackend");
        (void)input; (void)in_shape; (void)H_out; (void)W_out; (void)align_corners; (void)dt;
        return Storage{};
    }

    Storage interpolate_bilinear_backward(
        const Storage& grad_out, const Shape& in_shape,
        int H_out, int W_out, bool align_corners, Dtype dt) override {
        // No MLX backward for interpolate; fall back to CPU via to_cpu.
        // The op file should handle this case. Return stub.
        ErrorBuilder("gpu::interpolate_bilinear_backward").not_implemented("not yet in GpuBackend");
        (void)grad_out; (void)in_shape; (void)H_out; (void)W_out; (void)align_corners; (void)dt;
        return Storage{};  // unreachable
    }

    Storage interpolate_trilinear_forward(
        const Storage& input, const Shape& in_shape,
        int D_out, int H_out, int W_out, bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::interpolate_trilinear_forward").not_implemented("not yet in GpuBackend");
        (void)input; (void)in_shape; (void)D_out; (void)H_out; (void)W_out;
        (void)align_corners; (void)dt;
        return Storage{};
    }

    Storage interpolate_trilinear_backward(
        const Storage& grad_out, const Shape& in_shape,
        int D_out, int H_out, int W_out, bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::interpolate_trilinear_backward").not_implemented("");
        (void)grad_out; (void)in_shape; (void)D_out; (void)H_out; (void)W_out;
        (void)align_corners; (void)dt;
        return Storage{};
    }

    // ---- Spatial transforms stubs ----------------------------------------

    Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W,
        bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::affine_grid_forward").not_implemented("not yet in GpuBackend");
        (void)theta; (void)N; (void)H; (void)W; (void)align_corners; (void)dt;
        return Storage{};
    }

    Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W,
        bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::affine_grid_backward").not_implemented("");
        (void)grad_grid; (void)N; (void)H; (void)W; (void)align_corners; (void)dt;
        return Storage{};
    }

    Storage grid_sample_forward(
        const Storage& input, const Storage& grid,
        const Shape& in_shape, const Shape& grid_shape,
        int mode, int padding_mode, bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::grid_sample_forward").not_implemented("not yet in GpuBackend");
        (void)input; (void)grid; (void)in_shape; (void)grid_shape;
        (void)mode; (void)padding_mode; (void)align_corners; (void)dt;
        return Storage{};
    }

    std::vector<Storage> grid_sample_backward(
        const Storage& grad_out, const Storage& input, const Storage& grid,
        const Shape& in_shape, const Shape& grid_shape,
        int mode, int padding_mode, bool align_corners, Dtype dt) override {
        ErrorBuilder("gpu::grid_sample_backward").not_implemented("");
        (void)grad_out; (void)input; (void)grid; (void)in_shape; (void)grid_shape;
        (void)mode; (void)padding_mode; (void)align_corners; (void)dt;
        return {};
    }

    // ---- Vision stubs ---------------------------------------------------

    Storage bilinear_layer_forward(
        const Storage& x1, const Storage& x2, const Storage& weight,
        const Storage& bias, bool has_bias,
        const Shape& x1_shape, const Shape& x2_shape, const Shape& w_shape,
        Dtype dt) override {
        ErrorBuilder("gpu::bilinear_layer_forward").not_implemented("not yet in GpuBackend");
        (void)x1; (void)x2; (void)weight; (void)bias; (void)has_bias;
        (void)x1_shape; (void)x2_shape; (void)w_shape; (void)dt;
        return Storage{};
    }

    std::vector<Storage> bilinear_layer_backward(
        const Storage& grad_out, const Storage& x1, const Storage& x2,
        const Storage& weight,
        const Shape& x1_shape, const Shape& x2_shape, const Shape& w_shape,
        bool has_bias, Dtype dt) override {
        ErrorBuilder("gpu::bilinear_layer_backward").not_implemented("");
        (void)grad_out; (void)x1; (void)x2; (void)weight;
        (void)x1_shape; (void)x2_shape; (void)w_shape; (void)has_bias; (void)dt;
        return {};
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
