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

private:
    // ---- Helpers -------------------------------------------------------

    inline static const ::mlx::core::Device k_linalg_stream{::mlx::core::Device::cpu};

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
