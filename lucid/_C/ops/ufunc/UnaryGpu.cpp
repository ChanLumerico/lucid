// =====================================================================
// Lucid C++ engine — GPU kernels for the unary op family (Phase 3.7.2).
// =====================================================================
//
// Centralizing all `*Backward::gpu_kernel` definitions in one TU keeps
// the per-op .cpp files focused on CPU kernels and grad formulas while
// the MLX bridge include cost is paid once.

#include <cmath>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "Activation.h"
#include "Arith.h"
#include "Discrete.h"
#include "Exponential.h"
#include "Hyperbolic.h"
#include "Trig.h"

namespace lucid {

namespace {

// Single-input GPU helper — applies `f` to the input MLX array, then wraps
// the result with MemoryTracker accounting via gpu::wrap_mlx_array.
template <class F>
GpuStorage gpu_apply(const GpuStorage& a, Dtype dt, F&& f, const char* op) {
    if (!a.arr) {
        ErrorBuilder(op).fail("null GPU input");
    }
    auto out = f(*a.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}

}  // namespace

// ----------------- Arith -----------------
GpuStorage NegBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::negative(x); }, "neg");
}
GpuStorage AbsBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::abs(x); }, "abs");
}
GpuStorage SignBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::sign(x); }, "sign");
}
GpuStorage ReciprocalBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::reciprocal(x); }, "reciprocal");
}
GpuStorage SquareBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::square(x); }, "square");
}
GpuStorage CubeBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt, [](const auto& x) { return ::mlx::core::multiply(::mlx::core::square(x), x); },
        "cube");
}

// ----------------- Exponential -----------------
GpuStorage ExpBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::exp(x); }, "exp");
}
GpuStorage LogBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::log(x); }, "log");
}
GpuStorage Log2Backward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::log2(x); }, "log2");
}
GpuStorage SqrtBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::sqrt(x); }, "sqrt");
}

// ----------------- Trig -----------------
GpuStorage SinBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::sin(x); }, "sin");
}
GpuStorage CosBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::cos(x); }, "cos");
}
GpuStorage TanBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::tan(x); }, "tan");
}
GpuStorage AsinBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::arcsin(x); }, "arcsin");
}
GpuStorage AcosBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::arccos(x); }, "arccos");
}
GpuStorage AtanBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::arctan(x); }, "arctan");
}

// ----------------- Hyperbolic -----------------
GpuStorage SinhBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::sinh(x); }, "sinh");
}
GpuStorage CoshBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::cosh(x); }, "cosh");
}
GpuStorage TanhBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::tanh(x); }, "tanh");
}

// ----------------- Activation -----------------
GpuStorage ReluBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            return ::mlx::core::maximum(x, zero);
        },
        "relu");
}
GpuStorage SigmoidBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::sigmoid(x); }, "sigmoid");
}
GpuStorage SiluBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt, [](const auto& x) { return ::mlx::core::multiply(x, ::mlx::core::sigmoid(x)); },
        "silu");
}
GpuStorage GeluBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    // tanh-approximation GeLU:
    //   y = 0.5·x·(1 + tanh( √(2/π) · (x + 0.044715·x³) ))
    // Force F32 inside (AMP::ForceFP32 policy on the op).
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
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
        },
        "gelu");
}
GpuStorage SoftplusBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    // softplus(x) = log(1 + exp(x)). Numerically stable form:
    //   max(x, 0) + log1p(exp(-|x|))
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto log1p = ::mlx::core::log1p(::mlx::core::exp(neg_abs));
            return ::mlx::core::add(pos, log1p);
        },
        "softplus");
}
GpuStorage SeluBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
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
        },
        "selu");
}
GpuStorage MishBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    // y = x * tanh(softplus(x))
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
            return ::mlx::core::multiply(x, ::mlx::core::tanh(sp));
        },
        "mish");
}
GpuStorage HardSigmoidBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
            return ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
        },
        "hard_sigmoid");
}
GpuStorage HardSwishBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
            auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
            auto h = ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
            return ::mlx::core::multiply(x, h);
        },
        "hard_swish");
}
GpuStorage Relu6Backward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
            ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
            return ::mlx::core::minimum(::mlx::core::maximum(x, zero), six);
        },
        "relu6");
}

// ----------------- Discrete (no-grad) -----------------
GpuStorage RoundBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt, [](const auto& x) { return ::mlx::core::round(x, /*decimals=*/0); }, "round");
}
GpuStorage FloorBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::floor(x); }, "floor");
}
GpuStorage CeilBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(a, dt, [](const auto& x) { return ::mlx::core::ceil(x); }, "ceil");
}
GpuStorage InvertBackward::gpu_kernel(const GpuStorage& a, const Shape&, Dtype dt) {
    return gpu_apply(
        a, dt,
        [dt](const auto& x) {
            if (dt == Dtype::Bool) {
                return ::mlx::core::logical_not(x);
            }
            return ::mlx::core::bitwise_invert(x);
        },
        "invert");
}

}  // namespace lucid
