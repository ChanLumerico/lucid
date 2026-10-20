// lucid/_C/backend/gpu/HalfAccumulation.h
//
// Float32 accumulation for float16 / bfloat16 reductions on the GPU stream.
//
// MLX accumulates a reduction in its output dtype, and for a float input
// that is the input's dtype: the per-thread accumulator of its sum, prod
// and scan kernels is a ``half`` for a float16 input, and ``mean`` is that
// sum times 1/N.  So a float16 mean of 65536 ones overflowed to inf before
// the division could bring it back, a bfloat16 sum carried 8 bits through
// every partial sum (``ones(100000).mean()`` answered 1.0078), and 1/N is
// itself subnormal in float16 once N passes 16384.  The reference
// accumulates float16 and bfloat16 in float32 and rounds once at the end.
//
// So a reduction over a half input widens it to float32, reduces, and
// narrows the result back: the result dtype is unchanged, and a sum whose
// true value does not fit float16 still comes back inf, as the
// reference's does.  Float32 and every other dtype pass through as they
// were.  Reductions whose result is bounded by construction (softmax's
// backward sums ``z * g`` with ``sum(z) == 1``; grid_sample sums four
// corners) are left alone.
//
// The widening is not free: the cast writes a float32 copy that the
// reduction then reads, 10 bytes moved per element against 2.  Measured
// on an M1 Pro under load (best of 9, chained evaluation, the old and new
// builds alternating): a float16 mean of 4096x1024 went 100 -> 275 us, of
// 64x1024 46 -> 51 us.  Two things win most of it back:
//
//   * ``sum_widened`` is a matrix-vector product with a vector of ones
//     (``half_sum_by_gemv``).  MLX's GEMV accumulates in float32 and
//     rounds once into the half output — the semantics asked for — while
//     reading the input at its own width: a float16 sum of 8192x3072 over
//     axis 0 went 354 -> 199 us, faster than MLX's own half sum.  Axes that
//     do not lead or trail need a transposing copy first: 64x64x56x56 over
//     (0, 2, 3) went 149 -> 511 us (759 widened).
//   * the normalisation layers fold the widening into the element-wise
//     kernels around their reductions (``mlx_fused``).  A float16
//     ``BatchNorm2d`` forward of 32x64x56x56 still costs 1035 -> 1340 us
//     (its mean reads a widened copy); float32 got faster, 1917 -> 1171.
//
// Used by ``GpuBackend`` and by the op kernels that call MLX directly
// (``ProdBackward::gpu_kernel``).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/ops.h>

#include "ComplexParts.h"
#include "MlxBridge.h"

namespace lucid {
namespace backend {

// True for float16 and bfloat16.
inline bool is_half(::mlx::core::Dtype dt) {
    return dt == ::mlx::core::float16 || dt == ::mlx::core::bfloat16;
}

// ``x`` in float32 when it is float16 or bfloat16, else ``x`` itself.
inline ::mlx::core::array widen_half(const ::mlx::core::array& x) {
    return is_half(x.dtype()) ? ::mlx::core::astype(x, ::mlx::core::float32) : x;
}

// ``x`` in ``dt`` — a no-op when it already is.
inline ::mlx::core::array narrow_to(const ::mlx::core::array& x, ::mlx::core::Dtype dt) {
    return x.dtype() == dt ? x : ::mlx::core::astype(x, dt);
}

inline ::mlx::core::array widen_gpu(const Storage& s) {
    return widen_half(*std::get<GpuStorage>(s).arr);
}

inline Storage wrap_narrow(const ::mlx::core::array& a, Dtype dt) {
    return Storage{::lucid::gpu::wrap_mlx_array(narrow_to(a, ::lucid::gpu::to_mlx_dtype(dt)), dt)};
}

// The dtype a reduction over ``dt`` accumulates in.
inline ::mlx::core::Dtype accumulation_dtype(::mlx::core::Dtype dt) {
    return is_half(dt) ? ::mlx::core::float32 : dt;
}

// A result computed from ``widen_half`` of an ``in_dt`` input, back in
// ``in_dt`` when that was a half dtype.  Any other result is returned as
// MLX produced it — a bool sum stays int32, a complex norm stays real.
inline ::mlx::core::array narrow_half(const ::mlx::core::array& r, ::mlx::core::Dtype in_dt) {
    return is_half(in_dt) ? narrow_to(r, in_dt) : r;
}

// ``reduce(x)`` with a half ``x`` accumulated in float32, returned in
// ``x``'s dtype.
template <class Reduce>
::mlx::core::array reduce_widened(const ::mlx::core::array& x, Reduce&& reduce) {
    return narrow_half(reduce(widen_half(x)), x.dtype());
}

// A half ``sum`` as a matrix-vector product with a vector of ones, or
// nothing for a full reduction.  MLX's GEMV keeps a float32 accumulator
// and rounds once into the half output, which is the sum's contract, and
// reads ``x`` at two bytes an element where widening first moves ten.
//
// The reduced axes are brought to the end and the product taken one dot
// product per row (``x`` as M x K, times ones(K)).  When they already end
// ``x`` that is free; otherwise the transpose is a copy, still cheaper than
// widening (NCHW over (0, 2, 3): 497 us against 735).  When they lead and
// there are at least 64 columns the product is taken the other way round
// (ones(K) times ``x`` as K x M) without the copy — with fewer columns
// that form serialises (4 columns of a million rows: 1327 us).  A full
// reduction is a single dot product, slower than widening (313 us against
// 289 for 4M elements), and is left to the widened path.
inline std::optional<::mlx::core::array>
half_sum_by_gemv(const ::mlx::core::array& x, std::vector<int> axes, bool keepdims) {
    const int nd = static_cast<int>(x.ndim());
    if (!is_half(x.dtype()) || x.size() == 0 || axes.empty())
        return std::nullopt;
    for (int& ax : axes)
        ax = ax < 0 ? ax + nd : ax;
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    const int r = static_cast<int>(axes.size());
    if (r >= nd || axes.front() < 0 || axes.back() >= nd)
        return std::nullopt;
    bool leading = true;
    bool trailing = true;
    for (int i = 0; i < r; ++i) {
        leading = leading && axes[static_cast<std::size_t>(i)] == i;
        trailing = trailing && axes[static_cast<std::size_t>(i)] == nd - r + i;
    }
    std::int64_t reduced = 1;
    std::int64_t kept = 1;
    ::mlx::core::Shape out_shape;
    std::vector<int> kept_then_reduced;
    for (int d = 0; d < nd; ++d) {
        const auto extent = x.shape(d);
        if (std::binary_search(axes.begin(), axes.end(), d)) {
            reduced *= extent;
            if (keepdims)
                out_shape.push_back(1);
        } else {
            kept *= extent;
            out_shape.push_back(extent);
            kept_then_reduced.push_back(d);
        }
    }
    kept_then_reduced.insert(kept_then_reduced.end(), axes.begin(), axes.end());
    const auto K = static_cast<::mlx::core::ShapeElem>(reduced);
    const auto M = static_cast<::mlx::core::ShapeElem>(kept);
    if (leading && M >= 64) {
        auto cols = ::mlx::core::matmul(::mlx::core::ones({1, K}, x.dtype()),
                                        ::mlx::core::reshape(x, {K, M}));
        return ::mlx::core::reshape(cols, out_shape);
    }
    if (M < 2)
        return std::nullopt;
    const auto rows_first = trailing ? x : ::mlx::core::transpose(x, kept_then_reduced);
    auto rows = ::mlx::core::matmul(::mlx::core::reshape(rows_first, {M, K}),
                                    ::mlx::core::ones({K, 1}, x.dtype()));
    return ::mlx::core::reshape(rows, out_shape);
}

// ``sum`` accumulated in float32 for a half input, returned in its dtype.
inline ::mlx::core::array
sum_widened(const ::mlx::core::array& x, const std::vector<int>& axes, bool keepdims) {
    if (auto by_gemv = half_sum_by_gemv(x, axes, keepdims))
        return *std::move(by_gemv);
    return reduce_widened(
        x, [&](const ::mlx::core::array& w) { return ::mlx::core::sum(w, axes, keepdims); });
}

// Fuse an element-wise composite with any number of inputs and outputs
// into one Metal kernel — the n-ary form of ``GpuBackend::mlx_unary_fused``,
// with the same rules: ``Fn`` is capture-less and reads any dtype it needs
// from its inputs.  A widened reduction pays for the float32 copy it reads;
// fusing the widening into the element-wise producer of that copy (and the
// narrowing into the consumer of the statistics) is what keeps a half
// normalisation layer as fast as it was.
//
// MLX's shapeless compile gets a broadcast against an empty array wrong —
// (0, 3, 2, 2) minus (1, 3, 1, 1) came back (1, 3, 2, 2) — so a call with
// an empty input runs the composite eagerly.
template <class Fn>
std::vector<::mlx::core::array> mlx_fused(const std::vector<::mlx::core::array>& ins, Fn) {
    for (const auto& a : ins) {
        if (a.size() == 0)
            return Fn{}(ins);
    }
    static const std::function<std::vector<::mlx::core::array>(
        const std::vector<::mlx::core::array>&)>
        compiled = ::mlx::core::compile(
            [](const std::vector<::mlx::core::array>& in) -> std::vector<::mlx::core::array> {
                return Fn{}(in);
            },
            /*shapeless=*/true);
    return compiled(ins);
}

// ``(x - mean)^2`` in float32 for a half ``x`` — the summand of a widened
// variance, produced by one kernel instead of a cast, a subtract and a
// square.  ``mean`` is already in the accumulation dtype.
inline ::mlx::core::array squared_deviation(const ::mlx::core::array& x,
                                            const ::mlx::core::array& mean) {
    return mlx_fused({x, mean}, [](const std::vector<::mlx::core::array>& in) {
        return std::vector<::mlx::core::array>{
            ::mlx::core::square(::mlx::core::subtract(widen_half(in[0]), in[1]))};
    })[0];
}

// ``a * b`` in float32 for half inputs — the summand of a widened
// ``sum(grad * xnorm)``, in one kernel.
inline ::mlx::core::array product_widened(const ::mlx::core::array& a,
                                          const ::mlx::core::array& b) {
    return mlx_fused({a, b}, [](const std::vector<::mlx::core::array>& in) {
        return std::vector<::mlx::core::array>{
            ::mlx::core::multiply(widen_half(in[0]), widen_half(in[1]))};
    })[0];
}

// ``scale * (g - mean_g - xnorm * mean_g_xn)`` — the input gradient every
// normalisation layer ends with — in the accumulation dtype, rounded to
// ``g``'s dtype once, as one kernel.  ``mean_g`` and ``mean_g_xn`` are the
// widened means of ``g`` and ``g * xnorm`` over the normalised axes.
inline ::mlx::core::array normalized_input_grad(const ::mlx::core::array& g,
                                                const ::mlx::core::array& xnorm,
                                                const ::mlx::core::array& mean_g,
                                                const ::mlx::core::array& mean_g_xn,
                                                const ::mlx::core::array& scale) {
    return mlx_fused(
        {g, xnorm, mean_g, mean_g_xn, scale}, [](const std::vector<::mlx::core::array>& in) {
            auto inner = ::mlx::core::subtract(::mlx::core::subtract(widen_half(in[0]), in[2]),
                                               ::mlx::core::multiply(widen_half(in[1]), in[3]));
            return std::vector<::mlx::core::array>{
                narrow_to(::mlx::core::multiply(widen_half(in[4]), inner), in[0].dtype())};
        })[0];
}

// ``scatter_add_axis`` accumulated in float32 for a half input, returned in
// ``base``'s dtype.  MLX's scatter-add kernel accumulates in its operand
// dtype, so 1000 duplicate updates of 1.0 in bfloat16 stalled at 256.0
// (``256.0 + 1.0 == 256.0`` at 7-bit mantissa) and 3000 in float16 stalled at
// 2048.0.  The kernel has no complex64 form, so a complex base is summed one
// part at a time.
inline ::mlx::core::array scatter_add_axis_widened(const ::mlx::core::array& base,
                                                   const ::mlx::core::array& idx,
                                                   const ::mlx::core::array& updates,
                                                   int axis) {
    namespace mx = ::mlx::core;
    if (base.dtype() == mx::complex64)
        return by_complex_parts(base, updates, [&](const auto& b, const auto& u) {
            return mx::scatter_add_axis(b, idx, u, axis);
        });
    return narrow_half(mx::scatter_add_axis(widen_half(base), idx, widen_half(updates), axis),
                       base.dtype());
}

// Pooled bag ``sum`` (``mode == 0``) or ``mean`` (``mode == 1``) for
// ``embedding_bag_forward``, accumulated (and, for ``mean``, divided by the
// widened bag count) in float32 when ``emb`` is a half dtype, then rounded
// back to ``emb``'s dtype once.
inline ::mlx::core::array embedding_bag_sum_or_mean(const ::mlx::core::array& emb,
                                                    const ::mlx::core::array& seg,
                                                    const ::mlx::core::array& valid,
                                                    const ::mlx::core::array& oh,
                                                    int B,
                                                    int D,
                                                    int n_idx,
                                                    int mode) {
    namespace mx = ::mlx::core;
    const auto mdt = emb.dtype();
    const auto acc_dt = accumulation_dtype(mdt);
    auto keep_row = mx::broadcast_to(
        mx::reshape(mx::greater(valid, mx::array(0.0f, mdt)), {n_idx, 1}), {n_idx, D});
    auto contributions = mx::where(keep_row, widen_half(emb), mx::zeros({n_idx, D}, acc_dt));
    auto seg_col = mx::broadcast_to(mx::reshape(seg, {n_idx, 1}), {n_idx, D});
    auto out = mx::scatter_add_axis(mx::zeros({B, D}, acc_dt), mx::astype(seg_col, mx::int32),
                                    contributions, 0);
    if (mode == 1) {
        auto sizes = mx::reshape(mx::sum(widen_half(oh), std::vector<int>{1}, false), {B, 1});
        auto denom = mx::broadcast_to(mx::maximum(sizes, mx::array(1.0f, acc_dt)), {B, D});
        out = mx::divide(out, denom);
    }
    return narrow_half(out, mdt);
}

}  // namespace backend
}  // namespace lucid
