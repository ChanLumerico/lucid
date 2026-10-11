// lucid/_C/backend/gpu/Conversion.h
//
// Converting a value to another dtype on the MLX (Metal) stream, by the
// rules the CPU stream and the reference follow.  MLX's own conversion
// differs from them in two places:
//
//   * a float cast to int64 gave 0 for +-inf and wrapped past the range,
//     where the CPU gives 0 for NaN and saturates at the type's limits
//     (its int32 conversion already saturates);
//   * a complex value cast to bool read only the real part, so 2j was
//     False (LCD-408).  A complex number is true when either part is
//     nonzero, and every real target keeps the real part.
//
// ``any`` and ``all`` judge a complex element's truth by the same rule
// (``truth_operand``).

#pragma once

#include <cstdint>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/Dtype.h"
#include "MlxBridge.h"

namespace lucid {
namespace backend {

// Whether each element of ``x`` is nonzero, as bool.
inline ::mlx::core::array gpu_truth(const ::mlx::core::array& x) {
    namespace mx = ::mlx::core;
    if (x.dtype() == mx::complex64)
        return mx::logical_or(mx::not_equal(mx::real(x), mx::array(0.0f)),
                              mx::not_equal(mx::imag(x), mx::array(0.0f)));
    return mx::astype(x, mx::bool_);
}

// What a logical reduction reads for ``x``: its truth when it is complex,
// which MLX's reductions judge by the real part alone; ``x`` otherwise.
inline ::mlx::core::array truth_operand(const ::mlx::core::array& x) {
    return x.dtype() == ::mlx::core::complex64 ? gpu_truth(x) : x;
}

// ``x`` converted to ``dst_dt``.
inline ::mlx::core::array gpu_convert(const ::mlx::core::array& x, Dtype dst_dt) {
    namespace mx = ::mlx::core;
    const auto dst = gpu::to_mlx_dtype(dst_dt);
    if (dst_dt == Dtype::Bool)
        return gpu_truth(x);
    if (x.dtype() == mx::complex64 && dst != mx::complex64)
        return gpu_convert(mx::real(x), dst_dt);
    auto y = mx::astype(x, dst);
    if (dst_dt != Dtype::I64 || !mx::issubdtype(x.dtype(), mx::floating))
        return y;
    const auto hi = mx::array(9.223372036854775808e18f, x.dtype());  // 2^63
    const auto lo = mx::array(-9.223372036854775808e18f, x.dtype());
    y = mx::where(mx::greater_equal(x, hi), mx::array(std::int64_t{INT64_MAX}, mx::int64), y);
    y = mx::where(mx::less(x, lo), mx::array(std::int64_t{INT64_MIN}, mx::int64), y);
    return mx::where(mx::isnan(x), mx::array(std::int64_t{0}, mx::int64), y);
}

}  // namespace backend
}  // namespace lucid
