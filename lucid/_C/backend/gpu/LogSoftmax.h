// lucid/_C/backend/gpu/LogSoftmax.h
//
// ``log_softmax`` on the GPU stream, as ``x - logsumexp(x)`` along an axis.
//
// It was ``log(softmax(x))``, which underflows: a probability below the
// dtype's smallest subnormal is 0 and its log is -inf, so float32 answered
// -inf at a logit 104 below the row's maximum and float16 at 17, where the
// reference gives the finite difference.  A half input is reduced in
// float32.  MLX's logsumexp refuses an empty array ("Received empty
// array"), whose log_softmax is just as empty.
//
// MLX's logsumexp answers an infinite row maximum with that infinity, so a
// row holding +inf came out as -inf at its finite entries (LCD-286).  The
// reference, and the CPU kernel, which shifts by the maximum, give NaN
// across the whole row: x - m is inf - inf there.  An infinite logsumexp is
// only reachable that way (it is at most max + log(n)), so it is poisoned
// to NaN on the reduced array before the subtract — two elementwise ops on
// the reduced shape, leaving MLX's fused logsumexp in place.
#pragma once

#include <limits>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "HalfAccumulation.h"

namespace lucid::gpu {

inline ::mlx::core::array log_softmax_along(const ::mlx::core::array& x, int axis) {
    namespace mx = ::mlx::core;
    return backend::reduce_widened(x, [axis](const mx::array& w) {
        if (w.size() == 0)
            return w;
        auto lse = mx::logsumexp(w, axis, /*keepdims=*/true);
        const mx::array nan(std::numeric_limits<float>::quiet_NaN(), lse.dtype());
        return mx::subtract(w, mx::where(mx::isinf(lse), nan, lse));
    });
}

}  // namespace lucid::gpu
