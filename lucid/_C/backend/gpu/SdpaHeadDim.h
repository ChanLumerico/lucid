// lucid/_C/backend/gpu/SdpaHeadDim.h
//
// MLX's fused attention, reached at every head dim up to 128.
//
// MLX 0.32.3's fast::scaled_dot_product_attention has a fused kernel for
// more than 8 queries only at head dims 64, 72, 80, 96 and 128, and only
// when q and v share the width.  At any other width it silently runs its
// unfused fallback (q·kᵀ, softmax, ·v), which holds the (B, H, Lq, Lk)
// float scores: 17.3 MB above the inputs at B·H = 4, L = 1024, D = 16, for
// a 0.26 MB output (LCD-496).
//
// Zero columns leave q·kᵀ, and so the softmax, unchanged, and give zero
// output columns.  So padding q, k and v to the next fused width and
// slicing the output back is exact: bit for bit the fused kernel run on
// operands padded by hand.  The scale is the caller's, set from the
// unpadded width.
//
// The padded copies of q, k, v and the output hold 2·(Lq + Lk)·width
// elements per head.  Padding is taken once the scores it avoids are at
// least twice that.  Below the cutoff the scores are under
// 4·(Lq + Lk)·width, so the forward's extra memory is linear in the
// lengths on both sides of it.  Short calls also stay unfused because
// there the fallback's matmuls win: non-causal at L = 256 the padded call
// was 1.2-1.3x slower, and at the cutoff (L = 8·width) within 10%, while
// past it causal runs 0.4-0.6x (LCD-498).  The cutoff also implies MLX's
// own condition of more than 8 queries.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

namespace lucid::gpu {

// The smallest fused head dim holding both q's and v's width, or 0.
inline int sdpa_fused_width(int dq, int dv) {
    constexpr std::array<int, 5> kFused{64, 72, 80, 96, 128};
    const int need = std::max(dq, dv);
    for (int w : kFused)
        if (w >= need)
            return w;
    return 0;
}

inline ::mlx::core::array sdpa_fused(const ::mlx::core::array& q,
                                     const ::mlx::core::array& k,
                                     const ::mlx::core::array& v,
                                     float scale,
                                     const std::string& mask_mode,
                                     const std::optional<::mlx::core::array>& mask) {
    namespace mx = ::mlx::core;
    const std::int64_t lq = q.shape(-2);
    const std::int64_t lk = k.shape(-2);
    const int dv = v.shape(-1);
    const int width = sdpa_fused_width(q.shape(-1), dv);
    const bool pad =
        width != 0 && (q.shape(-1) != width || dv != width) && lq * lk >= 4 * (lq + lk) * width;
    if (!pad)
        return mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask);

    auto widen = [width](const mx::array& a) {
        return mx::pad(a, {static_cast<int>(a.ndim()) - 1}, {0}, {width - a.shape(-1)});
    };
    auto out = mx::fast::scaled_dot_product_attention(widen(q), widen(k), widen(v), scale,
                                                      mask_mode, mask);
    mx::Shape stop = out.shape();
    stop.back() = dv;
    return mx::contiguous(mx::slice(out, mx::Shape(out.ndim(), 0), std::move(stop)));
}

}  // namespace lucid::gpu
