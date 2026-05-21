// lucid/_C/kernel/primitives/Im2Col.h
//
// Re-export of the CPU im2col / col2im convolution transforms into the
// kernel primitives namespace.
//
// im2col unfolds each receptive-field patch of a 1-D / 2-D / 3-D
// feature map into a column of a flat matrix, so that the full
// convolution forward pass reduces to a single GEMM call.  col2im is
// its adjoint, scatter-accumulating gradient columns back into the
// original spatial layout — overlapping receptive fields sum, which is
// the correct gradient transform under any stride or dilation.
//
// Math
// ----
// For 2-D convolution with stride $s$, padding $p$, and dilation $d$:
// $$
//   \text{cols}[c \cdot K_H K_W + k_h K_W + k_w,\, o_h \cdot O_W + o_w]
//   = x[c,\, o_h \cdot s_h + k_h \cdot d_h - p_h,\,
//          o_w \cdot s_w + k_w \cdot d_w - p_w]
// $$
// with zero for out-of-bounds positions.  ``col2im`` is the linear
// adjoint of this map.
//
// Notes
// -----
// The actual function declarations and their full per-rank docstrings
// live in :file:`backend/cpu/Im2Col.h`.  This header exists so that
// :file:`ops/` convolution kernels can include a single short path
// (``#include "primitives/Im2Col.h"``) instead of reaching directly
// into the backend subtree.
//
// See Also
// --------
// backend::cpu::im2col_f32 / im2col_f64 : 2-D forward unfold.
// backend::cpu::col2im_f32 / col2im_f64 : 2-D backward fold.
// backend::cpu::im2col_1d_* / im2col_3d_* : 1-D and 3-D variants.

#pragma once

#include "../../backend/cpu/Im2Col.h"
