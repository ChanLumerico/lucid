#pragma once

// =====================================================================
// Lucid C++ engine — im2col / col2im for 1D, 2D, 3D convolution.
// =====================================================================
//
// Per-batch im2col: input x[b] has shape (C, *S) where S is the spatial
// extent for the chosen rank; output cols has shape (C·prod(K), prod(O)).
// The matmul `W_2d @ cols` then yields (C_out, prod(O)) which reshapes
// directly to (C_out, *O).
//
// col2im is the gradient transpose: scatter (C·prod(K), prod(O)) gradient
// back into (C, *S), accumulating values that came from overlapping kernel
// windows.
//
// Dilation: kernel-axis offset is `kx * dilation_x` so that effective
// kernel span in input space is `dilation_x * (K_x - 1) + 1`. Caller is
// responsible for computing OL/OH/... using that effective span.
//
// All three ranks share the same column layout (channel-major over the
// kernel's flattened multi-index, then the output's flattened multi-index)
// and the same flat-stride convention. Loops are explicitly nested per
// rank so the inner-most stride is contiguous in memory.
//
// Layer: backend/cpu/. F32 + F64 only.

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// ------------------------ 1D ------------------------

LUCID_INTERNAL void im2col_1d_f32(const float* x, float* cols,
                                  int C, int L, int KL, int OL,
                                  int stride_l, int pad_l, int dilation_l);
LUCID_INTERNAL void im2col_1d_f64(const double* x, double* cols,
                                  int C, int L, int KL, int OL,
                                  int stride_l, int pad_l, int dilation_l);
LUCID_INTERNAL void col2im_1d_f32(const float* cols, float* dx,
                                  int C, int L, int KL, int OL,
                                  int stride_l, int pad_l, int dilation_l);
LUCID_INTERNAL void col2im_1d_f64(const double* cols, double* dx,
                                  int C, int L, int KL, int OL,
                                  int stride_l, int pad_l, int dilation_l);

// ------------------------ 2D ------------------------

LUCID_INTERNAL void im2col_f32(const float* x, float* cols,
                               int C, int H, int W,
                               int KH, int KW, int OH, int OW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w);
LUCID_INTERNAL void im2col_f64(const double* x, double* cols,
                               int C, int H, int W,
                               int KH, int KW, int OH, int OW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w);

/// `dx` is zero-filled by caller; this function ACCUMULATES into it.
LUCID_INTERNAL void col2im_f32(const float* cols, float* dx,
                               int C, int H, int W,
                               int KH, int KW, int OH, int OW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w);
LUCID_INTERNAL void col2im_f64(const double* cols, double* dx,
                               int C, int H, int W,
                               int KH, int KW, int OH, int OW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w);

// ------------------------ 3D ------------------------

LUCID_INTERNAL void im2col_3d_f32(const float* x, float* cols,
                                  int C, int D, int H, int W,
                                  int KD, int KH, int KW,
                                  int OD, int OH, int OW,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dilation_d, int dilation_h, int dilation_w);
LUCID_INTERNAL void im2col_3d_f64(const double* x, double* cols,
                                  int C, int D, int H, int W,
                                  int KD, int KH, int KW,
                                  int OD, int OH, int OW,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dilation_d, int dilation_h, int dilation_w);
LUCID_INTERNAL void col2im_3d_f32(const float* cols, float* dx,
                                  int C, int D, int H, int W,
                                  int KD, int KH, int KW,
                                  int OD, int OH, int OW,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dilation_d, int dilation_h, int dilation_w);
LUCID_INTERNAL void col2im_3d_f64(const double* cols, double* dx,
                                  int C, int D, int H, int W,
                                  int KD, int KH, int KW,
                                  int OD, int OH, int OW,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dilation_d, int dilation_h, int dilation_w);

}  // namespace lucid::backend::cpu
