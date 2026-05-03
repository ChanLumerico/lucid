// lucid/_C/backend/cpu/Im2Col.h
//
// Im2col and col2im transforms for 1-D, 2-D, and 3-D convolutions on the CPU.
// im2col ("image to column") unfolds each receptive-field patch from the input
// feature map into a column of a matrix; the resulting matrix has shape
// (C*K, O) where C is input channels, K is the kernel spatial size, and O is
// the total number of output positions.  A single cblas_sgemm call on that
// column matrix then computes the full convolution forward pass.
//
// col2im is the adjoint (transposed) operation used in backward passes: it
// folds ("scatters-accumulates") the gradient columns back into the input
// gradient tensor of the original spatial shape.  Overlapping receptive fields
// are accumulated by addition, which is why col2im is the correct gradient
// transform even under stride > 1 or dilation > 1.
//
// Each spatial dimensionality (1-D, 2-D, 3-D) has separate f32 and f64
// entry points; they all delegate to typed template instantiations in the
// corresponding .cpp file.

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// 1-D im2col: unfolds input x of shape (C, L) into cols of shape (C*KL, OL).
// stride_l/pad_l/dilation_l follow the standard convolution convention.
LUCID_INTERNAL void im2col_1d_f32(const float* x,
                                  float* cols,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);
LUCID_INTERNAL void im2col_1d_f64(const double* x,
                                  double* cols,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);
// 1-D col2im: scatters-accumulates cols (shape C*KL, OL) back into dx (shape C, L).
// dx must be zeroed by the caller before col2im is invoked.
LUCID_INTERNAL void col2im_1d_f32(const float* cols,
                                  float* dx,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);
LUCID_INTERNAL void col2im_1d_f64(const double* cols,
                                  double* dx,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);

// 2-D im2col: unfolds x of shape (C, H, W) into cols of shape (C*KH*KW, OH*OW).
// Out-of-bounds receptive-field positions are zero-padded (pad_h, pad_w).
LUCID_INTERNAL void im2col_f32(const float* x,
                               float* cols,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int stride_h,
                               int stride_w,
                               int pad_h,
                               int pad_w,
                               int dilation_h,
                               int dilation_w);
LUCID_INTERNAL void im2col_f64(const double* x,
                               double* cols,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int stride_h,
                               int stride_w,
                               int pad_h,
                               int pad_w,
                               int dilation_h,
                               int dilation_w);

// 2-D col2im: scatters-accumulates cols back into dx (shape C, H, W).
// dx must be zeroed before calling.
LUCID_INTERNAL void col2im_f32(const float* cols,
                               float* dx,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int stride_h,
                               int stride_w,
                               int pad_h,
                               int pad_w,
                               int dilation_h,
                               int dilation_w);
LUCID_INTERNAL void col2im_f64(const double* cols,
                               double* dx,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int stride_h,
                               int stride_w,
                               int pad_h,
                               int pad_w,
                               int dilation_h,
                               int dilation_w);

// 3-D im2col: unfolds x of shape (C, D, H, W) into cols of shape
// (C*KD*KH*KW, OD*OH*OW).  Out-of-bounds positions are zero-padded.
LUCID_INTERNAL void im2col_3d_f32(const float* x,
                                  float* cols,
                                  int C,
                                  int D,
                                  int H,
                                  int W,
                                  int KD,
                                  int KH,
                                  int KW,
                                  int OD,
                                  int OH,
                                  int OW,
                                  int stride_d,
                                  int stride_h,
                                  int stride_w,
                                  int pad_d,
                                  int pad_h,
                                  int pad_w,
                                  int dilation_d,
                                  int dilation_h,
                                  int dilation_w);
LUCID_INTERNAL void im2col_3d_f64(const double* x,
                                  double* cols,
                                  int C,
                                  int D,
                                  int H,
                                  int W,
                                  int KD,
                                  int KH,
                                  int KW,
                                  int OD,
                                  int OH,
                                  int OW,
                                  int stride_d,
                                  int stride_h,
                                  int stride_w,
                                  int pad_d,
                                  int pad_h,
                                  int pad_w,
                                  int dilation_d,
                                  int dilation_h,
                                  int dilation_w);
// 3-D col2im: scatters-accumulates cols back into dx (shape C, D, H, W).
// dx must be zeroed before calling.
LUCID_INTERNAL void col2im_3d_f32(const float* cols,
                                  float* dx,
                                  int C,
                                  int D,
                                  int H,
                                  int W,
                                  int KD,
                                  int KH,
                                  int KW,
                                  int OD,
                                  int OH,
                                  int OW,
                                  int stride_d,
                                  int stride_h,
                                  int stride_w,
                                  int pad_d,
                                  int pad_h,
                                  int pad_w,
                                  int dilation_d,
                                  int dilation_h,
                                  int dilation_w);
LUCID_INTERNAL void col2im_3d_f64(const double* cols,
                                  double* dx,
                                  int C,
                                  int D,
                                  int H,
                                  int W,
                                  int KD,
                                  int KH,
                                  int KW,
                                  int OD,
                                  int OH,
                                  int OW,
                                  int stride_d,
                                  int stride_h,
                                  int stride_w,
                                  int pad_d,
                                  int pad_h,
                                  int pad_w,
                                  int dilation_d,
                                  int dilation_h,
                                  int dilation_w);

}  // namespace lucid::backend::cpu
