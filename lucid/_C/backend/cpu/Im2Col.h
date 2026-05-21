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

// Single-precision 1-D im2col unfold.
//
// Unfolds each length-$KL$ receptive field of a 1-D feature map into a
// column of a 2-D matrix so that conv1d can be expressed as a single GEMM.
// Out-of-bounds positions (from padding or dilation) are written as zero.
//
// Parameters
// ----------
// x : const float*
//     Input tensor, shape $(C, L)$ (row-major).
// cols : float*
//     Output column matrix, shape $(C \cdot KL, OL)$.
// C : int
//     Channel count.
// L : int
//     Input spatial length.
// KL : int
//     Kernel length.
// OL : int
//     Output spatial length.
// stride_l, pad_l, dilation_l : int
//     Convolution stride / zero-padding / dilation along the spatial axis.
//
// Math
// ----
// For each output index $o \in [0, OL)$, channel $c$ and kernel offset $k$:
// $$ \text{cols}[c \cdot KL + k, o] = x[c, o \cdot \text{stride} + k \cdot \text{dilation} - \text{pad}] $$
// with zero for out-of-bounds positions.
//
// Notes
// -----
// Memory layout assumes the standard NCHW-style channel-first convention,
// with the batch dimension handled at the caller (loop over $N$).
LUCID_INTERNAL void im2col_1d_f32(const float* x,
                                  float* cols,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);

// Double-precision 1-D im2col unfold.  See ``im2col_1d_f32``.
//
// Parameters
// ----------
// x : const double*
//     Input tensor, shape $(C, L)$.
// cols : double*
//     Output column matrix, shape $(C \cdot KL, OL)$.
// C, L, KL, OL : int
//     Channel count, input length, kernel length, output length.
// stride_l, pad_l, dilation_l : int
//     Convolution stride / padding / dilation.
LUCID_INTERNAL void im2col_1d_f64(const double* x,
                                  double* cols,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);

// Single-precision 1-D col2im fold (gradient transform).
//
// Inverse of ``im2col_1d_f32``: scatter-adds each column of ``cols`` back
// into the corresponding positions of $dx$.  Overlapping receptive fields
// (stride < kernel) accumulate by sum — this is the mathematically correct
// adjoint that the conv1d backward needs.
//
// Parameters
// ----------
// cols : const float*
//     Column matrix, shape $(C \cdot KL, OL)$.
// dx : float*
//     Output gradient tensor, shape $(C, L)$; **must be zeroed by the
//     caller** before this call.
// C, L, KL, OL : int
//     Channel count, input length, kernel length, output length.
// stride_l, pad_l, dilation_l : int
//     Match the convolution forward.
//
// Math
// ----
// $$ dx[c, l] = \sum_{o, k : o \cdot s + k \cdot d - p = l} \text{cols}[c \cdot KL + k, o] $$
LUCID_INTERNAL void col2im_1d_f32(const float* cols,
                                  float* dx,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);

// Double-precision 1-D col2im fold.  See ``col2im_1d_f32``.
//
// Parameters
// ----------
// cols : const double*
//     Column matrix.
// dx : double*
//     Output gradient tensor; **must be pre-zeroed**.
// C, L, KL, OL : int
//     Shape parameters.
// stride_l, pad_l, dilation_l : int
//     Convolution geometry.
LUCID_INTERNAL void col2im_1d_f64(const double* cols,
                                  double* dx,
                                  int C,
                                  int L,
                                  int KL,
                                  int OL,
                                  int stride_l,
                                  int pad_l,
                                  int dilation_l);

// Single-precision 2-D im2col unfold.
//
// Unfolds the 2-D feature map $x$ of shape $(C, H, W)$ into a matrix where
// each column is a $K_H \times K_W$ receptive-field patch flattened across
// channels.  The output column matrix has shape $(C \cdot K_H \cdot K_W,
// O_H \cdot O_W)$ so that conv2d = ``sgemm(W_reshape, cols)``.
//
// Out-of-bounds positions (those reaching outside the input due to padding,
// stride, or dilation) are written as zero.
//
// Parameters
// ----------
// x : const float*
//     Input tensor, shape $(C, H, W)$ row-major.
// cols : float*
//     Output column matrix, shape $(C \cdot K_H \cdot K_W, O_H \cdot O_W)$.
// C : int
//     Channel count.
// H, W : int
//     Input spatial dimensions.
// KH, KW : int
//     Kernel height and width.
// OH, OW : int
//     Output spatial dimensions.
// stride_h, stride_w : int
//     Per-axis stride.
// pad_h, pad_w : int
//     Per-axis zero-padding.
// dilation_h, dilation_w : int
//     Per-axis dilation.
//
// Math
// ----
// For each output position $(o_h, o_w)$, kernel offset $(k_h, k_w)$ and
// channel $c$, the corresponding input position is
// $$ i_h = o_h \cdot s_h + k_h \cdot d_h - p_h, \quad i_w = o_w \cdot s_w + k_w \cdot d_w - p_w $$
// and ``cols[c \cdot K_H K_W + k_h K_W + k_w, o_h \cdot O_W + o_w] = x[c, i_h, i_w]`` (zero if OOB).
//
// Notes
// -----
// Memory layout is NCHW (channel-first) — the batch dimension is iterated
// at the caller, with one im2col per batch element.
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

// Double-precision 2-D im2col unfold.  See ``im2col_f32``.
//
// Parameters
// ----------
// x : const double*
//     Input tensor, shape $(C, H, W)$.
// cols : double*
//     Output column matrix.
// C, H, W, KH, KW, OH, OW : int
//     Shape parameters.
// stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w : int
//     Convolution geometry.
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

// Single-precision 2-D col2im fold (gradient transform).
//
// Adjoint of ``im2col_f32``: scatters each receptive-field column of ``cols``
// back into the corresponding input positions of $dx$, accumulating by
// addition where windows overlap.  Used in the conv2d backward pass.
//
// Parameters
// ----------
// cols : const float*
//     Column matrix, shape $(C \cdot K_H \cdot K_W, O_H \cdot O_W)$.
// dx : float*
//     Output gradient tensor, shape $(C, H, W)$; **must be zeroed by the
//     caller** before this call.
// C, H, W, KH, KW, OH, OW : int
//     Shape parameters.
// stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w : int
//     Match the forward convolution.
//
// Notes
// -----
// Overlapping receptive fields accumulate by sum — this is the
// mathematically correct adjoint of im2col under any stride / dilation.
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

// Double-precision 2-D col2im fold.  See ``col2im_f32``.
//
// Parameters
// ----------
// cols : const double*
//     Column matrix.
// dx : double*
//     Output gradient tensor; **must be pre-zeroed**.
// C, H, W, KH, KW, OH, OW : int
//     Shape parameters.
// stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w : int
//     Convolution geometry.
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

// Single-precision 3-D im2col unfold.
//
// Unfolds a volumetric feature map $x$ of shape $(C, D, H, W)$ into a
// column matrix of shape $(C \cdot K_D \cdot K_H \cdot K_W, O_D \cdot O_H
// \cdot O_W)$.  Used by conv3d (e.g., for video / medical-imaging models).
// Out-of-bounds positions are zero-padded.
//
// Parameters
// ----------
// x : const float*
//     Input tensor, shape $(C, D, H, W)$ row-major.
// cols : float*
//     Output column matrix.
// C : int
//     Channel count.
// D, H, W : int
//     Input depth / height / width.
// KD, KH, KW : int
//     Kernel depth / height / width.
// OD, OH, OW : int
//     Output depth / height / width.
// stride_d, stride_h, stride_w : int
//     Per-axis strides.
// pad_d, pad_h, pad_w : int
//     Per-axis zero-padding.
// dilation_d, dilation_h, dilation_w : int
//     Per-axis dilation.
//
// Notes
// -----
// Memory layout is channel-first (NCDHW); batch dimension is iterated by
// the caller.
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

// Double-precision 3-D im2col unfold.  See ``im2col_3d_f32``.
//
// Parameters
// ----------
// x : const double*
//     Input tensor, shape $(C, D, H, W)$.
// cols : double*
//     Output column matrix.
// C, D, H, W, KD, KH, KW, OD, OH, OW : int
//     Shape parameters.
// stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w : int
//     Convolution geometry.
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

// Single-precision 3-D col2im fold (gradient transform).
//
// Adjoint of ``im2col_3d_f32``: scatter-accumulates the column-major
// gradient back into the volumetric input gradient $dx$.  Used in the
// conv3d backward pass.
//
// Parameters
// ----------
// cols : const float*
//     Column matrix, shape $(C \cdot K_D \cdot K_H \cdot K_W, O_D \cdot O_H \cdot O_W)$.
// dx : float*
//     Output gradient tensor, shape $(C, D, H, W)$; **must be zeroed by the
//     caller** before this call.
// C, D, H, W, KD, KH, KW, OD, OH, OW : int
//     Shape parameters.
// stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w : int
//     Match the forward convolution.
//
// Notes
// -----
// Overlapping receptive fields accumulate by sum.
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

// Double-precision 3-D col2im fold.  See ``col2im_3d_f32``.
//
// Parameters
// ----------
// cols : const double*
//     Column matrix.
// dx : double*
//     Output gradient tensor; **must be pre-zeroed**.
// C, D, H, W, KD, KH, KW, OD, OH, OW : int
//     Shape parameters.
// stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w : int
//     Convolution geometry.
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
