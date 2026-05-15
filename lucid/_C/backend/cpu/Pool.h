// lucid/_C/backend/cpu/Pool.h
//
// CPU pooling kernels for 1-D, 2-D, and 3-D spatial inputs.  Each kernel
// operates on a batch of NCHW/NCD HW/NCDHW tensors and uses a fixed-size
// sliding window with stride and symmetric padding.
//
// MaxPool forward stores the flat per-channel spatial index of the maximum
// element in the argmax buffer; this is required by the backward pass to
// route gradients to the correct position.  The argmax dtype is always I32.
//
// AvgPool uses count-include-pad semantics: the divisor is always KH*KW
// (or KL, KD*KH*KW for 1-D and 3-D), regardless of how many pad positions
// fall inside the window.
//
// All public entry points are f32/f64 wrappers over typed template
// instantiations defined in Pool.cpp.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// 1-D max pooling forward.  x has shape (B, C, L); y has shape (B, C, OL).
// argmax[i] is the flat index within the L dimension of the winning input.
LUCID_INTERNAL void max_pool1d_forward_f32(const float* x,
                                           float* y,
                                           std::int32_t* argmax,
                                           int B,
                                           int C,
                                           int L,
                                           int KL,
                                           int OL,
                                           int stride_l,
                                           int pad_l);
LUCID_INTERNAL void max_pool1d_forward_f64(const double* x,
                                           double* y,
                                           std::int32_t* argmax,
                                           int B,
                                           int C,
                                           int L,
                                           int KL,
                                           int OL,
                                           int stride_l,
                                           int pad_l);
// 1-D max pooling backward.  Scatters g to dx[argmax[i]] for each output i.
// dx must be zeroed by the caller.
LUCID_INTERNAL void max_pool1d_backward_f32(
    const float* g, const std::int32_t* argmax, float* dx, int B, int C, int L, int OL);
LUCID_INTERNAL void max_pool1d_backward_f64(
    const double* g, const std::int32_t* argmax, double* dx, int B, int C, int L, int OL);
// 1-D average pooling forward: count-include-pad, divisor = KL.
LUCID_INTERNAL void avg_pool1d_forward_f32(
    const float* x, float* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
LUCID_INTERNAL void avg_pool1d_forward_f64(
    const double* x, double* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
// 1-D average pooling backward: distributes g uniformly over each window.
// dx must be zeroed by the caller.
LUCID_INTERNAL void avg_pool1d_backward_f32(
    const float* g, float* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
LUCID_INTERNAL void avg_pool1d_backward_f64(
    const double* g, double* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// 2-D max pooling forward.  x has shape (B, C, H, W); y has shape (B, C, OH, OW).
// argmax[i] is the flat HW index of the maximum element for each output position.
LUCID_INTERNAL void max_pool2d_forward_f32(const float* x,
                                           float* y,
                                           std::int32_t* argmax,
                                           int B,
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
                                           int pad_w);
LUCID_INTERNAL void max_pool2d_forward_f64(const double* x,
                                           double* y,
                                           std::int32_t* argmax,
                                           int B,
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
                                           int pad_w);
// 2-D max pooling backward.  dx must be zeroed by the caller.
LUCID_INTERNAL void max_pool2d_backward_f32(const float* g,
                                            const std::int32_t* argmax,
                                            float* dx,
                                            int B,
                                            int C,
                                            int H,
                                            int W,
                                            int OH,
                                            int OW);
LUCID_INTERNAL void max_pool2d_backward_f64(const double* g,
                                            const std::int32_t* argmax,
                                            double* dx,
                                            int B,
                                            int C,
                                            int H,
                                            int W,
                                            int OH,
                                            int OW);
// 2-D average pooling forward: count-include-pad, divisor = KH*KW.
LUCID_INTERNAL void avg_pool2d_forward_f32(const float* x,
                                           float* y,
                                           int B,
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
                                           int pad_w);
LUCID_INTERNAL void avg_pool2d_forward_f64(const double* x,
                                           double* y,
                                           int B,
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
                                           int pad_w);
// 2-D average pooling backward: distributes g/KH*KW uniformly.  dx must be zeroed.
LUCID_INTERNAL void avg_pool2d_backward_f32(const float* g,
                                            float* dx,
                                            int B,
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
                                            int pad_w);
LUCID_INTERNAL void avg_pool2d_backward_f64(const double* g,
                                            double* dx,
                                            int B,
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
                                            int pad_w);

// 3-D max pooling forward.  x has shape (B, C, D, H, W); y has shape (B, C, OD, OH, OW).
// argmax[i] is the flat DHW index of the maximum element.
LUCID_INTERNAL void max_pool3d_forward_f32(const float* x,
                                           float* y,
                                           std::int32_t* argmax,
                                           int B,
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
                                           int sd,
                                           int sh,
                                           int sw,
                                           int pd,
                                           int ph,
                                           int pw);
LUCID_INTERNAL void max_pool3d_forward_f64(const double* x,
                                           double* y,
                                           std::int32_t* argmax,
                                           int B,
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
                                           int sd,
                                           int sh,
                                           int sw,
                                           int pd,
                                           int ph,
                                           int pw);
LUCID_INTERNAL void max_pool3d_backward_f32(const float* g,
                                            const std::int32_t* argmax,
                                            float* dx,
                                            int B,
                                            int C,
                                            int D,
                                            int H,
                                            int W,
                                            int OD,
                                            int OH,
                                            int OW);
LUCID_INTERNAL void max_pool3d_backward_f64(const double* g,
                                            const std::int32_t* argmax,
                                            double* dx,
                                            int B,
                                            int C,
                                            int D,
                                            int H,
                                            int W,
                                            int OD,
                                            int OH,
                                            int OW);
LUCID_INTERNAL void avg_pool3d_forward_f32(const float* x,
                                           float* y,
                                           int B,
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
                                           int sd,
                                           int sh,
                                           int sw,
                                           int pd,
                                           int ph,
                                           int pw);
LUCID_INTERNAL void avg_pool3d_forward_f64(const double* x,
                                           double* y,
                                           int B,
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
                                           int sd,
                                           int sh,
                                           int sw,
                                           int pd,
                                           int ph,
                                           int pw);
LUCID_INTERNAL void avg_pool3d_backward_f32(const float* g,
                                            float* dx,
                                            int B,
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
                                            int sd,
                                            int sh,
                                            int sw,
                                            int pd,
                                            int ph,
                                            int pw);
LUCID_INTERNAL void avg_pool3d_backward_f64(const double* g,
                                            double* dx,
                                            int B,
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
                                            int sd,
                                            int sh,
                                            int sw,
                                            int pd,
                                            int ph,
                                            int pw);

}  // namespace lucid::backend::cpu
