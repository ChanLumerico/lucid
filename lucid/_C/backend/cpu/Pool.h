#pragma once

// =====================================================================
// Lucid C++ engine — 1D / 2D / 3D pooling kernels.
// =====================================================================
//
// Layouts:
//   1D: x in (B, C, L);            y in (B, C, OL)
//   2D: x in (B, C, H, W);         y in (B, C, OH, OW)
//   3D: x in (B, C, D, H, W);      y in (B, C, OD, OH, OW)
// argmax_idx (max only) holds the flat-spatial linear input index of the
// max within the kernel window for each output cell:
//   1D: l index
//   2D: h*W + w
//   3D: d*H*W + h*W + w
//
// AvgPool divisor = prod(K) (matches PyTorch count_include_pad=True).
//
// Layer: backend/cpu/. F32 + F64.

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// ============================ 1D ============================

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
LUCID_INTERNAL void max_pool1d_backward_f32(
    const float* g, const std::int32_t* argmax, float* dx, int B, int C, int L, int OL);
LUCID_INTERNAL void max_pool1d_backward_f64(
    const double* g, const std::int32_t* argmax, double* dx, int B, int C, int L, int OL);
LUCID_INTERNAL void avg_pool1d_forward_f32(
    const float* x, float* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
LUCID_INTERNAL void avg_pool1d_forward_f64(
    const double* x, double* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
LUCID_INTERNAL void avg_pool1d_backward_f32(
    const float* g, float* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);
LUCID_INTERNAL void avg_pool1d_backward_f64(
    const double* g, double* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// ============================ 2D ============================

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

// ============================ 3D ============================

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
