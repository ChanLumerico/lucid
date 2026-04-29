#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/Im2Col.h
// =====================================================================
//
// Primitive: im2col / col2im façade.
//
// Routes to:
//   CPU → backend/cpu/Im2Col.h  (hand-rolled loops, Accelerate-friendly)
//   GPU → MLX conv ops handle im2col internally (no explicit call needed)
//
// Used by: conv1d/2d/3d forward, conv_transpose backward dx, unfold forward.
// Used by: conv* backward dW, conv_transpose forward, unfold backward.
//
// Layer: kernel/primitives/. Depends on backend/cpu/ only.

#include "../../backend/cpu/Im2Col.h"

// Re-exports all backend::cpu im2col/col2im variants:
//
// 1-D:
//   void im2col_1d_f32(const float* x, int N, int C, int L,
//                      int kernel, int stride, int pad, int dilation,
//                      float* col);
//   void col2im_1d_f32(const float* col, int N, int C, int L,
//                      int kernel, int stride, int pad, int dilation,
//                      float* x);
//   (f64 variants also available)
//
// 2-D:
//   void im2col_f32(const float* x, int N, int C, int H, int W,
//                   int kH, int kW, int sH, int sW,
//                   int pH, int pW, int dH, int dW, float* col);
//   void col2im_f32(...);
//   (f64 variants also available)
//
// 3-D:
//   void im2col_3d_f32(...);
//   void col2im_3d_f32(...);
//   (f64 variants also available)
//
// GPU path: MLX conv ops (mlx::core::conv1d/conv2d) perform im2col
// implicitly. No explicit im2col call is needed on GPU — pass tensors
// directly to mlx::core::conv*.
