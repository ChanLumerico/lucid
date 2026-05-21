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

// Single-precision 1-D max-pool forward over an NCL-laid-out input.
//
// For each output element the kernel scans the corresponding ``KL``-wide
// window and records both the maximum value into ``y`` and the *flat*
// position within the ``L`` dimension of the input row into ``argmax``.
// Out-of-bounds (pad) positions contribute ``-inf`` and so never win unless
// the whole window is padding (in which case the first pad slot wins).
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, L)``.
// y : float*
//     Output tensor of shape ``(B, C, OL)``.
// argmax : int32_t*
//     Companion buffer of shape ``(B, C, OL)`` holding the winning
//     ``L``-axis index of each output element; consumed by the backward
//     kernel.
// B, C, L : int
//     Batch, channel, and spatial extents of ``x``.
// KL : int
//     Pool window length.
// OL : int
//     Spatial extent of ``y``; caller computes it as
//     ``(L + 2*pad_l - KL) / stride_l + 1``.
// stride_l, pad_l : int
//     Window stride and symmetric zero-pad on the ``L`` axis.
//
// Shape
// -----
// $\text{OL} = \lfloor (L + 2 \cdot \text{pad\_l} - \text{KL}) /
// \text{stride\_l}\rfloor + 1$.
//
// See Also
// --------
// max_pool1d_backward_f32 : Reverse pass that consumes ``argmax``.
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

// Double-precision counterpart to :cpp:func:`max_pool1d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, L)``.
// y : double*
//     Output tensor of shape ``(B, C, OL)``.
// argmax : int32_t*
//     Winning ``L``-axis indices, shape ``(B, C, OL)``.
// B, C, L, KL, OL, stride_l, pad_l : int
//     Same meaning as in :cpp:func:`max_pool1d_forward_f32`.
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

// Single-precision 1-D max-pool backward — scatters each upstream gradient
// onto the input slot that won its pool window.
//
// The caller must zero ``dx`` first because windows may overlap (when
// ``stride_l < KL``) and the kernel uses ``+=`` to accumulate contributions.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OL)``.
// argmax : const int32_t*
//     Argmax indices from the matching forward call; shape ``(B, C, OL)``.
// dx : float*
//     Output gradient w.r.t. the original input; shape ``(B, C, L)``;
//     *accumulated* — caller must zero before invocation.
// B, C, L, OL : int
//     Layout extents.
LUCID_INTERNAL void max_pool1d_backward_f32(
    const float* g, const std::int32_t* argmax, float* dx, int B, int C, int L, int OL);

// Double-precision counterpart to :cpp:func:`max_pool1d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OL)``.
// argmax : const int32_t*
//     Argmax indices from the matching forward call.
// dx : double*
//     Output gradient of shape ``(B, C, L)``; *accumulated*.
// B, C, L, OL : int
//     Layout extents.
LUCID_INTERNAL void max_pool1d_backward_f64(
    const double* g, const std::int32_t* argmax, double* dx, int B, int C, int L, int OL);

// Single-precision 1-D average-pool forward with count-include-pad semantics.
//
// Every output cell is the mean of its ``KL``-wide window with the divisor
// fixed at ``KL`` regardless of how many positions fall on padding (pad
// values contribute ``0`` to the sum but still count toward the divisor).
// This matches the count-include-pad convention used by the reference
// framework's default ``AvgPool1d``.
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, L)``.
// y : float*
//     Output tensor of shape ``(B, C, OL)``.
// B, C, L : int
//     Layout extents of ``x``.
// KL : int
//     Pool window length and (count-include-pad) divisor.
// OL : int
//     Spatial extent of ``y``.
// stride_l, pad_l : int
//     Window stride and symmetric zero-pad.
//
// Math
// ----
// $$y_{b,c,o} = \frac{1}{\text{KL}} \sum_{k=0}^{\text{KL}-1}
//   x_{b,c,\,o\cdot \text{stride\_l} + k - \text{pad\_l}}.$$
LUCID_INTERNAL void avg_pool1d_forward_f32(
    const float* x, float* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// Double-precision counterpart to :cpp:func:`avg_pool1d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, L)``.
// y : double*
//     Output tensor of shape ``(B, C, OL)``.
// B, C, L, KL, OL, stride_l, pad_l : int
//     Same meaning as :cpp:func:`avg_pool1d_forward_f32`.
LUCID_INTERNAL void avg_pool1d_forward_f64(
    const double* x, double* y, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// Single-precision 1-D average-pool backward — broadcasts each upstream
// gradient uniformly back into its source window with the fixed ``1/KL``
// weight.
//
// The caller must zero ``dx`` because overlapping windows accumulate into the
// same input slots.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OL)``.
// dx : float*
//     Output gradient of shape ``(B, C, L)``; *accumulated*.
// B, C, L, KL, OL, stride_l, pad_l : int
//     Layout and window parameters matching the forward call.
LUCID_INTERNAL void avg_pool1d_backward_f32(
    const float* g, float* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// Double-precision counterpart to :cpp:func:`avg_pool1d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OL)``.
// dx : double*
//     Output gradient of shape ``(B, C, L)``; *accumulated*.
// B, C, L, KL, OL, stride_l, pad_l : int
//     Layout and window parameters matching the forward call.
LUCID_INTERNAL void avg_pool1d_backward_f64(
    const double* g, double* dx, int B, int C, int L, int KL, int OL, int stride_l, int pad_l);

// Single-precision 2-D max-pool forward over an NCHW input.
//
// For every output cell the kernel scans the corresponding ``KH × KW``
// window and writes both the maximum into ``y`` and the *flat* ``H*W`` index
// of the winning input position into ``argmax``.  Pad positions contribute
// ``-inf`` (an all-pad window yields the first pad slot as the argmax).
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, H, W)``.
// y : float*
//     Output tensor of shape ``(B, C, OH, OW)``.
// argmax : int32_t*
//     Per-output winning input position, encoded as ``h*W + w`` within the
//     2-D spatial slice; shape ``(B, C, OH, OW)``.
// B, C, H, W : int
//     Layout extents of ``x``.
// KH, KW : int
//     Window height and width.
// OH, OW : int
//     Output spatial extents.
// stride_h, stride_w : int
//     Window stride along H and W.
// pad_h, pad_w : int
//     Symmetric zero-pad on each spatial axis.
//
// Shape
// -----
// $\text{OH} = \lfloor (H + 2\cdot\text{pad\_h} - \text{KH})/\text{stride\_h}\rfloor + 1$,
// analogously for $\text{OW}$.
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

// Double-precision counterpart to :cpp:func:`max_pool2d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, H, W)``.
// y : double*
//     Output tensor of shape ``(B, C, OH, OW)``.
// argmax : int32_t*
//     Per-output winning ``h*W + w`` indices.
// B, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w : int
//     Layout and window parameters; see :cpp:func:`max_pool2d_forward_f32`.
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

// Single-precision 2-D max-pool backward — scatters each upstream gradient
// onto the input slot that won its pool window (per the forward ``argmax``).
//
// ``dx`` is *accumulated* and must be zeroed by the caller; this allows
// overlapping windows (``stride < kernel``) to sum contributions correctly.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OH, OW)``.
// argmax : const int32_t*
//     Per-output winning ``h*W + w`` indices from the matching forward call.
// dx : float*
//     Output gradient of shape ``(B, C, H, W)``; *accumulated*.
// B, C, H, W, OH, OW : int
//     Layout extents.
LUCID_INTERNAL void max_pool2d_backward_f32(const float* g,
                                            const std::int32_t* argmax,
                                            float* dx,
                                            int B,
                                            int C,
                                            int H,
                                            int W,
                                            int OH,
                                            int OW);

// Double-precision counterpart to :cpp:func:`max_pool2d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OH, OW)``.
// argmax : const int32_t*
//     Per-output winning ``h*W + w`` indices.
// dx : double*
//     Output gradient of shape ``(B, C, H, W)``; *accumulated*.
// B, C, H, W, OH, OW : int
//     Layout extents.
LUCID_INTERNAL void max_pool2d_backward_f64(const double* g,
                                            const std::int32_t* argmax,
                                            double* dx,
                                            int B,
                                            int C,
                                            int H,
                                            int W,
                                            int OH,
                                            int OW);

// Single-precision 2-D average-pool forward with count-include-pad semantics.
//
// Every output cell is the mean of its ``KH × KW`` window with divisor fixed
// at ``KH * KW`` regardless of how many positions fall on padding; padded
// positions contribute ``0`` to the sum but still count toward the divisor.
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, H, W)``.
// y : float*
//     Output tensor of shape ``(B, C, OH, OW)``.
// B, C, H, W : int
//     Layout extents of ``x``.
// KH, KW : int
//     Window height and width.  ``KH * KW`` is the (count-include-pad) divisor.
// OH, OW : int
//     Output spatial extents.
// stride_h, stride_w, pad_h, pad_w : int
//     Window stride and symmetric zero-pad on each axis.
//
// Math
// ----
// $$y_{b,c,oh,ow} = \frac{1}{\text{KH}\,\text{KW}} \sum_{kh, kw} x_{b, c, ih, iw},$$
// with $ih = oh\cdot\text{stride\_h} + kh - \text{pad\_h}$, similarly for $iw$.
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

// Double-precision counterpart to :cpp:func:`avg_pool2d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, H, W)``.
// y : double*
//     Output tensor of shape ``(B, C, OH, OW)``.
// B, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w : int
//     Layout and window parameters; see :cpp:func:`avg_pool2d_forward_f32`.
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

// Single-precision 2-D average-pool backward — broadcasts each upstream
// gradient back over its source window with the fixed ``1/(KH*KW)`` weight.
//
// ``dx`` is *accumulated* and must be zeroed by the caller because
// overlapping windows write to overlapping input slots.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OH, OW)``.
// dx : float*
//     Output gradient of shape ``(B, C, H, W)``; *accumulated*.
// B, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w : int
//     Layout and window parameters matching the forward call.
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

// Double-precision counterpart to :cpp:func:`avg_pool2d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OH, OW)``.
// dx : double*
//     Output gradient of shape ``(B, C, H, W)``; *accumulated*.
// B, C, H, W, KH, KW, OH, OW, stride_h, stride_w, pad_h, pad_w : int
//     Layout and window parameters matching the forward call.
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

// Single-precision 3-D max-pool forward over an NCDHW input.
//
// For every output voxel the kernel scans the corresponding
// ``KD × KH × KW`` window, writes the maximum into ``y``, and stores the flat
// ``d*H*W + h*W + w`` index of the winner into ``argmax``.  Pad voxels are
// treated as ``-inf``.
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, D, H, W)``.
// y : float*
//     Output tensor of shape ``(B, C, OD, OH, OW)``.
// argmax : int32_t*
//     Per-output winning flat 3-D index, shape ``(B, C, OD, OH, OW)``.
// B, C, D, H, W : int
//     Layout extents of ``x``.
// KD, KH, KW : int
//     Window depth, height, width.
// OD, OH, OW : int
//     Output spatial extents.
// sd, sh, sw : int
//     Window strides along D, H, W.
// pd, ph, pw : int
//     Symmetric zero-pads along D, H, W.
//
// Shape
// -----
// $\text{OD} = \lfloor (D + 2\cdot\text{pd} - \text{KD})/\text{sd}\rfloor + 1$,
// analogously for $\text{OH}, \text{OW}$.
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

// Double-precision counterpart to :cpp:func:`max_pool3d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, D, H, W)``.
// y : double*
//     Output tensor of shape ``(B, C, OD, OH, OW)``.
// argmax : int32_t*
//     Per-output winning flat 3-D index.
// B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw : int
//     Layout and window parameters; see :cpp:func:`max_pool3d_forward_f32`.
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

// Single-precision 3-D max-pool backward — scatters each upstream gradient
// onto the input voxel that won its pool window (per the forward ``argmax``).
//
// ``dx`` is *accumulated* and must be zeroed by the caller.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OD, OH, OW)``.
// argmax : const int32_t*
//     Per-output winning flat 3-D index from the matching forward call.
// dx : float*
//     Output gradient of shape ``(B, C, D, H, W)``; *accumulated*.
// B, C, D, H, W, OD, OH, OW : int
//     Layout extents.
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

// Double-precision counterpart to :cpp:func:`max_pool3d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OD, OH, OW)``.
// argmax : const int32_t*
//     Per-output winning flat 3-D index.
// dx : double*
//     Output gradient of shape ``(B, C, D, H, W)``; *accumulated*.
// B, C, D, H, W, OD, OH, OW : int
//     Layout extents.
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

// Single-precision 3-D average-pool forward with count-include-pad semantics.
//
// Each output voxel is the mean of its ``KD × KH × KW`` window with divisor
// fixed at ``KD * KH * KW``; padded positions contribute ``0`` to the sum
// but still count toward the divisor.
//
// Parameters
// ----------
// x : const float*
//     Input tensor of shape ``(B, C, D, H, W)``.
// y : float*
//     Output tensor of shape ``(B, C, OD, OH, OW)``.
// B, C, D, H, W : int
//     Layout extents of ``x``.
// KD, KH, KW : int
//     Window depth, height, width.  Product is the count-include-pad divisor.
// OD, OH, OW : int
//     Output spatial extents.
// sd, sh, sw, pd, ph, pw : int
//     Window strides and symmetric zero-pads on each axis.
//
// Math
// ----
// $$y_{b,c,od,oh,ow} = \frac{1}{\text{KD}\,\text{KH}\,\text{KW}}
//   \sum_{kd,kh,kw} x_{b,c, id, ih, iw},$$
// with $id = od\cdot\text{sd} + kd - \text{pd}$, similarly for $ih, iw$.
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

// Double-precision counterpart to :cpp:func:`avg_pool3d_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input tensor of shape ``(B, C, D, H, W)``.
// y : double*
//     Output tensor of shape ``(B, C, OD, OH, OW)``.
// B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw : int
//     Layout and window parameters; see :cpp:func:`avg_pool3d_forward_f32`.
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

// Single-precision 3-D average-pool backward — broadcasts each upstream
// gradient back over its source window with the fixed ``1/(KD*KH*KW)`` weight.
//
// ``dx`` is *accumulated* and must be zeroed by the caller.
//
// Parameters
// ----------
// g : const float*
//     Upstream gradient of shape ``(B, C, OD, OH, OW)``.
// dx : float*
//     Output gradient of shape ``(B, C, D, H, W)``; *accumulated*.
// B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw : int
//     Layout and window parameters matching the forward call.
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

// Double-precision counterpart to :cpp:func:`avg_pool3d_backward_f32`.
//
// Parameters
// ----------
// g : const double*
//     Upstream gradient of shape ``(B, C, OD, OH, OW)``.
// dx : double*
//     Output gradient of shape ``(B, C, D, H, W)``; *accumulated*.
// B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph, pw : int
//     Layout and window parameters matching the forward call.
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
