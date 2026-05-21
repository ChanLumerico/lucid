// lucid/_C/nn/PoolNd.h
//
// Autograd-aware N-dimensional MaxPool and AvgPool for $N \in \{1, 2, 3\}$.
//
// MaxPool slides a window over each spatial axis and emits the maximum
// element; AvgPool emits the arithmetic mean.  The output extent on
// each axis follows
//
// $$
// O_i = \left\lfloor
//   \frac{S_i + 2 p_i - K_i}{s_i} + 1
// \right\rfloor.
// $$
//
// ``MaxPoolNdBackward`` saves the argmax indices produced by the forward
// pass so that the backward pass can scatter ``grad_out`` only to the
// winning element of each window (a sparse routing).  ``AvgPoolNdBackward``
// stores no activations — the gradient is distributed uniformly across all
// elements of the window.
//
// When ``stride[i] == 0`` is passed to the public entry-point functions
// the forward pass treats it as a sentinel meaning "use ``K[i]``" — i.e.
// non-overlapping windows, the default for classic pooling layers.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for N-dimensional max pooling.
//
// Each output element is the maximum of a $K_0 \times \cdots \times
// K_{N-1}$ window of the (optionally zero-padded) input:
// $y[b, c, o] = \max_{0 \le k < K} x[b, c, s\,o + k]$ (extended to
// $N$ axes).  The forward pass also produces a tensor of *argmax*
// indices into the flat padded-input buffer; these are saved as
// ``saved_argmax_`` and consumed by the backward pass to route each
// gradient back to exactly one input element.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op (``"max_pool1d"`` / ``"max_pool2d"`` /
//     ``"max_pool3d"``, ``AmpPolicy::KeepInput``).
// K_ : int[N]
//     Pooling window size per spatial axis.
// stride_ : int[N]
//     Per-axis stride (already resolved from the ``0`` sentinel during
//     the forward pass).
// pad_ : int[N]
//     Per-axis zero-padding.
// saved_argmax_ : Storage
//     Flat indices of the per-window winners; same logical shape as
//     the output.
template <int N>

class LUCID_API MaxPoolNdBackward : public FuncOp<MaxPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];              // Pooling window size per axis.
    int stride_[N];         // Stride per axis (already resolved from 0 in forward).
    int pad_[N];            // Zero-padding per axis.
    Storage saved_argmax_;  // Flat argmax indices, same shape as output.

    // Forward — pool each window down to its maximum element.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input of shape ``(B, C, S_0, ..., S_{N-1})``.
    // K : array<int, N>
    //     Per-axis window size.
    // stride : array<int, N>
    //     Per-axis stride.  ``stride[i] == 0`` defaults to ``K[i]``
    //     (non-overlapping windows).
    // pad : array<int, N>
    //     Per-axis zero-padding amount.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output of shape ``(B, C, O_0, ..., O_{N-1})`` with each
    //     $O_i = \lfloor (S_i + 2 p_i - K_i)/s_i + 1 \rfloor$.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the input rank disagrees with ``N + 2`` or any computed
    //     output extent is non-positive.
    static TensorImplPtr
    forward(const TensorImplPtr& x, const int (&K)[N], const int (&stride)[N], const int (&pad)[N]);

    // Backward — scatter ``grad_out`` into the input positions recorded
    // in ``saved_argmax_``.  Elements that did not win any window
    // receive zero gradient.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient shaped like the forward output.
    //
    // Returns
    // -------
    // vector<Storage>
    //     Single-element vector containing ``dx`` with the original
    //     input shape.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for N-dimensional average pooling.
//
// Each output element is the arithmetic mean of a $K_0 \times \cdots
// \times K_{N-1}$ window of the (optionally zero-padded) input.  No
// activations need to be saved — the backward pass is a uniform scatter
// that distributes each output gradient over the $\prod_i K_i$ input
// elements of its window.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op (``"avg_pool1d"`` / ``"avg_pool2d"`` /
//     ``"avg_pool3d"``, ``AmpPolicy::KeepInput``).
// K_ : int[N]
//     Pooling window size per axis.
// stride_ : int[N]
//     Per-axis stride (already resolved from the ``0`` sentinel).
// pad_ : int[N]
//     Per-axis zero-padding.
template <int N>

class LUCID_API AvgPoolNdBackward : public FuncOp<AvgPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];
    int stride_[N];
    int pad_[N];

    // Forward — emit the mean of each window.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input of shape ``(B, C, S_0, ..., S_{N-1})``.
    // K : array<int, N>
    //     Per-axis window size.
    // stride : array<int, N>
    //     Per-axis stride.  ``stride[i] == 0`` defaults to ``K[i]``.
    // pad : array<int, N>
    //     Per-axis zero-padding amount.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output of shape ``(B, C, O_0, ..., O_{N-1})`` with each
    //     $O_i = \lfloor (S_i + 2 p_i - K_i)/s_i + 1 \rfloor$.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the input rank disagrees with ``N + 2`` or any computed
    //     output extent is non-positive.
    static TensorImplPtr
    forward(const TensorImplPtr& x, const int (&K)[N], const int (&stride)[N], const int (&pad)[N]);

    // Backward — distribute ``grad_out`` evenly over each pooling
    // window (each contributing element receives ``1 / prod(K)`` of the
    // upstream gradient).
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient shaped like the forward output.
    //
    // Returns
    // -------
    // vector<Storage>
    //     Single-element vector containing ``dx``.
    std::vector<Storage> apply(Storage grad_out) override;
};

using MaxPool1dBackward = MaxPoolNdBackward<1>;
using MaxPool2dBackward = MaxPoolNdBackward<2>;
using MaxPool3dBackward = MaxPoolNdBackward<3>;
using AvgPool1dBackward = AvgPoolNdBackward<1>;
using AvgPool2dBackward = AvgPoolNdBackward<2>;
using AvgPool3dBackward = AvgPoolNdBackward<3>;

// One-dimensional max pooling over a batch of sequences.
//
// Emits the maximum of each sliding window of length ``KL`` along the
// last axis of the input.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, L)``.
// KL : int
//     Window length.
// stride_l : int, optional
//     Stride.  ``0`` is a sentinel meaning "use ``KL``" (non-overlapping
//     windows).  Default: ``0``.
// pad_l : int, optional
//     Zero-padding added to both ends.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, L_out)`` where
//     $L_\text{out} = \lfloor (L + 2 p - KL)/s + 1 \rfloor$.
LUCID_API TensorImplPtr max_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);

// Two-dimensional max pooling over a batch of feature maps.
//
// Emits the maximum of each $KH \times KW$ sliding window.  Provides
// local translation invariance and is the canonical downsampling
// primitive in convolutional classification networks.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, H, W)``.
// KH, KW : int
//     Per-axis window size.
// stride_h, stride_w : int, optional
//     Per-axis stride.  ``0`` is a sentinel meaning "use ``K``".
//     Default: ``0``.
// pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, H_out, W_out)`` with
//     $H_\text{out} = \lfloor (H + 2 p_h - KH)/s_h + 1 \rfloor$
//     and analogously for $W_\text{out}$.
LUCID_API TensorImplPtr max_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

// Three-dimensional max pooling over a batch of volumes.
//
// Emits the maximum of each $KD \times KH \times KW$ window.  Used in
// video classification and volumetric (CT / MRI) models.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, D, H, W)``.
// KD, KH, KW : int
//     Per-axis window size.
// stride_d, stride_h, stride_w : int, optional
//     Per-axis stride.  ``0`` is a sentinel for non-overlapping
//     windows.  Default: ``0``.
// pad_d, pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, D_out, H_out, W_out)``.
LUCID_API TensorImplPtr max_pool3d_op(const TensorImplPtr& x,
                                      int KD,
                                      int KH,
                                      int KW,
                                      int stride_d = 0,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_d = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

// One-dimensional average pooling over a batch of sequences.
//
// Emits the arithmetic mean of each sliding window of length ``KL``.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, L)``.
// KL : int
//     Window length.
// stride_l : int, optional
//     Stride.  ``0`` is a sentinel meaning "use ``KL``".  Default: ``0``.
// pad_l : int, optional
//     Zero-padding added to both ends.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, L_out)`` where
//     $L_\text{out} = \lfloor (L + 2 p - KL)/s + 1 \rfloor$.
LUCID_API TensorImplPtr avg_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);

// Two-dimensional average pooling over a batch of feature maps.
//
// Smooth, differentiable counterpart to ``max_pool2d_op``.  A common
// choice for *global average pooling* heads (use a kernel equal to the
// full spatial extent).
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, H, W)``.
// KH, KW : int
//     Per-axis window size.
// stride_h, stride_w : int, optional
//     Per-axis stride.  ``0`` is a sentinel for non-overlapping windows.
//     Default: ``0``.
// pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, H_out, W_out)``.
LUCID_API TensorImplPtr avg_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

// Three-dimensional average pooling over a batch of volumes.
//
// The natural extension of 2-D average pooling for video and
// volumetric data.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, D, H, W)``.
// KD, KH, KW : int
//     Per-axis window size.
// stride_d, stride_h, stride_w : int, optional
//     Per-axis stride; ``0`` ⇒ ``K``.  Default: ``0``.
// pad_d, pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, D_out, H_out, W_out)``.
LUCID_API TensorImplPtr avg_pool3d_op(const TensorImplPtr& x,
                                      int KD,
                                      int KH,
                                      int KW,
                                      int stride_d = 0,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_d = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

}  // namespace lucid
