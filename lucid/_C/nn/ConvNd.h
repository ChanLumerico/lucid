// lucid/_C/nn/ConvNd.h
//
// Autograd-aware N-dimensional convolution (N = 1, 2, 3) and the im2col-based
// Unfold / Fold operations.
//
// The forward op implements the standard *cross-correlation* used by deep
// learning libraries (not a true mathematical convolution — the kernel is
// not flipped):
//
// $$
// y[b,\, c_o,\, o_0, \ldots, o_{N-1}] =
//   \sum_{c_i,\, k_0, \ldots, k_{N-1}}
//   x\bigl[b,\, c_i,\, s_0 o_0 + d_0 k_0,\, \ldots\bigr]
//   \cdot W[c_o,\, c_i,\, k_0, \ldots, k_{N-1}]
//   + b[c_o].
// $$
//
// CPU backend uses im2col + GEMM via Apple Accelerate; GPU backend uses the
// MLX convolutional primitives.  ``UnfoldBackward`` exposes the explicit
// im2col step on its own for users implementing custom convolutions.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the N-dimensional cross-correlation.
//
// Explicitly instantiated for $N \in \{1, 2, 3\}$.  The per-axis hyper-
// parameters (``stride_``, ``pad_``, ``dilation_``) are stored as
// fixed-size C arrays of length ``N`` to avoid heap allocation on the
// forward hot path.  ``groups_`` partitions both ``C_in`` and ``C_out``
// into independent filter blocks (``groups_ == C_in`` yields a
// *depthwise* convolution).
//
// Backward emits three gradient slots: ``dx`` via a transposed-conv
// against ``W``, ``dW`` via an im2col-then-matmul against the saved
// input, and ``db`` via a channel-wise reduction of ``grad_out``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op schema (``"conv1d"`` / ``"conv2d"`` / ``"conv3d"``)
//     with ``AmpPolicy::Promote`` — both inputs are cast to the active
//     AMP compute dtype before dispatch.
// stride_ : int[N]
//     Per-axis stride between successive kernel placements.
// pad_ : int[N]
//     Per-axis zero-padding amount applied symmetrically to both sides.
// dilation_ : int[N]
//     Per-axis kernel-element spacing (atrous convolution).
// groups_ : int
//     Number of input/output channel groups; defaults to ``1``.
template <int N>

class LUCID_API ConvNdBackward : public FuncOp<ConvNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];    // Convolution stride per spatial axis.
    int pad_[N];       // Zero-padding per spatial axis.
    int dilation_[N];  // Dilation (hole size) per spatial axis.
    int groups_ = 1;   // Number of filter groups.

    // Run the forward cross-correlation and attach the backward node.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input batch of shape ``(B, C_in, S_0, ..., S_{N-1})``.
    // W : TensorImplPtr
    //     Filter bank of shape ``(C_out, C_in / groups, K_0, ..., K_{N-1})``.
    // b : TensorImplPtr
    //     Bias of shape ``(C_out,)``; pass an empty tensor for a
    //     bias-less convolution.
    // stride : array<int, N>
    //     Per-axis stride ``(s_0, ..., s_{N-1})``.
    // pad : array<int, N>
    //     Per-axis zero-padding applied to each side.
    // dilation : array<int, N>
    //     Per-axis kernel-element spacing (atrous convolution).
    // groups : int
    //     Filter grouping; both ``C_in`` and ``C_out`` must be divisible
    //     by it.  ``groups == C_in`` yields depthwise convolution.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output of shape ``(B, C_out, O_0, ..., O_{N-1})`` where for
    //     each spatial axis $i$
    //     $$
    //     O_i = \left\lfloor
    //       \frac{S_i + 2 p_i - d_i (K_i - 1) - 1}{s_i} + 1
    //     \right\rfloor.
    //     $$
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the input rank disagrees with ``N + 2``, if ``C_in`` /
    //     ``C_out`` are not divisible by ``groups``, if ``C_in / groups``
    //     does not match ``W``'s second axis, if ``b`` is not 1-D of
    //     length ``C_out``, or if the computed output extent is
    //     non-positive.
    // DeviceMismatch
    //     If ``x``, ``W``, ``b`` are not all on the same device.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&dilation)[N],
                                 int groups);

    // Backward — compute gradients ``[dx, dW, db]`` for the three saved
    // inputs.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient with the forward output shape.
    //
    // Returns
    // -------
    // vector<Storage>
    //     Three slots in declaration order: ``dx`` (transposed-conv),
    //     ``dW`` (im2col + matmul), ``db`` (channel-wise reduction).
    std::vector<Storage> apply(Storage grad_out) override;
};

using Conv1dBackward = ConvNdBackward<1>;
using Conv2dBackward = ConvNdBackward<2>;
using Conv3dBackward = ConvNdBackward<3>;

// One-dimensional cross-correlation over a batch of signals.
//
// Computes the autograd-aware op
// $y[b, c_o, l] = \sum_{c_i, k} x[b, c_i, s\,l + d\,k] \cdot W[c_o, c_i, k] + b[c_o]$
// where ``s`` is the stride and ``d`` is the dilation.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, L)``.
// W : TensorImplPtr
//     Filters of shape ``(C_out, C_in / groups, KL)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_l : int, optional
//     Stride along the length axis.  Default: ``1``.
// pad_l : int, optional
//     Zero-padding added to both ends of ``L``.  Default: ``0``.
// dilation_l : int, optional
//     Kernel-element spacing along ``L``.  Default: ``1``.
// groups : int, optional
//     Channel grouping.  Default: ``1``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, L_out)`` where
//     $L_\text{out} = \lfloor (L + 2 p - d(KL - 1) - 1)/s + 1 \rfloor$.
LUCID_API TensorImplPtr conv1d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_l = 1,
                                  int pad_l = 0,
                                  int dilation_l = 1,
                                  int groups = 1);

// Two-dimensional cross-correlation over a batch of images / feature maps.
//
// Computes
// $$
// y[b, c_o, h, w] = \sum_{c_i, k_h, k_w}
//   x[b, c_i, s_h h + d_h k_h, s_w w + d_w k_w]
//   \cdot W[c_o, c_i, k_h, k_w] + b[c_o].
// $$
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, H, W)``.
// W : TensorImplPtr
//     Filters of shape ``(C_out, C_in / groups, KH, KW)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_h, stride_w : int, optional
//     Per-axis stride.  Default: ``1``.
// pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
// dilation_h, dilation_w : int, optional
//     Per-axis kernel-element spacing.  Default: ``1``.
// groups : int, optional
//     Channel grouping.  Default: ``1``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, H_out, W_out)`` with
//     $H_\text{out} = \lfloor (H + 2 p_h - d_h(KH - 1) - 1)/s_h + 1 \rfloor$
//     and analogously for $W_\text{out}$.
LUCID_API TensorImplPtr conv2d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_h = 1,
                                  int stride_w = 1,
                                  int pad_h = 0,
                                  int pad_w = 0,
                                  int dilation_h = 1,
                                  int dilation_w = 1,
                                  int groups = 1);

// Three-dimensional cross-correlation over a batch of volumes.
//
// Performs the natural extension of 2-D convolution to the additional
// depth axis ``D``, summing the kernel response across
// $(k_d, k_h, k_w)$.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, D, H, W)``.
// W : TensorImplPtr
//     Filters of shape ``(C_out, C_in / groups, KD, KH, KW)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_d, stride_h, stride_w : int, optional
//     Per-axis stride.  Default: ``1``.
// pad_d, pad_h, pad_w : int, optional
//     Per-axis zero-padding.  Default: ``0``.
// dilation_d, dilation_h, dilation_w : int, optional
//     Per-axis kernel-element spacing.  Default: ``1``.
// groups : int, optional
//     Channel grouping.  Default: ``1``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, D_out, H_out, W_out)`` where each
//     output extent follows
//     $X_\text{out} = \lfloor (X + 2 p_x - d_x(K_X - 1) - 1)/s_x + 1 \rfloor$
//     for $X \in \{D, H, W\}$.
LUCID_API TensorImplPtr conv3d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_d = 1,
                                  int stride_h = 1,
                                  int stride_w = 1,
                                  int pad_d = 0,
                                  int pad_h = 0,
                                  int pad_w = 0,
                                  int dilation_d = 1,
                                  int dilation_h = 1,
                                  int dilation_w = 1,
                                  int groups = 1);

// Autograd node for the Unfold (im2col) operation.
//
// Slides a kernel-sized window over each spatial dimension of the input
// and lays the resulting patches out as columns of a 3-D matrix:
//
// $$
// y[b,\, c \cdot \prod K + k_0 \cdot K_1 \cdots + k_{N-1},\,
//   o_0 \cdot \prod O' + \cdots + o_{N-1}]
//   = x[b,\, c,\, s_0 o_0 + d_0 k_0,\, \ldots].
// $$
//
// The output column count equals $\prod_i O_i$, the same value a
// matching convolution would produce.  Supports $N \in \{1, 2, 3\}$
// inferred from ``kernel.size()``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op (``"unfold"``, ``AmpPolicy::KeepInput``).
// kernel_ : vector<int>
//     Kernel sizes per spatial axis.
// stride_ : vector<int>
//     Strides per spatial axis.
// pad_ : vector<int>
//     Zero-padding per spatial axis.
// dilation_ : vector<int>
//     Dilations per spatial axis.
class LUCID_API UnfoldBackward : public FuncOp<UnfoldBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<int> kernel_;    // Kernel sizes per spatial axis.
    std::vector<int> stride_;    // Strides per spatial axis.
    std::vector<int> pad_;       // Padding per spatial axis.
    std::vector<int> dilation_;  // Dilations per spatial axis.

    // Run the forward im2col extraction.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input of shape ``(B, C, S_0, ..., S_{N-1})`` where
    //     ``N = kernel.size()``.
    // kernel : vector<int>
    //     Per-axis kernel size (length ``N``).
    // stride : vector<int>
    //     Per-axis stride.
    // pad : vector<int>
    //     Per-axis zero-padding.
    // dilation : vector<int>
    //     Per-axis dilation.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Column matrix of shape ``(B, C * prod(K), prod(O))``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the input rank disagrees with ``N + 2`` or any output
    //     extent is non-positive.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& stride,
                                 const std::vector<int>& pad,
                                 const std::vector<int>& dilation);

    // Backward — the fold (col2im) operation that scatters the patches
    // back to spatial positions, summing overlaps.
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

// Public entry point for Unfold (im2col).
//
// Functional wrapper around ``UnfoldBackward::forward`` for building
// custom convolution-like ops in user code.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, S_0, ..., S_{N-1})``.
// kernel : vector<int>
//     Per-axis kernel size; length determines ``N``.
// stride : vector<int>
//     Per-axis stride.
// pad : vector<int>
//     Per-axis zero-padding.
// dilation : vector<int>
//     Per-axis kernel-element spacing.
//
// Returns
// -------
// TensorImplPtr
//     Patch-as-columns tensor of shape ``(B, C * prod(K), prod(O))``.
LUCID_API TensorImplPtr unfold_op(const TensorImplPtr& x,
                                  const std::vector<int>& kernel,
                                  const std::vector<int>& stride,
                                  const std::vector<int>& pad,
                                  const std::vector<int>& dilation);

// Fold (col2im) — the inverse of Unfold for 2-D inputs.
//
// Scatters each column back into its source location in the spatial
// grid; values at overlapping positions are summed (this is the formal
// adjoint of Unfold, not a true inverse when stride permits overlap).
//
// Parameters
// ----------
// x : TensorImplPtr
//     Column matrix of shape ``(N, C * kH * kW, L)`` produced by
//     ``unfold_op`` or any equivalent column layout.
// output_size : vector<int>
//     Target spatial size ``(outH, outW)``.
// kernel_size : vector<int>
//     Kernel size per spatial axis used during the matching unfold.
// stride : vector<int>
//     Stride per spatial axis.
// padding : vector<int>
//     Zero-padding per spatial axis.
// dilation : vector<int>
//     Dilation per spatial axis.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(N, C, outH, outW)`` with overlapping
//     contributions summed.
LUCID_API TensorImplPtr fold_op(const TensorImplPtr& x,
                                const std::vector<int>& output_size,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& stride,
                                const std::vector<int>& padding,
                                const std::vector<int>& dilation);

}  // namespace lucid
