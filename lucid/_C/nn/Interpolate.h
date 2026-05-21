// lucid/_C/nn/Interpolate.h
//
// Autograd-aware image / volume interpolation operations: bilinear (2-D),
// trilinear (3-D), and nearest-neighbour (2-D / 3-D, inference only).
//
// All differentiable variants resize the trailing spatial axes of a tensor in
// the standard ``(N, C, ...)`` channel-first layout.  The batch and channel
// dimensions are preserved unchanged.  Coordinate mapping is controlled by
// ``align_corners`` exactly as described in the public ``nn.interpolate``
// helper:
//
// * ``align_corners=true`` — the source coordinate is computed as
//   $x_{\text{in}} = i \cdot (S_{\text{in}} - 1) / (S_{\text{out}} - 1)$,
//   so that the four (or six / eight) corner pixels of the input map exactly
//   onto the corner pixels of the output.
// * ``align_corners=false`` — the source coordinate is computed as
//   $x_{\text{in}} = (i + 0.5) \cdot S_{\text{in}} / S_{\text{out}} - 0.5$,
//   placing the resampling grid on pixel *centres* and producing a half-pixel
//   shift relative to the corner-aligned variant.
//
// The nearest-neighbour ops are inference-only — they produce a fixed output
// and never attach an autograd node.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for 2-D bilinear interpolation.
//
// Resizes a 4-D tensor of shape ``(N, C, H_in, W_in)`` to
// ``(N, C, H_out, W_out)`` by linearly interpolating between the four
// neighbouring input pixels:
// $$
//   y[i, j] = \sum_{a, b \in \{0, 1\}}
//     w_{i}^{a} \, w_{j}^{b} \, x[i_0 + a, j_0 + b]
// $$
// where $(i_0, j_0)$ is the integer floor of the source coordinate and
// $w_i^a$, $w_j^b$ are the fractional bilinear weights computed from the
// ``align_corners`` rule above.  The backward distributes ``grad_out`` back
// to the same four pixels weighted by the *same* bilinear coefficients,
// recomputed in the backend (no per-sample weight cache is saved).
//
// Math
// ----
// $$
//   x_{\text{in}} = \begin{cases}
//     j \cdot (W_{\text{in}} - 1) / (W_{\text{out}} - 1) & \text{align\_corners} \\
//     (j + 0.5) \cdot W_{\text{in}} / W_{\text{out}} - 0.5 & \text{otherwise}
//   \end{cases}
// $$
// (analogous formula for $y_{\text{in}}$).
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"interpolate_bilinear"``, ``AmpPolicy::Promote``.
// H_in_, W_in_ : int
//     Saved input spatial dimensions.
// H_out_, W_out_ : int
//     Requested output spatial dimensions.
// align_corners_ : bool
//     Coordinate mapping mode (see file header).
// orig_shape_ : Shape
//     Full ``(N, C, H_in, W_in)`` shape for backward reconstruction.
class LUCID_API InterpolateBilinearBackward : public FuncOp<InterpolateBilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int H_in_ = 0, W_in_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;  // Full (N, C, H_in, W_in) shape for backward.

    // Compute the bilinear resize with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     4-D input tensor of shape ``(N, C, H_in, W_in)``.
    // H_out : int
    //     Desired output height.
    // W_out : int
    //     Desired output width.
    // align_corners : bool
    //     Coordinate mapping mode (see file header).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Resampled tensor of shape ``(N, C, H_out, W_out)``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``input`` is not 4-D.
    static TensorImplPtr
    forward(const TensorImplPtr& input, int H_out, int W_out, bool align_corners);

    // Backward pass: scatters ``grad_out`` to source pixels using the same
    // bilinear weights as the forward.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for 3-D trilinear interpolation.
//
// Resizes a 5-D tensor of shape ``(N, C, D_in, H_in, W_in)`` to
// ``(N, C, D_out, H_out, W_out)`` by linearly interpolating between the
// eight neighbouring input voxels.  Conceptually the trilinear weights are
// the product of three 1-D linear weights (one per spatial axis), giving
// $$
//   y[d, i, j] = \sum_{a, b, c \in \{0, 1\}}
//     w_{d}^{a} \, w_{i}^{b} \, w_{j}^{c} \, x[d_0 + a, i_0 + b, j_0 + c]
// $$
// where the per-axis source coordinates follow the same ``align_corners`` rule
// described in the file header.
//
// Math
// ----
// Output size formula (per axis $S \in \{D, H, W\}$):
// $$
//   S_{\text{out}} = \text{user-specified}, \quad
//   s_{\text{in}}^{(k)} = \begin{cases}
//     k \cdot (S_{\text{in}} - 1) / (S_{\text{out}} - 1) & \text{align\_corners} \\
//     (k + 0.5) \cdot S_{\text{in}} / S_{\text{out}} - 0.5 & \text{otherwise}
//   \end{cases}
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"interpolate_trilinear"``, ``AmpPolicy::Promote``.
// D_in_, H_in_, W_in_ : int
//     Saved input spatial dimensions.
// D_out_, H_out_, W_out_ : int
//     Requested output spatial dimensions.
// align_corners_ : bool
//     Coordinate mapping mode.
// orig_shape_ : Shape
//     Full ``(N, C, D_in, H_in, W_in)`` shape for backward reconstruction.
class LUCID_API InterpolateTrilinearBackward : public FuncOp<InterpolateTrilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int D_in_ = 0, H_in_ = 0, W_in_ = 0;
    int D_out_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;

    // Compute the trilinear resize with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     5-D input tensor of shape ``(N, C, D_in, H_in, W_in)``.
    // D_out : int
    //     Desired output depth.
    // H_out : int
    //     Desired output height.
    // W_out : int
    //     Desired output width.
    // align_corners : bool
    //     Coordinate mapping mode.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Resampled tensor of shape ``(N, C, D_out, H_out, W_out)``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``input`` is not 5-D.
    static TensorImplPtr
    forward(const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners);

    // Backward pass: scatters ``grad_out`` to source voxels using the same
    // trilinear weights as the forward.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public 2-D bilinear resize with autograd.
//
// Thin wrapper that delegates to ``InterpolateBilinearBackward::forward``.
//
// Parameters
// ----------
// input : TensorImplPtr
//     4-D input tensor ``(N, C, H_in, W_in)``.
// H_out : int
//     Desired output height.
// W_out : int
//     Desired output width.
// align_corners : bool
//     Coordinate mapping mode.
//
// Returns
// -------
// TensorImplPtr
//     Bilinearly resampled tensor of shape ``(N, C, H_out, W_out)``.
LUCID_API TensorImplPtr interpolate_bilinear_op(const TensorImplPtr& input,
                                                int H_out,
                                                int W_out,
                                                bool align_corners);

// Public 3-D trilinear resize with autograd.
//
// Parameters
// ----------
// input : TensorImplPtr
//     5-D input tensor ``(N, C, D_in, H_in, W_in)``.
// D_out : int
//     Desired output depth.
// H_out : int
//     Desired output height.
// W_out : int
//     Desired output width.
// align_corners : bool
//     Coordinate mapping mode.
//
// Returns
// -------
// TensorImplPtr
//     Trilinearly resampled tensor of shape ``(N, C, D_out, H_out, W_out)``.
LUCID_API TensorImplPtr interpolate_trilinear_op(
    const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners);

// Public 2-D nearest-neighbour resize (no autograd).
//
// Each output pixel receives the value of the nearest input pixel:
// $$
//   y[i, j] = x\!\left[\lfloor (i + 0.5) \cdot H_{\text{in}} / H_{\text{out}} \rfloor,\;
//                       \lfloor (j + 0.5) \cdot W_{\text{in}} / W_{\text{out}} \rfloor\right]
// $$
// No backward node is attached — the operation is inference-only.
//
// Parameters
// ----------
// input : TensorImplPtr
//     4-D input tensor ``(N, C, H_in, W_in)``.
// H_out : int
//     Desired output height.
// W_out : int
//     Desired output width.
//
// Returns
// -------
// TensorImplPtr
//     Resampled tensor of shape ``(N, C, H_out, W_out)``.
LUCID_API TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input, int H_out, int W_out);

// Public 3-D nearest-neighbour resize (no autograd).
//
// Volumetric counterpart of ``interpolate_nearest_2d_op``; produces a
// ``(N, C, D_out, H_out, W_out)`` output by floor-rounding source voxel
// coordinates.  No backward node is attached.
//
// Parameters
// ----------
// input : TensorImplPtr
//     5-D input tensor ``(N, C, D_in, H_in, W_in)``.
// D_out : int
//     Desired output depth.
// H_out : int
//     Desired output height.
// W_out : int
//     Desired output width.
//
// Returns
// -------
// TensorImplPtr
//     Resampled tensor of shape ``(N, C, D_out, H_out, W_out)``.
LUCID_API TensorImplPtr interpolate_nearest_3d_op(const TensorImplPtr& input,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out);

}  // namespace lucid
