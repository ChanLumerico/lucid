// lucid/_C/nn/Spatial.h
//
// Autograd-aware spatial transformation primitives: affine grid generation
// and differentiable image resampling.  Together these two operations form
// the building blocks of Spatial Transformer Networks (Jaderberg et al.,
// 2015), which learn to align inputs by predicting an affine matrix that is
// applied through a differentiable resampling step.
//
// The convention follows the standard reference implementation:
//
// * ``theta`` describes the affine transform from *output* coordinates to
//   *input* coordinates, both normalised to $[-1, 1]$.
// * The sampling grid is computed by applying ``theta`` to the regular
//   $H \times W$ output lattice.
// * ``grid_sample`` then reads ``input`` at the resulting (typically
//   non-integer) positions using bilinear / nearest / bicubic interpolation
//   and one of three out-of-bounds padding modes.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for affine grid generation.
//
// Builds an ``(N, H, W, 2)`` sampling grid by applying the per-sample affine
// matrix $\theta_n \in \mathbb{R}^{2 \times 3}$ to a regular $H \times W$
// output lattice:
// $$
//   \begin{pmatrix} x_s \\ y_s \end{pmatrix}
//   = \theta_n \begin{pmatrix} x_t \\ y_t \\ 1 \end{pmatrix}
// $$
// where $(x_t, y_t)$ are the normalised output coordinates in $[-1, 1]$ and
// $(x_s, y_s)$ are the corresponding source coordinates that will be passed
// to ``grid_sample``.  The backward propagates ``grad_out`` (shape
// ``(N, H, W, 2)``) back into ``theta`` by summing the outer products with
// the homogeneous lattice coordinates.
//
// Math
// ----
// Lattice generation depends on ``align_corners_``:
// $$
//   x_t = \begin{cases}
//     -1 + 2 j / (W - 1)   & \text{align\_corners} \\
//     -1 + (2 j + 1) / W   & \text{otherwise}
//   \end{cases}, \quad
//   y_t = \begin{cases}
//     -1 + 2 i / (H - 1)   & \text{align\_corners} \\
//     -1 + (2 i + 1) / H   & \text{otherwise}
//   \end{cases}
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"affine_grid"``, ``AmpPolicy::Promote``.
// align_corners_ : bool
//     When ``true``, the four corners of the output lattice map exactly to
//     ``[-1, +1]`` on each axis; when ``false``, the lattice is placed on
//     pixel *centres*.
// N_, H_, W_ : int
//     Batch size and output spatial dimensions.
// orig_theta_shape_ : Shape
//     Original ``(N, 2, 3)`` shape used to materialise the gradient.
class LUCID_API AffineGridBackward : public FuncOp<AffineGridBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool align_corners_ = true;
    int N_ = 0, H_ = 0, W_ = 0;
    Shape orig_theta_shape_;

    // Generate the sampling grid with autograd wiring.
    //
    // Parameters
    // ----------
    // theta : TensorImplPtr
    //     Affine matrices of shape ``(N, 2, 3)``.
    // N : int
    //     Batch size (must equal ``theta.shape()[0]``).
    // H : int
    //     Output grid height.
    // W : int
    //     Output grid width.
    // align_corners : bool
    //     Lattice placement mode.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Sampling grid of shape ``(N, H, W, 2)`` with last-axis ordering
    //     ``(x, y)`` (i.e. ``[..., 0]`` is the x-coordinate).
    static TensorImplPtr
    forward(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);

    // Backward pass: gradient w.r.t. the affine matrices ``theta``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for differentiable image resampling (grid sample).
//
// Reads a 4-D input image at the (typically non-integer) positions specified
// by a sampling grid:
// $$
//   y[n, c, i, j] = \mathcal{I}\bigl(x_{n, c}, \; \text{grid}[n, i, j]\bigr)
// $$
// where $\mathcal{I}$ is the chosen interpolation kernel and the grid stores
// normalised $(x, y)$ coordinates in $[-1, 1]$.  For bilinear interpolation
// the four neighbouring pixels around the unnormalised coordinate
// $u = (x + 1) \cdot (W_{\text{in}} - 1) / 2$,
// $v = (y + 1) \cdot (H_{\text{in}} - 1) / 2$ (under ``align_corners_=true``)
// are combined with the standard bilinear weights; analogous neighbourhoods
// of four (nearest) or sixteen (bicubic) pixels are used in the other modes.
//
// Out-of-bounds samples are handled by ``padding_mode_``: ``zeros`` returns
// $0$, ``border`` clamps the source coordinate, ``reflection`` reflects it
// off the image boundary.  The backward distributes ``grad_out`` back to both
// the input pixels (through the interpolation weights) and the grid (through
// the analytical derivative of the interpolation kernel).
//
// Math
// ----
// Coordinate unnormalisation:
// $$
//   u = \begin{cases}
//     (x + 1) (W_{\text{in}} - 1) / 2 & \text{align\_corners} \\
//     (x + 1) W_{\text{in}} / 2 - 0.5 & \text{otherwise}
//   \end{cases}
// $$
// (analogous formula for $v$).
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"grid_sample"``, ``AmpPolicy::Promote``.
// mode_ : int
//     Interpolation kernel: ``0`` = bilinear, ``1`` = nearest, ``2`` = bicubic.
// padding_mode_ : int
//     Out-of-bounds policy: ``0`` = zeros, ``1`` = border, ``2`` = reflection.
// align_corners_ : bool
//     Coordinate mapping mode (see file header).
// input_shape_ : Shape
//     Saved ``(N, C, H_in, W_in)`` for backward reconstruction.
// grid_shape_ : Shape
//     Saved ``(N, H_out, W_out, 2)`` for backward reconstruction.
//
// References
// ----------
// Jaderberg et al., "Spatial Transformer Networks" (NeurIPS 2015).
class LUCID_API GridSampleBackward : public FuncOp<GridSampleBackward, 2> {
public:
    static const OpSchema schema_v1;
    int mode_ = 0;          // 0=bilinear, 1=nearest, 2=bicubic.
    int padding_mode_ = 0;  // 0=zeros, 1=border, 2=reflection.
    bool align_corners_ = true;
    Shape input_shape_;  // Saved for backward shape reconstruction.
    Shape grid_shape_;

    // Resample ``input`` at the positions specified by ``grid``.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     4-D input image of shape ``(N, C, H_in, W_in)``.
    // grid : TensorImplPtr
    //     Sampling grid of shape ``(N, H_out, W_out, 2)`` with normalised
    //     coordinates in $[-1, 1]$.
    // mode : int
    //     Interpolation kernel code (``0/1/2`` for bilinear/nearest/bicubic).
    // padding_mode : int
    //     Out-of-bounds policy code (``0/1/2`` for zeros/border/reflection).
    // align_corners : bool
    //     Coordinate mapping mode.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Resampled tensor of shape ``(N, C, H_out, W_out)``.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& grid,
                                 int mode,
                                 int padding_mode,
                                 bool align_corners);

    // Backward pass: gradients w.r.t. ``input`` and ``grid``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public affine-grid entry point.
//
// Thin wrapper that delegates to ``AffineGridBackward::forward``.
//
// Parameters
// ----------
// theta : TensorImplPtr
//     Affine matrices of shape ``(N, 2, 3)``.
// N : int
//     Batch size (must equal ``theta.shape()[0]``).
// H : int
//     Output grid height.
// W : int
//     Output grid width.
// align_corners : bool
//     Lattice placement mode.
//
// Returns
// -------
// TensorImplPtr
//     Sampling grid of shape ``(N, H, W, 2)``.
LUCID_API TensorImplPtr
affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);

// Public grid-sample entry point.
//
// Thin wrapper that delegates to ``GridSampleBackward::forward``.
//
// Parameters
// ----------
// input : TensorImplPtr
//     4-D input image ``(N, C, H_in, W_in)``.
// grid : TensorImplPtr
//     Sampling grid ``(N, H_out, W_out, 2)`` with coordinates in $[-1, 1]$.
// mode : int
//     Interpolation kernel code (``0/1/2`` for bilinear/nearest/bicubic).
// padding_mode : int
//     Out-of-bounds policy code (``0/1/2`` for zeros/border/reflection).
// align_corners : bool
//     Coordinate mapping mode.
//
// Returns
// -------
// TensorImplPtr
//     Resampled tensor of shape ``(N, C, H_out, W_out)``.
LUCID_API TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                                       const TensorImplPtr& grid,
                                       int mode,
                                       int padding_mode,
                                       bool align_corners);

}  // namespace lucid
