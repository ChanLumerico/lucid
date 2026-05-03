// lucid/_C/nn/Spatial.h
//
// Autograd-aware spatial transformation operations: affine grid generation and
// grid sampling (differentiable image resampling).
//
// AffineGridBackward: generates a sampling grid from an affine transformation
//   matrix theta of shape (N, 2, 3).  The output grid has shape (N, H, W, 2)
//   where each position holds normalized (x, y) sample coordinates.
//
// GridSampleBackward: resamples a 4-D input image (N, C, H_in, W_in) at
//   positions specified by a grid (N, H_out, W_out, 2).  Supports bilinear,
//   nearest, and bicubic interpolation modes (mode_ integer), and zero, border,
//   or reflection padding modes (padding_mode_).
//
// These two operations are the building blocks of Spatial Transformer Networks.

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
// theta must be (N, 2, 3); output shape is (N, H, W, 2).
// align_corners_ affects the coordinate mapping: when true, corners of the
// grid map to corners of the input image; when false, they map to pixel centers.
class LUCID_API AffineGridBackward : public FuncOp<AffineGridBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool align_corners_ = true;
    int N_ = 0, H_ = 0, W_ = 0;
    Shape orig_theta_shape_;

    // theta – (N, 2, 3) affine matrices.  N, H, W are the target output dims.
    static TensorImplPtr
    forward(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for grid sampling (differentiable image resampling).
//
// input – (N, C, H_in, W_in); grid – (N, H_out, W_out, 2) with coordinates
// in [-1, 1].  mode_ and padding_mode_ are integer codes forwarded to the backend.
class LUCID_API GridSampleBackward : public FuncOp<GridSampleBackward, 2> {
public:
    static const OpSchema schema_v1;
    int mode_ = 0;          // 0=bilinear, 1=nearest, 2=bicubic.
    int padding_mode_ = 0;  // 0=zeros, 1=border, 2=reflection.
    bool align_corners_ = true;
    Shape input_shape_;  // Saved for backward shape reconstruction.
    Shape grid_shape_;

    // Returns output of shape (N, C, H_out, W_out).
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& grid,
                                 int mode,
                                 int padding_mode,
                                 bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: delegates to AffineGridBackward::forward.
LUCID_API TensorImplPtr
affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);

// Public entry point: delegates to GridSampleBackward::forward.
LUCID_API TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                                       const TensorImplPtr& grid,
                                       int mode,
                                       int padding_mode,
                                       bool align_corners);

}  // namespace lucid
