#pragma once

// =====================================================================
// Lucid C++ engine — affine_grid + grid_sample.
// =====================================================================
//
//   affine_grid(theta, N, H, W, align_corners)
//     Builds a sampling grid from a batch of 2x3 affine matrices.
//     theta:  (N, 2, 3), differentiable
//     output: (N, H, W, 2) with (x, y) sampling coordinates
//
//     align_corners=True : x_norm[w] = -1 + 2w/(W-1)
//     align_corners=False: x_norm[w] = -1 + (2w+1)/W
//
//   grid_sample(input, grid, mode, padding_mode, align_corners)
//     2-D image sampling with bilinear or nearest interpolation.
//     input:  (N, C, H_in, W_in), differentiable
//     grid:   (N, H_out, W_out, 2), differentiable (bilinear) / no-grad (nearest)
//     output: (N, C, H_out, W_out)
//
//     mode:         0 = bilinear, 1 = nearest
//     padding_mode: 0 = zeros, 1 = border

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API AffineGridBackward : public FuncOp<AffineGridBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool align_corners_ = true;
    int N_ = 0, H_ = 0, W_ = 0;
    Shape orig_theta_shape_;
    static TensorImplPtr forward(
        const TensorImplPtr& theta, int N, int H, int W, bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API GridSampleBackward : public FuncOp<GridSampleBackward, 2> {
public:
    static const OpSchema schema_v1;
    int mode_ = 0;
    int padding_mode_ = 0;
    bool align_corners_ = true;
    Shape input_shape_;
    Shape grid_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& grid,
                                 int mode,
                                 int padding_mode,
                                 bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr
affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);

LUCID_API TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                                       const TensorImplPtr& grid,
                                       int mode,
                                       int padding_mode,
                                       bool align_corners);

}  // namespace lucid
