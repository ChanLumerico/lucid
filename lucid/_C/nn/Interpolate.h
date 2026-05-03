// lucid/_C/nn/Interpolate.h
//
// Autograd-aware image/volume interpolation: bilinear (2-D), trilinear (3-D),
// and nearest-neighbor (2-D and 3-D, no autograd).
//
// InterpolateBilinearBackward: resizes a 4-D (N, C, H_in, W_in) tensor to
//   (N, C, H_out, W_out) using bilinear interpolation.  align_corners affects
//   how source coordinates are mapped to destination pixels.
//
// InterpolateTrilinearBackward: resizes a 5-D (N, C, D_in, H_in, W_in) tensor
//   to (N, C, D_out, H_out, W_out) using trilinear interpolation.
//
// Nearest-neighbor variants produce fixed outputs (no backward node) and are
// provided for inference-only use cases.

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
// H_in_, W_in_ are saved from the input; H_out_, W_out_ are the requested
// output sizes.  The backward distributes gradients to the four surrounding
// input pixels weighted by the bilinear interpolation coefficients.
class LUCID_API InterpolateBilinearBackward : public FuncOp<InterpolateBilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int H_in_ = 0, W_in_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;  // Full (N, C, H_in, W_in) shape for backward.

    // input must be 4-D (N, C, H, W).
    static TensorImplPtr
    forward(const TensorImplPtr& input, int H_out, int W_out, bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for 3-D trilinear interpolation.
//
// input must be 5-D (N, C, D, H, W).  The backward distributes gradients to
// the eight surrounding voxels weighted by the trilinear interpolation weights.
class LUCID_API InterpolateTrilinearBackward : public FuncOp<InterpolateTrilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int D_in_ = 0, H_in_ = 0, W_in_ = 0;
    int D_out_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;

    // input must be 5-D (N, C, D, H, W).
    static TensorImplPtr
    forward(const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Bilinear resize with autograd.
LUCID_API TensorImplPtr interpolate_bilinear_op(const TensorImplPtr& input,
                                                int H_out,
                                                int W_out,
                                                bool align_corners);

// Trilinear resize with autograd.
LUCID_API TensorImplPtr interpolate_trilinear_op(
    const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners);

// Nearest-neighbor 2-D resize; no backward node is attached.
LUCID_API TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input, int H_out, int W_out);

// Nearest-neighbor 3-D resize; no backward node is attached.
LUCID_API TensorImplPtr interpolate_nearest_3d_op(const TensorImplPtr& input,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out);

}  // namespace lucid
