#pragma once

// =====================================================================
// Lucid C++ engine — interpolate (bilinear / trilinear / nearest).
// =====================================================================
//
//   interpolate_bilinear (4-D)   input: [N, C, H, W]   → [N, C, H_out, W_out]
//   interpolate_trilinear (5-D)  input: [N, C, D, H, W] → [N, C, D_out, H_out, W_out]
//   interpolate_nearest (4-D)    same shapes; no autograd
//   interpolate_nearest_3d (5-D) same shapes; no autograd
//
// align_corners=True : src = i * (in - 1) / (out - 1)
// align_corners=False: src = (i + 0.5) * in / out - 0.5  (then clamp)
//
// "area" is implemented as a Python composition over avg_pool — no kernel
// needed here.

#include "../api.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"
#include "../autograd/FuncOp.h"

namespace lucid {

class LUCID_API InterpolateBilinearBackward
    : public FuncOp<InterpolateBilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int H_in_ = 0, W_in_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                  int H_out, int W_out, bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API InterpolateTrilinearBackward
    : public FuncOp<InterpolateTrilinearBackward, 1> {
public:
    static const OpSchema schema_v1;
    int D_in_ = 0, H_in_ = 0, W_in_ = 0;
    int D_out_ = 0, H_out_ = 0, W_out_ = 0;
    bool align_corners_ = false;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                  int D_out, int H_out, int W_out,
                                  bool align_corners);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr interpolate_bilinear_op(const TensorImplPtr& input,
                                                  int H_out, int W_out,
                                                  bool align_corners);
LUCID_API TensorImplPtr interpolate_trilinear_op(const TensorImplPtr& input,
                                                   int D_out, int H_out, int W_out,
                                                   bool align_corners);
// Nearest-neighbor variants (no autograd — round is non-differentiable).
LUCID_API TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input,
                                                    int H_out, int W_out);
LUCID_API TensorImplPtr interpolate_nearest_3d_op(const TensorImplPtr& input,
                                                    int D_out, int H_out, int W_out);

}  // namespace lucid
