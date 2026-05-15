// lucid/_C/nn/Vision.h
//
// Miscellaneous vision utilities:
//
//   one_hot_op         – encode integer labels as one-hot float vectors; no grad.
//   rotate_op          – rotate a 4-D image batch by a fixed angle; no grad.
//   BilinearLayerBackward – autograd-aware bilinear form y = x1 @ W @ x2^T + b,
//                       where W has shape (D_out, D1, D2).

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the bilinear layer: y = x1 @ W @ x2^T + b.
//
// W must be (D_out, D1, D2); x1 last dim must equal D1; x2 last dim must
// equal D2.  All leading "batch" dimensions of x1 and x2 must match.
// bias may be null; the backward detects a null bias by checking whether the
// saved bias shape (input_shapes_[3]) is empty.
class LUCID_API BilinearLayerBackward : public FuncOp<BilinearLayerBackward, 4> {
public:
    static const OpSchema schema_v1;
    Shape orig_x1_shape_;  // Saved for backward shape reconstruction.
    Shape orig_x2_shape_;

    // bias may be null (no bias term); pass nullptr if unused.
    static TensorImplPtr forward(const TensorImplPtr& x1,
                                 const TensorImplPtr& x2,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& bias);
    std::vector<Storage> apply(Storage grad_out) override;
};

// One-hot encode integer indices into vectors of length num_classes.
// Returns shape (*input.shape, num_classes) with dtype out_dtype; no grad.
LUCID_API TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes, Dtype out_dtype);

// Rotate a 4-D image batch (N, C, H, W) by angle_deg degrees around (cy, cx).
// The rotation is performed in-place per channel; no backward node is attached.
LUCID_API TensorImplPtr rotate_op(const TensorImplPtr& input,
                                  double angle_deg,
                                  double cy,
                                  double cx);

// Public entry point: delegates to BilinearLayerBackward::forward.
LUCID_API TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                          const TensorImplPtr& x2,
                                          const TensorImplPtr& weight,
                                          const TensorImplPtr& bias);

}  // namespace lucid
