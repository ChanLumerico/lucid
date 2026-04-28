#pragma once

// =====================================================================
// Lucid C++ engine — vision / utility kernels.
// =====================================================================
//
//   one_hot(input, num_classes)
//     Integer indices → one-hot tensor.  Output shape: input.shape + [C].
//     No autograd (integer input, output is constant given the index).
//
//   rotate(input, angle, cy, cx)
//     2-D image rotation (N, C, H, W) by `angle` degrees around (cy, cx).
//     Nearest-neighbor sampling; out-of-bounds → 0.
//     No autograd (discrete pixel remap).
//
//   bilinear_layer(x1, x2, weight, bias)
//     Learned bilinear layer: y[..., k] = sum_{i,j} x1[i] * weight[k,i,j] *
//     x2[j] + bias[k].  Forward + backward; differentiable in all inputs.

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API BilinearLayerBackward : public FuncOp<BilinearLayerBackward, 4> {
public:
    static const OpSchema schema_v1;
    Shape orig_x1_shape_;
    Shape orig_x2_shape_;
    static TensorImplPtr forward(const TensorImplPtr& x1,
                                 const TensorImplPtr& x2,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& bias);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes, Dtype out_dtype);
LUCID_API TensorImplPtr rotate_op(const TensorImplPtr& input,
                                  double angle_deg,
                                  double cy,
                                  double cx);
LUCID_API TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                          const TensorImplPtr& x2,
                                          const TensorImplPtr& weight,
                                          const TensorImplPtr& bias);

}  // namespace lucid
