#pragma once

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
    static TensorImplPtr
    forward(const TensorImplPtr& theta, int N, int H, int W, bool align_corners);
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
