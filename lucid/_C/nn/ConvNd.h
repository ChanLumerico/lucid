#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

template <int N>

class LUCID_API ConvNdBackward : public FuncOp<ConvNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];
    int pad_[N];
    int dilation_[N];
    int groups_ = 1;

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&dilation)[N],
                                 int groups);
    std::vector<Storage> apply(Storage grad_out) override;
};

using Conv1dBackward = ConvNdBackward<1>;
using Conv2dBackward = ConvNdBackward<2>;
using Conv3dBackward = ConvNdBackward<3>;

LUCID_API TensorImplPtr conv1d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_l = 1,
                                  int pad_l = 0,
                                  int dilation_l = 1,
                                  int groups = 1);

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

class LUCID_API UnfoldBackward : public FuncOp<UnfoldBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<int> kernel_;
    std::vector<int> stride_;
    std::vector<int> pad_;
    std::vector<int> dilation_;

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& stride,
                                 const std::vector<int>& pad,
                                 const std::vector<int>& dilation);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr unfold_op(const TensorImplPtr& x,
                                  const std::vector<int>& kernel,
                                  const std::vector<int>& stride,
                                  const std::vector<int>& pad,
                                  const std::vector<int>& dilation);

}  // namespace lucid
