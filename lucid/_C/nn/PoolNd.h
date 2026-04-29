#pragma once

// =====================================================================
// Lucid C++ engine — N-D pooling (1D / 2D / 3D unified).
// =====================================================================
//
//   {max,avg}_pool{1,2,3}d(x, kernel, stride, padding)
//     x : (B, C, *S)   |S| = N
//     y : (B, C, *O)   O[i] = (S[i] + 2·pad[i] - K[i])/stride[i] + 1
//
// MaxPool saves an int32 argmax index per output position; backward
// scatters g into those positions. AvgPool has no saved tensor (just
// kernel/stride/pad); backward distributes g/prod(K) uniformly.
//
// AMP policy: KeepInput.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

template <int N>
/// Autograd backward node for MaxPoolNd.
class LUCID_API MaxPoolNdBackward : public FuncOp<MaxPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];
    int stride_[N];
    int pad_[N];
    Storage saved_argmax_;  // int32; shape (B, C, *O)

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const int (&K)[N],
                                 const int (&stride)[N],
                                 const int (&pad)[N]);
    std::vector<Storage> apply(Storage grad_out) override;
};

template <int N>
/// Autograd backward node for AvgPoolNd.
class LUCID_API AvgPoolNdBackward : public FuncOp<AvgPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];
    int stride_[N];
    int pad_[N];

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const int (&K)[N],
                                 const int (&stride)[N],
                                 const int (&pad)[N]);
    std::vector<Storage> apply(Storage grad_out) override;
};

using MaxPool1dBackward = MaxPoolNdBackward<1>;
using MaxPool2dBackward = MaxPoolNdBackward<2>;
using MaxPool3dBackward = MaxPoolNdBackward<3>;
using AvgPool1dBackward = AvgPoolNdBackward<1>;
using AvgPool2dBackward = AvgPoolNdBackward<2>;
using AvgPool3dBackward = AvgPoolNdBackward<3>;

/// Max pool1d.
LUCID_API TensorImplPtr max_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);
/// Max pool2d.
LUCID_API TensorImplPtr max_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);
/// Max pool3d.
LUCID_API TensorImplPtr max_pool3d_op(const TensorImplPtr& x,
                                      int KD,
                                      int KH,
                                      int KW,
                                      int stride_d = 0,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_d = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);
/// Avg pool1d.
LUCID_API TensorImplPtr avg_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);
/// Avg pool2d.
LUCID_API TensorImplPtr avg_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);
/// Avg pool3d.
LUCID_API TensorImplPtr avg_pool3d_op(const TensorImplPtr& x,
                                      int KD,
                                      int KH,
                                      int KW,
                                      int stride_d = 0,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_d = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

}  // namespace lucid
