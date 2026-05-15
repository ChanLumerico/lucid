// lucid/_C/nn/PoolNd.h
//
// Autograd-aware N-dimensional MaxPool and AvgPool for N = 1, 2, 3.
//
// MaxPoolNdBackward saves the argmax indices from the forward pass so that
// the backward pass can route gradients only to the winning element in each
// pooling window (max-unpooling / scatter).
//
// AvgPoolNdBackward does not need to save any activation; the gradient is
// simply divided equally among all elements in each window.
//
// Stride defaults to the kernel size (non-overlapping windows) when passed
// as 0 from the entry-point functions.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for N-dimensional Max Pooling.
//
// saved_argmax_ contains flat per-output-element indices into the padded
// input buffer.  These are produced by IBackend::max_pool_nd_forward and
// consumed by IBackend::max_pool_nd_backward.
template <int N>

class LUCID_API MaxPoolNdBackward : public FuncOp<MaxPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];              // Pooling window size per axis.
    int stride_[N];         // Stride per axis (already resolved from 0 in forward).
    int pad_[N];            // Zero-padding per axis.
    Storage saved_argmax_;  // Flat argmax indices, same shape as output.

    // If stride_in[i] == 0 the stride defaults to K[i] (non-overlapping).
    static TensorImplPtr
    forward(const TensorImplPtr& x, const int (&K)[N], const int (&stride)[N], const int (&pad)[N]);

    // Backward: scatter grad_out into the positions given by saved_argmax_.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for N-dimensional Average Pooling.
//
// No activations need to be saved; the backward is a uniform scatter.
template <int N>

class LUCID_API AvgPoolNdBackward : public FuncOp<AvgPoolNdBackward<N>, 1> {
public:
    static const OpSchema schema_v1;
    int K_[N];
    int stride_[N];
    int pad_[N];

    // If stride_in[i] == 0 the stride defaults to K[i].
    static TensorImplPtr
    forward(const TensorImplPtr& x, const int (&K)[N], const int (&stride)[N], const int (&pad)[N]);

    // Backward: distribute grad_out evenly over each pooling window.
    std::vector<Storage> apply(Storage grad_out) override;
};

using MaxPool1dBackward = MaxPoolNdBackward<1>;
using MaxPool2dBackward = MaxPoolNdBackward<2>;
using MaxPool3dBackward = MaxPoolNdBackward<3>;
using AvgPool1dBackward = AvgPoolNdBackward<1>;
using AvgPool2dBackward = AvgPoolNdBackward<2>;
using AvgPool3dBackward = AvgPoolNdBackward<3>;

// stride == 0 means "use kernel size".
LUCID_API TensorImplPtr max_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);

LUCID_API TensorImplPtr max_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

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

LUCID_API TensorImplPtr avg_pool1d_op(const TensorImplPtr& x,
                                      int KL,
                                      int stride_l = 0,
                                      int pad_l = 0);

LUCID_API TensorImplPtr avg_pool2d_op(const TensorImplPtr& x,
                                      int KH,
                                      int KW,
                                      int stride_h = 0,
                                      int stride_w = 0,
                                      int pad_h = 0,
                                      int pad_w = 0);

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
