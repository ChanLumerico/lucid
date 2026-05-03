// lucid/_C/nn/ConvNd.h
//
// Autograd-aware N-dimensional convolution (N = 1, 2, 3) and the im2col-based
// Unfold operation.
//
// ConvNdBackward<N> implements the standard cross-correlation:
//   out[b, c_out, o...] = sum_{c_in, k...} x[b, c_in, s*o+d*k...] * W[c_out, c_in/g, k...]
//                         + b[c_out]
// where s = stride, d = dilation, g = groups (depthwise-style grouping).
//
// CPU backend uses im2col + GEMM via Apple Accelerate.
// GPU backend uses MLX convolutional primitives.
//
// UnfoldBackward extracts sliding local patches from the input into a column
// matrix of shape (B, C*prod(K), prod(O)); this is the explicit im2col step
// used by custom conv implementations in user code.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for N-dimensional convolution.
//
// Explicitly instantiated for N=1,2,3.  stride_, pad_, and dilation_ arrays
// have fixed size N to avoid heap allocation on the hot path.
// groups_ divides both C_in and C_out into independent filter groups.
template <int N>

class LUCID_API ConvNdBackward : public FuncOp<ConvNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];    // Convolution stride per spatial axis.
    int pad_[N];       // Zero-padding per spatial axis.
    int dilation_[N];  // Dilation (hole size) per spatial axis.
    int groups_ = 1;   // Number of filter groups.

    // Run the forward pass.
    // x       – input (B, C_in, S_0, ..., S_{N-1}).
    // W       – weight (C_out, C_in/groups, K_0, ..., K_{N-1}).
    // b       – bias (C_out,).
    // stride  – per-axis stride.
    // pad     – per-axis zero-padding.
    // dilation – per-axis dilation.
    // groups  – filter grouping; C_in and C_out must both be divisible.
    // Returns output of shape (B, C_out, O_0, ..., O_{N-1}).
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&dilation)[N],
                                 int groups);

    // Compute gradients for x (dx), W (dW), and b (db).
    std::vector<Storage> apply(Storage grad_out) override;
};

using Conv1dBackward = ConvNdBackward<1>;
using Conv2dBackward = ConvNdBackward<2>;
using Conv3dBackward = ConvNdBackward<3>;

// 1-D convolution: x (B, C_in, L), W (C_out, C_in/g, KL).
LUCID_API TensorImplPtr conv1d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_l = 1,
                                  int pad_l = 0,
                                  int dilation_l = 1,
                                  int groups = 1);

// 2-D convolution: x (B, C_in, H, W), W (C_out, C_in/g, KH, KW).
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

// 3-D convolution: x (B, C_in, D, H, W), W (C_out, C_in/g, KD, KH, KW).
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

// Autograd node for the Unfold (im2col) operation.
//
// Extracts all kernel-sized patches from the input and lays them out as
// columns.  Output shape: (B, C * prod(K), prod(O)).
// Supports 1-D, 2-D, and 3-D inputs.
class LUCID_API UnfoldBackward : public FuncOp<UnfoldBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<int> kernel_;    // Kernel sizes per spatial axis.
    std::vector<int> stride_;    // Strides per spatial axis.
    std::vector<int> pad_;       // Padding per spatial axis.
    std::vector<int> dilation_;  // Dilations per spatial axis.

    // x must be (B, C, S_0, ..., S_{N-1}) with N matching kernel.size().
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& stride,
                                 const std::vector<int>& pad,
                                 const std::vector<int>& dilation);

    // Backward is the fold (col2im) operation.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point for Unfold.
LUCID_API TensorImplPtr unfold_op(const TensorImplPtr& x,
                                  const std::vector<int>& kernel,
                                  const std::vector<int>& stride,
                                  const std::vector<int>& pad,
                                  const std::vector<int>& dilation);

}  // namespace lucid
