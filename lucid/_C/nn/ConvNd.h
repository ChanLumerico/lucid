#pragma once

// =====================================================================
// Lucid C++ engine — N-D convolution (1D / 2D / 3D unified).
// =====================================================================
//
//   conv{1,2,3}d(x, W, b, stride, padding, dilation, groups)
//     x : (B, C_in,  *S)   where |S| = N
//     W : (C_out, C_in/G, *K) where |K| = N
//     b : (C_out,)
//     y : (B, C_out, *O)  with O[i] = (S[i] + 2·pad[i] - eff_K[i])/stride[i] + 1
//                          eff_K[i] = dilation[i] · (K[i] - 1) + 1
//
// Groups split the input channel axis into G slices and the output
// channel axis the same way; each group's slice runs an independent
// conv. C_in must be divisible by G; C_out must be divisible by G.
// Weight stores (C_out, C_in/G, *K).
//
// CPU forward dispatches into the rank-specific im2col / col2im kernels
// in `backend/cpu/Im2Col.{h,cpp}` per group. GPU forward / backward
// route through `mlx::core::conv_general` which natively accepts
// dilation and groups.
//
// AMP policy: Promote.

#include <vector>

#include "../api.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"
#include "../autograd/FuncOp.h"

namespace lucid {

template <int N>
class LUCID_API ConvNdBackward : public FuncOp<ConvNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];
    int pad_[N];
    int dilation_[N];
    int groups_ = 1;

    static TensorImplPtr forward(const TensorImplPtr& x, const TensorImplPtr& W,
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

LUCID_API TensorImplPtr conv1d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_l = 1, int pad_l = 0,
                                  int dilation_l = 1, int groups = 1);
LUCID_API TensorImplPtr conv2d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_h = 1, int stride_w = 1,
                                  int pad_h = 0, int pad_w = 0,
                                  int dilation_h = 1, int dilation_w = 1,
                                  int groups = 1);
LUCID_API TensorImplPtr conv3d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int stride_d = 1, int stride_h = 1, int stride_w = 1,
                                  int pad_d = 0, int pad_h = 0, int pad_w = 0,
                                  int dilation_d = 1, int dilation_h = 1, int dilation_w = 1,
                                  int groups = 1);

// im2col / col2im exposed as standalone ops for nn.functional.unfold.
//   unfold(x, kernel_size, stride, padding, dilation)
//     x : (B, C, *S)
//     y : (B, C·prod(K), prod(O))   — the column buffer used by conv.
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
