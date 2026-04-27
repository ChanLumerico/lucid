#pragma once

// =====================================================================
// Lucid C++ engine — N-D transposed convolution (1D / 2D / 3D unified).
// =====================================================================
//
//   conv_transpose{1,2,3}d(x, W, b, stride, padding, output_padding)
//     x : (B, C_in,  *S)        |S| = N
//     W : (C_in, C_out, *K)     |K| = N   (note: C_in first — PyTorch convention)
//     b : (C_out,)
//     y : (B, C_out, *O)
//        with O[i] = (S[i]-1)·stride[i] - 2·pad[i] + K[i] + output_padding[i]
//
// Forward is the spatial-inverse of regular convolution. Equivalent to:
//   - Insert (stride-1) zeros between input elements (input dilation)
//   - Pad the dilated input by (K - 1 - p)
//   - Convolve with stride=1
// We implement directly via sgemm + col2im on CPU, and via mlx::conv_transpose
// on GPU. Backward dx routes through a regular conv forward (since
// ConvTranspose's adjoint is regular conv); dW uses the same dilation trick
// as conv backward, with x and grad swapped.
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
class LUCID_API ConvTransposeNdBackward
    : public FuncOp<ConvTransposeNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];
    int pad_[N];
    int opad_[N];

    static TensorImplPtr forward(const TensorImplPtr& x, const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&opad)[N]);
    std::vector<Storage> apply(Storage grad_out) override;
};

using ConvTranspose1dBackward = ConvTransposeNdBackward<1>;
using ConvTranspose2dBackward = ConvTransposeNdBackward<2>;
using ConvTranspose3dBackward = ConvTransposeNdBackward<3>;

LUCID_API TensorImplPtr conv_transpose1d_op(
    const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b,
    int stride_l = 1, int pad_l = 0, int opad_l = 0);
LUCID_API TensorImplPtr conv_transpose2d_op(
    const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b,
    int stride_h = 1, int stride_w = 1,
    int pad_h = 0, int pad_w = 0,
    int opad_h = 0, int opad_w = 0);
LUCID_API TensorImplPtr conv_transpose3d_op(
    const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b,
    int stride_d = 1, int stride_h = 1, int stride_w = 1,
    int pad_d = 0, int pad_h = 0, int pad_w = 0,
    int opad_d = 0, int opad_h = 0, int opad_w = 0);

}  // namespace lucid
