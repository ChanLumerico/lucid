// lucid/_C/nn/ConvTransposeNd.h
//
// Autograd-aware N-dimensional transposed convolution (deconvolution) for
// N = 1, 2, 3.
//
// The transposed convolution is the gradient of a regular convolution with
// respect to its input.  For each spatial axis i:
//   O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + K[i] + opad[i]
//
// The optional output_padding (opad) adds extra rows/columns to disambiguate
// the output size when multiple input sizes map to the same strided output.
// Weight layout: (C_in, C_out, K_0, ..., K_{N-1}) — note the transposed
// channel order relative to regular convolution.
//
// CPU uses col2im + GEMM; GPU uses the MLX transpose-conv primitive.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for N-dimensional transposed convolution.
//
// opad_[N] stores per-axis output padding used when reconstructing shapes
// during backward.
template <int N>

class LUCID_API ConvTransposeNdBackward : public FuncOp<ConvTransposeNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];  // Upsampling stride per axis.
    int pad_[N];     // Input padding removed from output per axis.
    int opad_[N];    // Output padding added to disambiguate output size.

    // Run the forward pass.
    // x      – input (B, C_in, S_0, ..., S_{N-1}).
    // W      – weight (C_in, C_out, K_0, ..., K_{N-1}).
    // b      – bias (C_out,).
    // stride – per-axis upsampling factor.
    // pad    – per-axis input padding (amount subtracted from O).
    // opad   – per-axis output padding (amount added to O); must be < stride.
    // Returns output of shape (B, C_out, O_0, ..., O_{N-1}).
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&opad)[N]);

    // Backward of the transposed convolution is a regular convolution.
    std::vector<Storage> apply(Storage grad_out) override;
};

using ConvTranspose1dBackward = ConvTransposeNdBackward<1>;
using ConvTranspose2dBackward = ConvTransposeNdBackward<2>;
using ConvTranspose3dBackward = ConvTransposeNdBackward<3>;

// 1-D transposed convolution.
LUCID_API TensorImplPtr conv_transpose1d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_l = 1,
                                            int pad_l = 0,
                                            int opad_l = 0);

// 2-D transposed convolution.
LUCID_API TensorImplPtr conv_transpose2d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

// 3-D transposed convolution.
LUCID_API TensorImplPtr conv_transpose3d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_d = 1,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_d = 0,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_d = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

}  // namespace lucid
