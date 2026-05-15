// lucid/_C/nn/BatchNorm.h
//
// Autograd-aware Batch Normalization for 1-D, 2-D, and 3-D spatial inputs.
// The template parameter N is the number of spatial dimensions: 1 for
// sequences (B, C, L), 2 for images (B, C, H, W), 3 for volumes (B, C, D, H, W).
//
// This header exposes the training-path forward only; running-stats inference
// is handled separately in NormExt (BatchNormEvalBackward).  Running statistics
// are updated on the Python side via tensor._copy_from() after each forward call.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Batch Normalization.
//
// Parameterized by N, the number of spatial dimensions.  The template is
// explicitly instantiated for N=1,2,3 in the .cpp file.
// saved_mean_ and saved_rstd_ are per-channel statistics computed over the
// (B, S0, S1, ...) axes and needed by the backward pass.
template <int N>

class LUCID_API BatchNormNdBackward : public FuncOp<BatchNormNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    Storage saved_mean_;    // Per-channel mean, shape (C,).
    Storage saved_rstd_;    // Per-channel reciprocal std, shape (C,).
    int B_ = 0, C_ = 0;     // Batch and channel counts.
    int S_[N > 0 ? N : 1];  // Spatial sizes (guard against N==0).

    // Run the training-mode forward pass.
    // x     – input of shape (B, C, S0, ..., S_{N-1}).
    // gamma – scale of shape (C,).
    // beta  – shift of shape (C,).
    // eps   – numerical stability constant.
    // Returns normalized output with the same shape as x.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Compute gradients for x (dx), gamma (d_gamma), and beta (d_beta).
    std::vector<Storage> apply(Storage grad_out) override;
};

using BatchNorm1dBackward = BatchNormNdBackward<1>;
using BatchNorm2dBackward = BatchNormNdBackward<2>;
using BatchNorm3dBackward = BatchNormNdBackward<3>;

// Batch normalization for 1-D spatial inputs (B, C, L).
LUCID_API TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);

// Batch normalization for 2-D spatial inputs (B, C, H, W).
LUCID_API TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps = 1e-5);

// Batch normalization for 3-D spatial inputs (B, C, D, H, W).
LUCID_API TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5);

}  // namespace lucid
