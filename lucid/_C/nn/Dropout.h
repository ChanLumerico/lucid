// lucid/_C/nn/Dropout.h
//
// Five stochastic regularization operators, each with its own autograd node:
//   Dropout        – element-wise Bernoulli mask, inverted scaling.
//   DropoutNd      – channel-wise Bernoulli mask, same scale as Dropout.
//   AlphaDropout   – SELU-preserving dropout; keeps mean and variance.
//   DropBlock      – spatial block dropout for 4-D (N,C,H,W) feature maps.
//   DropPath       – per-sample path dropout used in vision transformers.
//
// All variants require an explicit Generator when `set_deterministic(true)` is
// active; otherwise they fall back to the process-global default generator.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

// Autograd node for standard element-wise dropout.
//
// The scaled Bernoulli mask (mask * 1/(1-p)) is retained in mask_ for use
// during backward; the gradient is simply multiplied by the same mask.
// When p==0 or training==false the node is wired with an empty mask and the
// backward pass clones grad_out unchanged.
class LUCID_API DropoutBackward : public FuncOp<DropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;  // Drop probability (0 = pass-through).
    Storage mask_;    // Scaled Bernoulli mask saved for backward.

    // Apply Bernoulli mask to a, scale surviving elements by 1/(1-p).
    // gen may be null (uses default_generator()); throws under deterministic
    // mode when gen is null and p > 0.
    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for channel-wise Dropout (DropoutNd in reference parlance).
//
// A single Bernoulli value is drawn per (batch, channel) pair and broadcast
// over all spatial dimensions, so entire feature-map channels are zeroed.
// mask_ stores the full-resolution expanded-and-scaled mask for backward.
class LUCID_API DropoutNdBackward : public FuncOp<DropoutNdBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // Full-resolution mask after broadcast (same shape as input).

    // Expects input of rank >= 2, layout (N, C, ...).
    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for alpha-dropout (SELU-compatible).
//
// Dropped elements are replaced with kAlphaPrime (≈ -1.7581) and the output
// is affinely rescaled so that the distribution mean and variance are
// preserved.  a_coef_ = (keep*(1+p*alpha'^2))^{-0.5} is saved for backward.
class LUCID_API AlphaDropoutBackward : public FuncOp<AlphaDropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    double a_coef_ = 1.0;  // Affine scale factor saved for backward.
    Storage mask_;         // Raw Bernoulli mask (0/1, before scaling).

    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);

    // Backward: grad_in = grad_out * (mask * a_coef_).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for DropBlock (structured spatial dropout on 4-D inputs).
//
// A seed mask of Bernoulli samples is generated; IBackend::drop_block_mask
// dilates them into contiguous blocks of size block_size x block_size.
// Input must be 4-D (N, C, H, W).
class LUCID_API DropBlockBackward : public FuncOp<DropBlockBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // Keep mask after dilation (0/1 float, same shape as input).

    // block_size – side length of each dropped block.
    // eps        – added to the denominator when computing gamma to avoid div/0.
    static TensorImplPtr
    forward(const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for DropPath (stochastic depth, per-sample).
//
// A single Bernoulli value is drawn per sample in the batch and broadcast
// over all other dimensions, so entire residual paths are dropped together.
// When scale_by_keep is true, surviving paths are scaled by 1/(1-p).
class LUCID_API DropPathBackward : public FuncOp<DropPathBackward, 1> {
public:
    static const OpSchema schema_v1;
    Storage mask_;  // Per-sample mask broadcast to full input shape.

    // scale_by_keep – divide surviving elements by keep probability.
    static TensorImplPtr
    forward(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen);

LUCID_API TensorImplPtr dropoutnd_op(const TensorImplPtr& a,
                                     double p,
                                     bool training,
                                     Generator* gen);

LUCID_API TensorImplPtr alpha_dropout_op(const TensorImplPtr& a,
                                         double p,
                                         bool training,
                                         Generator* gen);

LUCID_API TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);

LUCID_API TensorImplPtr drop_path_op(const TensorImplPtr& a,
                                     double p,
                                     bool scale_by_keep,
                                     Generator* gen);

}  // namespace lucid
