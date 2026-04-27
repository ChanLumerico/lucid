#pragma once

// =====================================================================
// Lucid C++ engine — Dropout (inverted-dropout convention).
// =====================================================================
//
//   dropout(x, p, training, generator)
//
//   if !training:  y = x   (no graph wiring needed beyond identity)
//   if  training:  m = Bernoulli(1 - p),  y = x * m / (1 - p)
//
// Inverted convention scales at training time so inference is a pure
// identity (no scaling). Saves the mask for backward.
//
// Backward: dx = g * mask / (1 - p)
//
// AMP policy: KeepInput. Determinism: respects `lucid.set_deterministic`
// indirectly via the `Generator` passed in (the generator's sequence is
// reproducible by seed).
//
// Layer: autograd/ops/nn/.

#include "../api.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"
#include "../autograd/FuncOp.h"

namespace lucid {

class Generator;

class LUCID_API DropoutBackward : public FuncOp<DropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;     // drop probability
    Storage mask_;        // saved during training forward

    static TensorImplPtr forward(const TensorImplPtr& a, double p,
                                 bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Channel-wise dropout (dropout1d/2d/3d). Input shape (B, C, *spatial).
// Mask shape (B, C, 1, 1, ..., 1) so an entire feature map is dropped together.
class LUCID_API DropoutNdBackward : public FuncOp<DropoutNdBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // (B, C, 1, ..., 1) scaled by 1/(1-p)

    static TensorImplPtr forward(const TensorImplPtr& a, double p,
                                 bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// SELU-friendly dropout. Replaces dropped values with the saturating
// negative SELU constant rescaled so input mean/var are preserved.
//   y = a · (x · mask + α' · (1 - mask)) + b
// where α' = −λα, a = ((1−p)(1 + p·α'²))⁻¹ᐟ², b = −a·p·α'
class LUCID_API AlphaDropoutBackward : public FuncOp<AlphaDropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    double a_coef_ = 1.0;
    Storage mask_;

    static TensorImplPtr forward(const TensorImplPtr& a, double p,
                                 bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// DropBlock — spatial structured dropout for 4-D feature maps.
// Samples a Bernoulli mask at gamma rate, dilates each "1" into a
// block_size × block_size square, then drops the union. Output is
// rescaled by the kept fraction.
class LUCID_API DropBlockBackward : public FuncOp<DropBlockBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // saved final block_mask (already scaled)

    static TensorImplPtr forward(const TensorImplPtr& a, std::int64_t block_size,
                                 double p, double eps, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

// DropPath (stochastic depth). Per-sample dropout: mask shape
// (B, 1, 1, ..., 1). If scale_by_keep, divide by keep_prob.
class LUCID_API DropPathBackward : public FuncOp<DropPathBackward, 1> {
public:
    static const OpSchema schema_v1;
    Storage mask_;

    static TensorImplPtr forward(const TensorImplPtr& a, double p,
                                 bool scale_by_keep, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Public API. `gen=nullptr` falls back to `default_generator()`.
LUCID_API TensorImplPtr dropout_op(const TensorImplPtr& a, double p,
                                   bool training, Generator* gen);
LUCID_API TensorImplPtr dropoutnd_op(const TensorImplPtr& a, double p,
                                      bool training, Generator* gen);
LUCID_API TensorImplPtr alpha_dropout_op(const TensorImplPtr& a, double p,
                                          bool training, Generator* gen);
LUCID_API TensorImplPtr drop_block_op(const TensorImplPtr& a,
                                       std::int64_t block_size, double p,
                                       double eps, Generator* gen);
LUCID_API TensorImplPtr drop_path_op(const TensorImplPtr& a, double p,
                                      bool scale_by_keep, Generator* gen);

}  // namespace lucid
