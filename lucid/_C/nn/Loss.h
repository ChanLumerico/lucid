// lucid/_C/nn/Loss.h
//
// Autograd-aware loss functions.  All forward methods produce a scalar (or
// element-wise) loss tensor and attach the corresponding backward node when
// any input requires a gradient.  All operations are forced to FP32 to avoid
// precision loss during accumulation.
//
// Supported losses:
//   MseLoss          – mean squared error:  mean((input - target)^2).
//   BCELoss          – binary cross-entropy on probabilities in (0, 1).
//   BCEWithLogits    – numerically stable BCE applied directly to logits.
//   CrossEntropy     – log-softmax + NLL; saves the softmax probabilities.
//   NLLLoss          – negative log-likelihood on log-probability inputs.
//   HuberLoss        – smooth L1 loss parameterized by delta.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Reduction mode applied after per-element loss computation.
enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

// Autograd node for Mean Squared Error loss.
//
// Saves {input, target}.  Backward: grad_input = 2*(input-target)*grad_out/N
// (or *grad_out for Sum; no scaling for None).
class LUCID_API MseLossBackward : public FuncOp<MseLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;  // Element-wise shape before reduction.

    // input and target must have identical shapes and dtypes.
    static TensorImplPtr
    forward(const TensorImplPtr& input, const TensorImplPtr& target, Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Binary Cross-Entropy loss.
//
// Saves {input, target, weight}.  eps_ clamps the log argument away from 0.
// Backward: grad_input = weight * (-target/input + (1-target)/(1-input)) * grad_out / N.
class LUCID_API BCELossBackward : public FuncOp<BCELossBackward, 3> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;  // Clamp added to log argument for numerical safety.
    Shape orig_shape_;

    // weight must have the same shape as input; pass a ones tensor if unused.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 Reduction reduction,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for BCE-with-logits (sigmoid + BCE fused for stability).
//
// Saves {input, target, weight, pos_weight}.  The computation is:
//   max(x, 0) - x*y + log(1 + exp(-|x|)) weighted by pos_weight.
class LUCID_API BCEWithLogitsBackward : public FuncOp<BCEWithLogitsBackward, 4> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;

    // pos_weight re-weights positive examples; pass a ones tensor if unused.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Cross-Entropy loss (log-softmax + NLL).
//
// The backend computes softmax probabilities internally and returns them in
// result.saved_aux; they are stored in saved_softmax_ for the backward pass.
// saved_valid_count_ holds the number of non-ignored samples for the Mean
// reduction denominator.  Only the input tensor is wired as a saved edge;
// target and (optional) weight are stored directly in Storage fields.
class LUCID_API CrossEntropyBackward : public FuncOp<CrossEntropyBackward, 1> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;
    int ignore_index_ = -100;  // Class index whose samples do not contribute.
    bool has_weight_ = false;
    Shape orig_input_shape_;
    Storage saved_softmax_;      // Per-sample softmax probabilities.
    Storage saved_target_;       // Integer class labels.
    Storage saved_weight_;       // Optional per-class weights.
    Storage saved_valid_count_;  // Count of non-ignored samples (for Mean).

    // input must be (N, C, ...).  weight_or_null may be null (no class weights).
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 double eps,
                                 int ignore_index);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Negative Log-Likelihood loss.
//
// Expects log-probabilities as input (output of log-softmax).  Backward:
// grad_input[b, t[b], ...] = -weight[t[b]] * grad_out[b] / valid_count.
class LUCID_API NLLLossBackward : public FuncOp<NLLLossBackward, 1> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    int ignore_index_ = -100;
    bool has_weight_ = false;
    Shape orig_input_shape_;
    Storage saved_target_;
    Storage saved_weight_;
    Storage saved_valid_count_;

    // input must be (N, C, ...).
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 int ignore_index);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Huber (smooth L1) loss.
//
// For |r| <= delta: 0.5 * r^2.  For |r| > delta: delta * (|r| - 0.5*delta).
// delta_ must be positive; the default is 1.0 (equivalent to SmoothL1Loss).
class LUCID_API HuberLossBackward : public FuncOp<HuberLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double delta_ = 1.0;
    Shape orig_shape_;

    // Throws if delta <= 0.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 double delta,
                                 Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry points; reduction is passed as int and cast to Reduction internally.
LUCID_API TensorImplPtr mse_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    int reduction);

LUCID_API TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight,
                                    int reduction,
                                    double eps);

LUCID_API TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                           const TensorImplPtr& target,
                                           const TensorImplPtr& weight,
                                           const TensorImplPtr& pos_weight,
                                           int reduction);

LUCID_API TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                                         const TensorImplPtr& target,
                                         const TensorImplPtr& weight_or_null,
                                         int reduction,
                                         double eps,
                                         int ignore_index);

LUCID_API TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight_or_null,
                                    int reduction,
                                    int ignore_index);

LUCID_API TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                                      const TensorImplPtr& target,
                                      double delta,
                                      int reduction);

// Connectionist Temporal Classification loss.
// log_probs: (T, N, C); targets: flat (N*S,) int32;
// input_lengths / target_lengths: (N,) int32.
// Returns per-sample losses (N,) — apply reduction in Python.
LUCID_API TensorImplPtr ctc_loss_op(const TensorImplPtr& log_probs,
                                     const TensorImplPtr& targets,
                                     const TensorImplPtr& input_lengths,
                                     const TensorImplPtr& target_lengths,
                                     int blank,
                                     bool zero_infinity);

}  // namespace lucid
