#pragma once

// =====================================================================
// Lucid C++ engine — fused loss kernels.
// =====================================================================
//
// All losses share a `Reduction` enum. The fixed-arity FuncOp template
// requires every input slot to be filled, so the Python wrappers fill
// missing optional weights with ones tensors (no extra grad cost since
// ones tensors have requires_grad=False).
//
//   mse_loss(input, target, reduction)
//   bce_loss(input, target, weight, reduction, eps)
//   bce_with_logits(input, target, weight, pos_weight, reduction)
//   cross_entropy(input, target, weight, reduction, eps, ignore_index)
//   nll_loss(input, target, weight, reduction, ignore_index)
//   huber_loss(input, target, delta, reduction)
//
// Targets for cross_entropy / nll_loss are class indices (any integer
// dtype); for the other losses they are F32/F64 of input shape.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

// MSE loss — autograd flows to (input, target).
/// Autograd backward node for MseLoss.
class LUCID_API MseLossBackward : public FuncOp<MseLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

// BCE — input must be in [0, 1]. Slots: 0=input, 1=target, 2=weight.
/// Autograd backward node for BCELoss.
class LUCID_API BCELossBackward : public FuncOp<BCELossBackward, 3> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 Reduction reduction,
                                 double eps);
    std::vector<Storage> apply(Storage grad_out) override;
};

// BCE with logits — slots: 0=input, 1=target, 2=weight, 3=pos_weight.
/// Autograd backward node for BCEWithLogits.
class LUCID_API BCEWithLogitsBackward : public FuncOp<BCEWithLogitsBackward, 4> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Cross-entropy = LogSoftmax + NLL fused. Only `input` participates in
// autograd (target is integer, weight typically not differentiable).
/// Autograd backward node for CrossEntropy.
class LUCID_API CrossEntropyBackward : public FuncOp<CrossEntropyBackward, 1> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;
    int ignore_index_ = -100;
    bool has_weight_ = false;
    Shape orig_input_shape_;
    Storage saved_softmax_;
    Storage saved_target_;
    Storage saved_weight_;
    Storage saved_valid_count_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 double eps,
                                 int ignore_index);
    std::vector<Storage> apply(Storage grad_out) override;
};

// NLL — input is log-probabilities. Only `input` differentiable.
/// Autograd backward node for NLLLoss.
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
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 int ignore_index);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Huber — slots: 0=input, 1=target.
/// Autograd backward node for HuberLoss.
class LUCID_API HuberLossBackward : public FuncOp<HuberLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double delta_ = 1.0;
    Shape orig_shape_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 double delta,
                                 Reduction reduction);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Mse loss.
LUCID_API TensorImplPtr mse_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    int reduction);
/// Bce loss.
LUCID_API TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight,
                                    int reduction,
                                    double eps);
/// Bce with logits.
LUCID_API TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                           const TensorImplPtr& target,
                                           const TensorImplPtr& weight,
                                           const TensorImplPtr& pos_weight,
                                           int reduction);
/// Cross entropy.
LUCID_API TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                                         const TensorImplPtr& target,
                                         const TensorImplPtr& weight_or_null,
                                         int reduction,
                                         double eps,
                                         int ignore_index);
/// Nll loss.
LUCID_API TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight_or_null,
                                    int reduction,
                                    int ignore_index);
/// Huber loss.
LUCID_API TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                                      const TensorImplPtr& target,
                                      double delta,
                                      int reduction);

}  // namespace lucid
