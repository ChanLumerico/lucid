// lucid/_C/optim/Ada.h
//
// Adaptive gradient optimizers: Adamax, Adagrad, and Adadelta. These
// three optimizers share the property of adapting per-parameter step
// sizes based on historical gradient information, but differ in how
// they accumulate that history and apply it.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// Adamax optimizer (Kingma & Ba, 2014).
//
// A variant of Adam that uses the infinity norm of gradients for the
// second-moment estimate instead of the L2 norm. The u_ buffer holds
// the element-wise maximum of beta2 * u and |g|:
//   u = max(beta2 * u, |g|)
// This makes the effective step size bounded and the update rule:
//   p -= (lr / (1 - beta1^t)) * m / (u + eps)
// The infinity-norm variant is more stable for embeddings and sparse
// gradients where some elements receive very large updates occasionally.
class LUCID_API Adamax : public Optimizer {
public:
    Adamax(std::vector<std::shared_ptr<TensorImpl>> params,
           double lr = 2e-3,
           double beta1 = 0.9,
           double beta2 = 0.999,
           double eps = 1e-8,
           double weight_decay = 0.0);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "adamax_v1"; }

protected:
    // Apply the Adamax update: advance m via EMA, update u via max-norm,
    // then take the bias-corrected step.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialized m (first moment) and u (inf-norm) buffers.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_; // Per-parameter first-moment estimates.
    std::vector<Storage> u_; // Per-parameter infinity-norm estimates.
    std::int64_t step_count_;
};

// Adagrad optimizer (Duchi et al., 2011).
//
// Accumulates the sum of squared gradients in sum_sq_grad_ and divides
// the learning rate by its square root. Unlike RMSprop there is no
// forgetting factor — the accumulator grows monotonically, so the
// effective learning rate decreases to zero over time. This is
// beneficial for sparse features but can be too aggressive for dense
// neural network weights.
//
// initial_accumulator_value_ allows seeding sum_sq_grad_ with a
// positive constant to avoid division by a near-zero value on the
// first step.
class LUCID_API Adagrad : public Optimizer {
public:
    Adagrad(std::vector<std::shared_ptr<TensorImpl>> params,
            double lr = 1e-2,
            double eps = 1e-10,
            double weight_decay = 0.0,
            double initial_accumulator_value = 0.0);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "adagrad_v1"; }

protected:
    // Accumulate g^2 into sum_sq_grad_, then apply p -= lr * g / (sqrt(ss) + eps).
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate sum_sq_grad_ and, if initial_accumulator_value_ != 0,
    // fill it with that scalar value rather than zeros.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eps_, weight_decay_, initial_accumulator_value_;
    // Per-parameter cumulative sum of squared gradients.
    std::vector<Storage> sum_sq_grad_;
};

// Adadelta optimizer (Zeiler, 2012).
//
// Removes the need for a global learning rate by maintaining two
// running averages:
//   sq_avg_:           running average of squared gradients
//   accumulated_update_: running average of squared parameter updates
// The update is computed as:
//   sq_avg = rho * sq_avg + (1 - rho) * g^2
//   update = sqrt(accumulated_update + eps) / sqrt(sq_avg + eps) * g
//   accumulated_update = rho * accumulated_update + (1 - rho) * update^2
//   p -= lr * update
// The lr parameter acts as a global scale factor rather than the
// primary step size controller; it is often set to 1.0.
class LUCID_API Adadelta : public Optimizer {
public:
    Adadelta(std::vector<std::shared_ptr<TensorImpl>> params,
             double lr = 1.0,
             double rho = 0.9,
             double eps = 1e-6,
             double weight_decay = 0.0);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "adadelta_v1"; }

protected:
    // Apply the Adadelta update: advance both running averages and
    // compute the adaptive step.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialized sq_avg_ and accumulated_update_ buffers.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, rho_, eps_, weight_decay_;
    // Per-parameter running average of squared gradients.
    std::vector<Storage> sq_avg_;
    // Per-parameter running average of squared parameter updates.
    std::vector<Storage> accumulated_update_;
};

}  // namespace lucid
