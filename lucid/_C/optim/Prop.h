// lucid/_C/optim/Prop.h
//
// Propagation-based adaptive optimizers: RMSprop and Rprop.
// RMSprop uses a running mean-square of gradients to normalize the
// learning rate; Rprop uses only the sign of the gradient with
// per-parameter adaptive step sizes based on gradient sign agreement.

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// RMSprop (Tieleman & Hinton, unpublished).
//
// Maintains a running mean square of gradients square_avg_ and divides
// the learning rate by its square root. Supports two optional
// extensions:
//   centered=true:  subtracts the squared running gradient mean from
//                   square_avg_ to obtain the variance (grad_avg_ is
//                   only allocated in this case).
//   momentum != 0:  accumulates scaled updates in moment_buf_ before
//                   applying them to the parameter (moment_buf_ is
//                   only allocated when momentum != 0).
//
// Update rule:
//   sq = alpha * sq + (1 - alpha) * g^2
//   avg = sq - ga^2  (centered) or sq (non-centered)
//   update = g / sqrt(avg + eps)
//   buf = momentum * buf + update   (when momentum != 0)
//   p -= lr * buf                   (or lr * update when no momentum)
class LUCID_API RMSprop : public Optimizer {
public:
    RMSprop(std::vector<std::shared_ptr<TensorImpl>> params,
            double lr = 1e-2,
            double alpha = 0.99,
            double eps = 1e-8,
            double weight_decay = 0.0,
            double momentum = 0.0,
            bool centered = false);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "rmsprop_v1"; }

protected:
    // Apply the RMSprop update for one parameter, dispatching to GPU
    // or the scalar CPU loop.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate square_avg_ (always), grad_avg_ (if centered),
    // and moment_buf_ (if momentum != 0).
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, alpha_, eps_, weight_decay_, momentum_;
    bool centered_;
    // Per-parameter running mean-square of gradients.
    std::vector<Storage> square_avg_;
    // Per-parameter running mean of gradients (centered mode only).
    std::vector<Storage> grad_avg_;
    // Per-parameter momentum accumulation buffer.
    std::vector<Storage> moment_buf_;
};

// Rprop (Riedmiller & Braun, 1993) — Resilient Backpropagation.
//
// Ignores gradient magnitudes entirely and adapts a per-parameter
// step size based solely on sign consistency between consecutive
// gradient values:
//   sign change (g * g_prev > 0): multiply step by eta_plus_
//   sign change (g * g_prev < 0): multiply step by eta_minus_,
//                                  and zero out the gradient to avoid
//                                  an overshooting correction on the next step
//   no change:                    keep step unchanged
// Step sizes are clamped to [step_min_, step_max_]. The parameter is
// then updated by subtracting sign(g) * step.
//
// prev_grad_ and step_size_ are the only per-parameter state.
// step_size_ is initialized to lr_ (not 1.0) so the initial step
// has a sensible scale.
class LUCID_API Rprop : public Optimizer {
public:
    Rprop(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-2,
          double eta_minus = 0.5,
          double eta_plus = 1.2,
          double step_min = 1e-6,
          double step_max = 50.0);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "rprop_v1"; }

protected:
    // Apply the Rprop step-size adaptation and parameter update.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate prev_grad_ (zero-initialized) and step_size_
    // (initialized to lr_ on every element).
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eta_minus_, eta_plus_, step_min_, step_max_;
    // Previous gradient, stored to detect sign changes.
    std::vector<Storage> prev_grad_;
    // Per-element adaptive step sizes.
    std::vector<Storage> step_size_;
};

}  // namespace lucid
