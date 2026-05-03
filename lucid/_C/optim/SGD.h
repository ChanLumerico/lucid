// lucid/_C/optim/SGD.h
//
// Stochastic Gradient Descent and Averaged SGD optimizers.
// Both classes derive from Optimizer and implement their update rules
// on CPU (scalar loop over raw buffers) and GPU (MLX array ops).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// SGD with optional momentum, weight decay, and Nesterov acceleration.
//
// Update rule (with momentum):
//   v_t  = momentum * v_{t-1} + (1 - dampening) * g_t
//   p_t  = p_{t-1} - lr * (g_t + momentum * v_t)   [Nesterov]
//        = p_{t-1} - lr * v_t                        [classical]
//
// Without momentum the rule simplifies to p -= lr * g (with optional
// L2 weight-decay term added to g before the step).
//
// Invariants: lr >= 0, momentum >= 0, weight_decay >= 0. Nesterov
// requires momentum > 0 and dampening == 0 (enforced in constructor).
// moment_ is sized lazily on the first step; it is empty when
// momentum == 0 because no velocity buffer is needed.
class LUCID_API SGD : public Optimizer {
public:
    SGD(std::vector<std::shared_ptr<TensorImpl>> params,
        double lr,
        double momentum = 0.0,
        double dampening = 0.0,
        double weight_decay = 0.0,
        bool nesterov = false);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    double momentum() const { return momentum_; }
    double weight_decay() const { return weight_decay_; }

    std::string state_dict_id() const override { return "sgd_v1"; }

protected:
    // Execute the SGD or SGD-with-momentum update for a single parameter,
    // dispatching to the CPU scalar loop or the MLX GPU path.
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Allocate a zero-filled velocity buffer for this slot when momentum != 0.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;
    // Per-parameter velocity buffers; entry i is active only when momentum != 0.
    std::vector<Storage> moment_;
};

// Averaged SGD (ASGD) with optional momentum.
//
// Runs standard SGD with an additional running average ax_ of the
// parameter trajectory. The average is computed starting from step t0_
// using a decaying coefficient: ax = (1 - coef) * ax + coef * p - lambd * ax
// where coef = 1 / (alpha * step + 1). The averaged weights ax can be
// used for inference after training, providing lower variance estimates
// than the instantaneous weights.
class LUCID_API ASGD : public Optimizer {
public:
    ASGD(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-3,
         double momentum = 0.0,
         double weight_decay = 0.0,
         double alpha = 0.75,
         double t0 = 1e6,
         double lambd = 1e-4);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "asgd_v1"; }

protected:
    // Apply the ASGD update: standard SGD step followed by an update
    // of the running parameter average ax_ once the step count passes t0_.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate the velocity buffer (if momentum != 0) and the parameter
    // average buffer ax_, initializing ax_ to a copy of the current parameter.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, momentum_, weight_decay_, alpha_, t0_, lambd_;
    // Per-parameter SGD velocity buffers (active when momentum != 0).
    std::vector<Storage> moment_;
    // Per-parameter running averages of the parameter trajectory.
    std::vector<Storage> ax_;
    // Per-parameter step counters used to compute the averaging coefficient.
    std::vector<std::int64_t> step_;
};

}  // namespace lucid
