#pragma once

// =====================================================================
// Lucid C++ engine — Adam-family optimizers (mirrors lucid/optim/adam.py).
// =====================================================================
//
// Contains:
//   - Adam   (Kingma & Ba, 2014; with optional L2 weight decay)
//   - AdamW  (decoupled weight decay; Loshchilov & Hutter 2017)
//   - NAdam  (Nesterov-accelerated Adam, with momentum_decay schedule)
//   - RAdam  (Rectified Adam; Liu et al. 2020)
//
// Adam formula:
//     m ← β1·m + (1 − β1)·g           (first moment)
//     v ← β2·v + (1 − β2)·g²          (second moment)
//     m̂ ← m / (1 − β1^t)              (bias correction)
//     v̂ ← v / (1 − β2^t)
//     param ← param − lr · m̂ / (√v̂ + ε)
//
// AdamW: wd is applied directly to the parameter (not folded into g):
//     param ← param · (1 − lr·wd) − lr · m̂ / (√v̂ + ε)
//
// State per parameter: two buffers (m, v) of the same shape as the
// parameter, allocated lazily on first step. `step_count` is global per
// optimizer (matches PyTorch behavior).

#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

class LUCID_API Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0,
         /*decoupled wd =*/bool amsgrad = false);
    // amsgrad is reserved for future support; Phase 4 does not use it.

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    double beta1() const { return beta1_; }
    double beta2() const { return beta2_; }
    double eps() const { return eps_; }

    std::string state_dict_id() const override { return "adam_v1"; }

protected:
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    bool amsgrad_;
    std::int64_t step_count_;

    std::vector<Storage> m_;
    std::vector<Storage> v_;
};

class LUCID_API AdamW : public Optimizer {
public:
    AdamW(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 1e-2);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }

    std::string state_dict_id() const override { return "adamw_v1"; }

protected:
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    std::int64_t step_count_;

    std::vector<Storage> m_;
    std::vector<Storage> v_;
};

// =====================================================================
// NAdam — Nesterov-accelerated Adam (PyTorch parameterization)
// =====================================================================
class LUCID_API NAdam : public Optimizer {
public:
    NAdam(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 2e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0,
          double momentum_decay = 0.004);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "nadam_v1"; }

protected:
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_, momentum_decay_;
    std::vector<Storage> m_, v_;
    std::vector<double> mu_product_;
    std::int64_t step_count_;
};

// =====================================================================
// RAdam — Rectified Adam (Liu et al. 2020). Falls back to plain SGD-with-
// momentum when the variance estimator is unstable (rho_t ≤ 5).
// =====================================================================
class LUCID_API RAdam : public Optimizer {
public:
    RAdam(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0);

    void set_lr(double lr) override { lr_ = lr; }
    double lr() const override { return lr_; }
    std::string state_dict_id() const override { return "radam_v1"; }

protected:
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_, v_;
    std::int64_t step_count_;
};

}  // namespace lucid
