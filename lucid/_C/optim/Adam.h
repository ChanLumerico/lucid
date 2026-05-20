// lucid/_C/optim/Adam.h
//
// Adam-family optimizers: Adam, AdamW, NAdam, and RAdam. All four
// maintain first-moment (m) and second-moment (v) estimates for each
// parameter and share a common low-level kernel (adam_step_cpu /
// adam_step_gpu) parameterized by flags and scalars.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "../api.h"
#include "../core/Dtype.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace mlx::core {
class array;
}  // namespace mlx::core

namespace lucid {

class TensorImpl;

// Per-step scalar cache for Adam-family GPU updates.  All 8 broadcast
// scalars (beta1, 1-beta1, beta2, 1-beta2, eps, lr, 1/bc1, 1/bc2) plus
// the optional weight-decay scalar are rebuilt **once per step** at the
// first parameter and reused across the remaining ~64 parameters of a
// typical ResNet-18 / transformer step.  Without this, ``adam_step_gpu``
// built ``params.size() * 8`` MLX arrays per call (520 for ResNet-18) —
// each MLX array construction has constant overhead, and the bias
// correction factors don't even depend on which parameter is updated.
//
// The cache invalidates when ``step_count_`` advances or the param
// dtype changes (mixed-precision case).
struct AdamScalarCache {
    bool valid = false;
    Dtype dt = Dtype::F32;
    std::int64_t for_step = 0;
    std::unique_ptr<mlx::core::array> b1;
    std::unique_ptr<mlx::core::array> omb1;
    std::unique_ptr<mlx::core::array> b2;
    std::unique_ptr<mlx::core::array> omb2;
    std::unique_ptr<mlx::core::array> eps_a;
    std::unique_ptr<mlx::core::array> lr_a;
    std::unique_ptr<mlx::core::array> wd_a;       // present iff weight_decay != 0
    std::unique_ptr<mlx::core::array> inv_bc1;
    std::unique_ptr<mlx::core::array> inv_bc2;
    std::unique_ptr<mlx::core::array> wd_factor;  // (1 - lr*wd) for AdamW

    AdamScalarCache();
    ~AdamScalarCache();
    AdamScalarCache(const AdamScalarCache&) = delete;
    AdamScalarCache& operator=(const AdamScalarCache&) = delete;
};

// Adam optimizer (Kingma & Ba, 2014).
//
// Maintains first-moment estimate m and second-moment estimate v for
// each parameter. Update rule after bias correction:
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   p -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Weight decay is added to the gradient before the moment update
// (L2 regularization form). amsgrad is declared but not yet
// implemented and will raise not_implemented at construction time.
//
// step_count_ is a global counter incremented once per step() call
// regardless of how many parameters are updated, so all parameters
// share the same bias-correction factor within a step.
class LUCID_API Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0,
         bool amsgrad = false);

    // 3.4 perf: invalidate the scalar cache on lr change so the
    // ``lr_a`` / ``wd_factor`` arrays are rebuilt next step.
    void set_lr(double lr) override {
        lr_ = lr;
        scalar_cache_.valid = false;
    }
    double lr() const override { return lr_; }
    double beta1() const { return beta1_; }
    double beta2() const { return beta2_; }
    double eps() const { return eps_; }

    std::string state_dict_id() const override { return "adam_v1"; }

    std::vector<NamedBuffers> state_buffers() const override;
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;
    std::int64_t step_count() const override { return step_count_; }
    void set_step_count(std::int64_t s) override { step_count_ = s; }

protected:
    // Apply the standard Adam update (decoupled_wd = false) to a single
    // parameter, dispatching to GPU or CPU.
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Allocate zero-initialized m and v buffers for this slot.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    bool amsgrad_;
    // Global step counter; bias correction factors are functions of this.
    std::int64_t step_count_;

    std::vector<Storage> m_;  // Per-parameter first-moment estimates.
    std::vector<Storage> v_;  // Per-parameter second-moment estimates.

    // 3.4 perf: see AdamScalarCache documentation above.
    AdamScalarCache scalar_cache_;
};

// AdamW optimizer (Loshchilov & Hutter, 2017).
//
// Identical to Adam except weight decay is applied directly to the
// parameter before the gradient step (decoupled weight decay), rather
// than being added to the gradient. This decoupling prevents weight
// decay from interacting with the adaptive learning-rate scaling.
//
// Update rule:
//   p *= (1 - lr * weight_decay)    // decoupled decay
//   p -= lr * m_hat / (sqrt(v_hat) + eps)
class LUCID_API AdamW : public Optimizer {
public:
    AdamW(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 1e-2);

    // 3.4 perf: invalidate the scalar cache on lr change (same rationale
    // as in ``Adam::set_lr``).
    void set_lr(double lr) override {
        lr_ = lr;
        scalar_cache_.valid = false;
    }
    double lr() const override { return lr_; }

    std::string state_dict_id() const override { return "adamw_v1"; }

    std::vector<NamedBuffers> state_buffers() const override;
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;
    std::int64_t step_count() const override { return step_count_; }
    void set_step_count(std::int64_t s) override { step_count_ = s; }

protected:
    // Apply the decoupled-weight-decay Adam update (decoupled_wd = true).
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Allocate zero-initialized m and v buffers for this slot.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    std::int64_t step_count_;

    std::vector<Storage> m_;
    std::vector<Storage> v_;

    // 3.4 perf: see AdamScalarCache documentation above.
    AdamScalarCache scalar_cache_;
};

// NAdam optimizer (Dozat, 2016) — Adam with Nesterov momentum.
//
// Replaces the standard momentum-corrected gradient term with a
// Nesterov look-ahead. The update decomposes into two terms:
//   term1 = lr * (1 - mu) / (1 - mu_prod) * g / denom
//   term2 = lr * mu_next / (1 - mu_prod_next) * m / denom
// where mu is a per-step decaying momentum coefficient and mu_prod is
// the running product of all mu values seen so far.
// mu_product_ is a per-parameter scalar, not a Storage, because it is
// a single floating-point accumulator rather than a tensor.
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
    // Compute per-step mu and mu_next, advance mu_product_, then apply
    // the two-term Nesterov update.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialized m, v buffers; initialize mu_product_ to 1.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_, momentum_decay_;
    std::vector<Storage> m_, v_;
    // Running product of all per-step momentum coefficients mu_t.
    std::vector<double> mu_product_;
    std::int64_t step_count_;
};

// RAdam optimizer (Liu et al., 2019) — Rectified Adam.
//
// Computes a variance-based rectification factor r_t. When the
// approximated SMA rho_t is large enough (> 5), the second-moment
// estimate is used and r_t corrects for the variance of the adaptive
// learning rate. When rho_t is too small, the update falls back to a
// plain momentum step using only the bias-corrected first moment.
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
    // Compute rho_t and the rectification factor r_t; apply the
    // adaptive (use_rect == true) or plain momentum update accordingly.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialized m and v buffers for this slot.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_, v_;
    std::int64_t step_count_;
};

}  // namespace lucid
