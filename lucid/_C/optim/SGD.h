#pragma once

// =====================================================================
// Lucid C++ engine — SGD-family optimizers (mirrors lucid/optim/sgd.py).
// =====================================================================
//
// Contains:
//   - SGD   (vanilla / momentum / Nesterov / weight decay)
//   - ASGD  (Averaged SGD — running average of params past `t0`)

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// =====================================================================
// SGD
// =====================================================================
//
//   SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
//
// Per-step update (matches PyTorch torch.optim.SGD):
//   if weight_decay != 0:  g ← g + wd · param
//   if momentum != 0:
//     buf ← momentum · buf + (1 − dampening) · g       (init buf = g)
//     if nesterov:  g ← g + momentum · buf
//     else:         g ← buf
//   param ← param − lr · g
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
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;
    std::vector<Storage> moment_;
};

// =====================================================================
// ASGD — Averaged SGD (Lucid Python flavor; not bit-equiv to PyTorch's
// dynamic-eta variant)
// =====================================================================
//
//   if wd != 0: g ← g + wd · param
//   if mom != 0: buf ← mom · buf + g; g ← buf
//   param ← param − lr · g
//   if step >= t0:
//     coef = 1 / (alpha · step + 1)
//     ax ← (1 − coef) · ax + coef · param − lambd · ax
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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, momentum_, weight_decay_, alpha_, t0_, lambd_;
    std::vector<Storage> moment_;
    std::vector<Storage> ax_;
    std::vector<std::int64_t> step_;
};

}  // namespace lucid
