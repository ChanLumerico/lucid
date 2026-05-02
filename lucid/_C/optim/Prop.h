#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, alpha_, eps_, weight_decay_, momentum_;
    bool centered_;
    std::vector<Storage> square_avg_;
    std::vector<Storage> grad_avg_;
    std::vector<Storage> moment_buf_;
};

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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eta_minus_, eta_plus_, step_min_, step_max_;
    std::vector<Storage> prev_grad_;
    std::vector<Storage> step_size_;
};

}  // namespace lucid
