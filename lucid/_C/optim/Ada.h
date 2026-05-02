#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_;
    std::vector<Storage> u_;
    std::int64_t step_count_;
};

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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eps_, weight_decay_, initial_accumulator_value_;
    std::vector<Storage> sum_sq_grad_;
};

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
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, rho_, eps_, weight_decay_;
    std::vector<Storage> sq_avg_;
    std::vector<Storage> accumulated_update_;
};

}  // namespace lucid
