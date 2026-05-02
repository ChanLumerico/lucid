#pragma once

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
         bool amsgrad = false);

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
