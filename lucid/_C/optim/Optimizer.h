#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class TensorImpl;

class LUCID_API Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<TensorImpl>> params)
        : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    void step();

    void zero_grad();

    std::size_t num_params() const { return params_.size(); }

    virtual void set_lr(double lr) = 0;
    virtual double lr() const = 0;

    virtual std::string state_dict_id() const = 0;

protected:
    virtual void
    update_one(std::size_t slot_idx, std::shared_ptr<TensorImpl>& param, const Storage& grad) = 0;

    virtual void init_state_slot(std::size_t slot_idx,
                                 const std::shared_ptr<TensorImpl>& param) = 0;

    std::vector<std::shared_ptr<TensorImpl>> params_;

    std::vector<bool> state_initialized_;
};

}  // namespace lucid
