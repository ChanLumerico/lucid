// lucid/_C/optim/Optimizer.h
//
// Abstract base class for all parameter optimizers in the Lucid engine.
// An Optimizer owns a flat list of shared TensorImpl parameter pointers
// and drives the training loop by calling update_one() per parameter on
// each optimizer step. Per-parameter state (momentum buffers, moment
// estimates, etc.) is managed by derived classes and keyed by a slot
// index that corresponds to the position of the parameter in params_.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class TensorImpl;

// Snapshot a state Storage into a fresh TensorImpl with deep-copied data.
// Defined in Optimizer.cpp; used by derived classes' state_buffers().
LUCID_API std::shared_ptr<TensorImpl>
clone_state_storage(const Storage& src, const Shape& shape, Dtype dtype, Device device);

// Overwrite ``dst`` with bytes from ``src`` in place.  Both must already share
// shape/dtype/device — only the buffer contents are updated.
LUCID_API void overwrite_state_storage(Storage& dst, const Storage& src);

// Owns a collection of trainable parameters and provides the canonical
// training-step interface.
//
// Derived classes override update_one() with their specific update
// mathematics and init_state_slot() to lazily allocate per-parameter
// state the first time a gradient is seen. The base step() method
// handles gradient presence checks, lazy state initialization, and
// version bump bookkeeping so derived classes need not repeat that
// logic. Copying is deleted because optimizer state is not safely
// copyable (state Storages hold raw heap buffers).
class LUCID_API Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<TensorImpl>> params)
        : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    // Iterate over all parameters; for each with an available gradient,
    // lazily initialize per-parameter state on the first call, invoke
    // update_one(), then bump the parameter's version counter to
    // invalidate any stale autograd views.
    void step();

    // Zero all gradient Storage values so the next backward pass
    // accumulates into fresh buffers.
    void zero_grad();

    std::size_t num_params() const { return params_.size(); }

    // Allow schedulers to update the learning rate between optimizer steps.
    virtual void set_lr(double lr) = 0;
    virtual double lr() const = 0;

    // Short identifier for checkpoint serialization (e.g., "sgd_v1").
    virtual std::string state_dict_id() const = 0;

    // Snapshot of the optimizer's per-parameter mutable state, exposed as a
    // list of (name, tensors) pairs.  ``tensors`` runs parallel to ``params_``;
    // entries are clones, so callers may freely keep them after the optimizer
    // is destroyed.  Optimizers with no Python-visible state return an empty
    // vector.
    using NamedBuffers = std::pair<std::string, std::vector<std::shared_ptr<TensorImpl>>>;
    virtual std::vector<NamedBuffers> state_buffers() const { return {}; }

    // Restore the per-parameter state in-place from a snapshot.  ``bufs`` must
    // match the layout returned by ``state_buffers`` (same names, same per-slot
    // shapes/dtypes); mismatches raise.  After this returns, every state slot
    // is treated as initialised so ``step()`` will not re-zero it.
    virtual void load_state_buffers(const std::vector<NamedBuffers>& bufs) { (void)bufs; }

    // Global iteration counter — Adam uses it for bias correction, NAdam for
    // its momentum schedule, etc.  Optimizers without one return 0 and ignore
    // the setter.
    virtual std::int64_t step_count() const { return 0; }
    virtual void set_step_count(std::int64_t) {}

protected:
    // Apply the optimizer's update rule for a single parameter.
    // slot_idx is the stable index of the parameter in params_, used to
    // look up pre-allocated state buffers.
    virtual void
    update_one(std::size_t slot_idx, std::shared_ptr<TensorImpl>& param, const Storage& grad) = 0;

    // Allocate and zero-initialize per-parameter state (e.g., momentum
    // buffers, second-moment estimates) for the given slot. Called once
    // per parameter on the first optimizer step that has a gradient.
    virtual void init_state_slot(std::size_t slot_idx,
                                 const std::shared_ptr<TensorImpl>& param) = 0;

    std::vector<std::shared_ptr<TensorImpl>> params_;

    // Parallel to params_; tracks whether init_state_slot has been called
    // for each slot so state is only allocated once.
    std::vector<bool> state_initialized_;
};

}  // namespace lucid
