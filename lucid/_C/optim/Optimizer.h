#pragma once

// =====================================================================
// Lucid C++ engine — Optimizer base (Phase 4).
// =====================================================================
//
// An Optimizer holds:
//   1. A list of parameter TensorImplPtrs to update
//   2. Per-parameter state buffers (momentum, Adam moments, etc.)
//   3. Hyperparameters (lr, betas, weight_decay, …)
//
// Lifecycle:
//   - Constructed with a parameter list and hyperparameters.
//   - Each training step: compute grads via the autograd engine, then call
//     `step()` to update parameters in-place. Call `zero_grad()` to reset
//     gradient storage between iterations.
//   - State buffers are lazy-initialized on the first `step()` call so
//     constructors don't allocate device memory until needed.
//
// Threading:
//   Not thread-safe across threads. Optimizers are designed to be called
//   from a single training loop. To shard gradient computation across
//   workers, accumulate grads into per-tensor `grad_storage_` (already
//   thread-safe via accumulate_into) then call step() from the main thread.
//
// In-place semantics:
//   step() modifies `param->storage_` directly and bumps `param->version_`
//   to invalidate any saved_inputs that referenced it. A backward pass
//   that re-runs after step() and was holding a saved_input snapshot will
//   correctly throw VersionMismatch (Item #9 retrofit).
//
// Layer: optim/. Depends on core/, autograd/, backend/.

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

    /// Apply one parameter update. Each subclass implements the actual
    /// formula. This base method handles common bookkeeping: skipping
    /// params whose grad isn't populated, bumping version counters, and
    /// dispatching to the subclass's `update_one()` per parameter.
    void step();

    /// Clear `grad_storage_` on every managed parameter.
    void zero_grad();

    /// Number of parameters under management.
    std::size_t num_params() const { return params_.size(); }

    /// Hyperparameter mutation (used by LR schedulers).
    virtual void set_lr(double lr) = 0;
    virtual double lr() const = 0;

    /// Optional state hooks for checkpoint/restore (Phase 5.5+).
    virtual std::string state_dict_id() const = 0;

protected:
    /// Subclass implements the per-parameter update. `param->storage_` is
    /// modified in place; `grad` is the read-only gradient storage. The
    /// `slot_idx` identifies which parameter this is — used to look up
    /// per-parameter state buffers (e.g. moments[slot_idx]).
    virtual void update_one(std::size_t slot_idx,
                            std::shared_ptr<TensorImpl>& param,
                            const Storage& grad) = 0;

    /// Subclass hook to lazily allocate state for a single parameter on
    /// its first appearance. Called from `step()` when a slot is unseen.
    virtual void init_state_slot(std::size_t slot_idx,
                                 const std::shared_ptr<TensorImpl>& param) = 0;

    std::vector<std::shared_ptr<TensorImpl>> params_;

    // Tracks which slots have already had state initialized.
    std::vector<bool> state_initialized_;
};

}  // namespace lucid
