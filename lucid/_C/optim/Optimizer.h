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

// Deep-copy an optimizer state buffer into a new owning TensorImpl.
//
// Snapshots a state ``Storage`` (CPU or GPU) into a freshly allocated
// ``TensorImpl`` whose backing buffer is independent of the original.
// This is the primitive used by derived optimizers when implementing
// ``state_buffers()`` so that the returned tensors survive any future
// in-place updates by ``step()``.
//
// Parameters
// ----------
// src : const Storage&
//     Source state buffer.  Must be either ``CpuStorage`` or
//     ``GpuStorage``.  For ``GpuStorage`` the underlying MLX array is
//     materialised via ``eval()`` before copying.
// shape : const Shape&
//     Logical shape to attach to the returned ``TensorImpl``.  Must
//     match the element count implied by ``src``.
// dtype : Dtype
//     Dtype tag to record on the new ``TensorImpl``.
// device : Device
//     Device tag to record on the new ``TensorImpl``.
//
// Returns
// -------
// std::shared_ptr<TensorImpl>
//     A non-leaf tensor whose storage is a deep copy of ``src``.
//
// Raises
// ------
// std::runtime_error
//     If ``src`` is neither a ``CpuStorage`` nor a ``GpuStorage``.
//
// Notes
// -----
// CPU copies use ``allocate_aligned_bytes`` followed by ``memcpy``;
// GPU copies build an independent MLX array.  In both cases the
// resulting buffer can be mutated freely without affecting the source
// optimizer state.
//
// See Also
// --------
// overwrite_state_storage : in-place inverse used by
//     ``load_state_buffers``.
LUCID_API std::shared_ptr<TensorImpl>
clone_state_storage(const Storage& src, const Shape& shape, Dtype dtype, Device device);

// Overwrite an existing state buffer in place from a saved snapshot.
//
// Used by ``Optimizer::load_state_buffers`` to restore a checkpoint
// into the optimizer's live state without reallocating or rebinding
// any storage objects.  ``dst`` keeps its identity (pointer, shape,
// dtype, device) — only the underlying bytes / MLX array are replaced.
//
// Parameters
// ----------
// dst : Storage&
//     Live optimizer state buffer to update in place.
// src : const Storage&
//     Snapshot whose contents should be written into ``dst``.  Must be
//     on the same device as ``dst`` and have the same byte size.
//
// Raises
// ------
// std::runtime_error
//     If ``dst`` and ``src`` are on different devices, or if their
//     CPU byte sizes disagree.
//
// See Also
// --------
// clone_state_storage : produces the snapshots consumed here.
LUCID_API void overwrite_state_storage(Storage& dst, const Storage& src);

// Abstract base class owning trainable parameters and the training step.
//
// ``Optimizer`` keeps a flat list of ``TensorImpl`` parameters and runs
// one update per training iteration.  Derived classes inject their
// specific update mathematics via ``update_one`` and lazily allocate
// per-parameter state (momentum buffers, moment estimates, parameter
// averages, ...) via ``init_state_slot``.  The base class handles the
// cross-cutting bookkeeping: skipping parameters without gradients,
// allocating state exactly once per slot, and bumping each parameter's
// version counter so autograd notices the in-place mutation.
//
// State storage layout
// --------------------
// Per-parameter state is keyed by an integer ``slot_idx`` that equals
// the position of the parameter in ``params_``.  Derived classes hold
// one ``std::vector<Storage>`` (or several) parallel to ``params_``;
// entry ``i`` is allocated on the first call to ``update_one`` for
// slot ``i``.  The parallel ``state_initialized_`` vector tracks
// allocation so state is allocated exactly once.
//
// Step contract
// -------------
// ``step()`` is called by the Python wrapper once per training iteration,
// *after* the backward pass has populated ``param->grad_storage()``.
// It performs three actions per slot:
//
// 1. Skip the slot if the parameter pointer is null or its gradient is
//    absent (matching the reference framework's "no grad = no update"
//    convention).
// 2. Lazily allocate state via ``init_state_slot`` if this is the slot's
//    first observed gradient.
// 3. Apply the update via ``update_one`` and bump the parameter's
//    version counter via ``TensorImpl::bump_version`` so any autograd
//    node that captured the parameter before the update sees the change.
//
// ``zero_grad()`` clears every parameter's gradient storage so the next
// backward pass starts from a fresh zero buffer.
//
// Copy semantics
// --------------
// Copy is deleted because optimizer state owns raw heap buffers (CPU)
// and MLX arrays (GPU) whose semantics under shallow copy would be
// ambiguous.  Optimizers are owned by the Python wrapper through a
// ``std::shared_ptr`` and accessed via pybind11 reference bindings.
//
// Serialisation
// -------------
// ``state_dict_id()`` returns a short version-tagged identifier (e.g.
// ``"sgd_v1"``) baked into checkpoints so the Python loader can refuse
// mismatched versions.  ``state_buffers()`` / ``load_state_buffers()``
// expose the per-parameter mutable state as named ``TensorImpl`` lists
// for ``torch.save``-style checkpointing — entries are clones, so the
// caller is free to keep them after the optimizer is destroyed.
//
// Subclass responsibilities
// -------------------------
// Concrete optimizers must override ``update_one``, ``init_state_slot``,
// ``set_lr`` / ``lr``, and ``state_dict_id``.  Optimizers with global
// step state (Adam, NAdam, ...) must additionally override
// ``step_count`` / ``set_step_count``.  Optimizers with Python-visible
// per-parameter state must override ``state_buffers`` /
// ``load_state_buffers``.
//
// Attributes
// ----------
// params_ : std::vector<std::shared_ptr<TensorImpl>>
//     Flat list of trainable parameters.  Indexed by ``slot_idx`` in
//     the protected hooks.  May contain null entries; nulls are
//     silently skipped by ``step()``.
// state_initialized_ : std::vector<bool>
//     Parallel to ``params_``.  Marks whether ``init_state_slot`` has
//     run for each slot; grown lazily by ``step()`` to match
//     ``params_.size()`` on first use.
//
// See Also
// --------
// SGD, ASGD : plain and averaged stochastic gradient descent.
// Adam, AdamW, NAdam, RAdam, Adamax, Adafactor : adaptive moment family.
// Adagrad, Adadelta, RMSProp : adaptive learning-rate family.
class LUCID_API Optimizer {
public:
    // Construct an optimizer from a flat list of trainable parameters.
    //
    // Parameters
    // ----------
    // params : std::vector<std::shared_ptr<TensorImpl>>
    //     Parameters to optimise.  May be empty (a no-op optimizer) and
    //     may contain null entries (silently skipped by ``step``).  The
    //     caller's vector is moved into the optimizer; subsequent
    //     mutations of the caller's copy do not affect this optimizer.
    //
    // Notes
    // -----
    // ``state_initialized_`` is sized lazily on the first ``step()``
    // rather than here, which lets the Python wrapper extend ``params_``
    // post-construction in the rare param-group reconfiguration case.
    explicit Optimizer(std::vector<std::shared_ptr<TensorImpl>> params)
        : params_(std::move(params)) {}

    // Virtual destructor — ensures derived ``Adam`` / ``SGD`` / ``RMSProp``
    // destructors run when held by a base-class pointer.  Defaulted; the
    // per-parameter state buffers are released by their own ``std::vector``
    // members in the derived classes, and ``params_`` releases its
    // ``shared_ptr`` refs automatically.
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    // Run one optimizer update over all registered parameters.
    //
    // For each non-null parameter that has an accumulated gradient,
    // ``step()`` lazily initialises its state slot, invokes the derived
    // class's ``update_one`` to apply the update rule, then bumps the
    // parameter's autograd version counter.  Parameters without a
    // gradient are skipped — this matches the reference framework's
    // "no grad = no update" semantics and lets a single optimizer
    // safely cover partially active sub-modules (e.g. frozen layers).
    //
    // Notes
    // -----
    // Called once per training iteration, after ``loss.backward()``.
    // The accompanying Python wrapper takes care of forcing the
    // resulting MLX graph to ``eval()`` only when requested via
    // ``AUTO_EVAL_AFTER_STEP``; by default ``step()`` is itself lazy
    // so that forward → backward → step can fuse into one MLX
    // submission.
    void step();

    // Zero every parameter's gradient storage.
    //
    // Calls ``TensorImpl::zero_grad`` on each non-null parameter.
    // Should be called before each forward/backward pass so that the
    // next backward accumulates into a clean buffer rather than into
    // the previous step's gradient.
    void zero_grad();

    // Number of parameter slots managed by this optimizer.
    //
    // Returns
    // -------
    // std::size_t
    //     ``params_.size()`` — includes null entries.
    std::size_t num_params() const { return params_.size(); }

    // Set the learning rate (called by LR schedulers between steps).
    //
    // Parameters
    // ----------
    // lr : double
    //     New learning rate.  Must be non-negative.
    virtual void set_lr(double lr) = 0;

    // Current learning rate.
    //
    // Returns
    // -------
    // double
    //     The value last passed to ``set_lr`` (or the constructor).
    virtual double lr() const = 0;

    // Short version-tagged identifier baked into checkpoints.
    //
    // Returns
    // -------
    // std::string
    //     A stable short tag like ``"sgd_v1"`` or ``"adam_v1"`` used by
    //     the Python loader to detect mismatched optimizer versions in
    //     ``load_state_dict``.
    virtual std::string state_dict_id() const = 0;

    // Named per-parameter state buffer list (one entry per kind of state).
    using NamedBuffers = std::pair<std::string, std::vector<std::shared_ptr<TensorImpl>>>;

    // Snapshot the per-parameter mutable state for checkpointing.
    //
    // Returns
    // -------
    // std::vector<NamedBuffers>
    //     A list of ``(name, tensors)`` pairs.  ``name`` identifies the
    //     buffer kind (e.g. ``"momentum_buffer"``); ``tensors`` runs
    //     parallel to ``params_`` with one entry per parameter slot
    //     (or a null entry if no state was allocated for that slot).
    //     Entries are *clones* — callers may keep them after the
    //     optimizer is destroyed.  Optimizers with no Python-visible
    //     state return an empty vector.
    //
    // Notes
    // -----
    // Implementations typically call ``clone_state_storage`` for each
    // active slot.  Slots whose state was never allocated (e.g. SGD
    // with ``momentum == 0``) should contribute a null pointer rather
    // than a synthetic zero buffer.
    //
    // See Also
    // --------
    // load_state_buffers : the inverse operation.
    virtual std::vector<NamedBuffers> state_buffers() const { return {}; }

    // Restore the per-parameter state in place from a snapshot.
    //
    // Parameters
    // ----------
    // bufs : const std::vector<NamedBuffers>&
    //     Must match the layout returned by ``state_buffers`` for this
    //     optimizer: same buffer names, same length, same per-slot
    //     shapes / dtypes.  Mismatches raise.
    //
    // Notes
    // -----
    // After this returns, every state slot is treated as initialised so
    // ``step()`` will not re-zero it on the next call.  Implementations
    // typically delegate the byte copy to ``overwrite_state_storage``.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If the buffer count, names, shapes, dtypes, or devices do not
    //     match the live optimizer state.
    //
    // See Also
    // --------
    // state_buffers : the inverse operation.
    virtual void load_state_buffers(const std::vector<NamedBuffers>& bufs) { (void)bufs; }

    // Global step counter used by some optimizers for bias correction.
    //
    // Returns
    // -------
    // std::int64_t
    //     Current step count (Adam bias correction, NAdam momentum
    //     schedule, ...).  Optimizers without a global counter return 0.
    virtual std::int64_t step_count() const { return 0; }

    // Override the global step counter (used by ``load_state_dict``).
    //
    // Parameters
    // ----------
    // count : std::int64_t
    //     New step count.  Optimizers without a global counter ignore
    //     this call.
    virtual void set_step_count(std::int64_t) {}

protected:
    // Apply the optimizer's update rule for a single parameter.
    //
    // Parameters
    // ----------
    // slot_idx : std::size_t
    //     Stable index of the parameter in ``params_``.  Used to look
    //     up the pre-allocated state buffers (momentum, second moment,
    //     parameter average, ...).
    // param : std::shared_ptr<TensorImpl>&
    //     Parameter tensor to update in place.  Guaranteed non-null and
    //     guaranteed to have an accumulated gradient.
    // grad : const Storage&
    //     The parameter's gradient storage for the current step.
    //
    // Notes
    // -----
    // Implementations must mutate ``param``'s storage in place — the
    // base ``step()`` is responsible for bumping the version counter
    // after this returns.  Both CPU and GPU paths are typically
    // dispatched from inside this method based on ``param->device()``.
    virtual void
    update_one(std::size_t slot_idx, std::shared_ptr<TensorImpl>& param, const Storage& grad) = 0;

    // Allocate and zero-initialise per-parameter state for one slot.
    //
    // Parameters
    // ----------
    // slot_idx : std::size_t
    //     Position in ``params_`` whose state should be materialised.
    // param : const std::shared_ptr<TensorImpl>&
    //     The parameter occupying that slot; provides the shape, dtype,
    //     and device for the state buffer(s).
    //
    // Notes
    // -----
    // Called exactly once per slot — on the first ``step()`` for which
    // that slot has a gradient.  Implementations should size their
    // ``std::vector<Storage>`` state members lazily here rather than in
    // the constructor so that disabled features (e.g. ``momentum == 0``
    // in SGD) skip the allocation entirely.
    virtual void init_state_slot(std::size_t slot_idx,
                                 const std::shared_ptr<TensorImpl>& param) = 0;

    // Flat list of trainable parameters; indexed by ``slot_idx``.
    std::vector<std::shared_ptr<TensorImpl>> params_;

    // Per-slot flag; true once ``init_state_slot`` has run for that slot.
    // Grown lazily by ``step()`` to match ``params_.size()``.
    std::vector<bool> state_initialized_;
};

}  // namespace lucid
