// lucid/_C/optim/Prop.h
//
// Propagation-based adaptive optimisers — RMSprop and Rprop.
//
// :class:`RMSprop` normalises the gradient by an EMA of its
// squared magnitude (the "running mean square").  :class:`Rprop`
// discards gradient magnitudes entirely, adapting a per-parameter
// step from the *sign agreement* between consecutive gradients.
//
// References
// ----------
// Tieleman & Hinton, "Lecture 6.5 — RMSProp" (Coursera 2012, unpublished).
// Riedmiller & Braun, "A Direct Adaptive Method for Faster
//   Backpropagation Learning: The RPROP Algorithm" (ICNN 1993).

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// RMSprop optimiser — gradient normalisation by EMA of squared
// magnitude, with optional centring and Polyak-style momentum.
//
// The plain rule replaces Adagrad's monotonic accumulator with an
// exponential moving average, so the effective learning rate stops
// shrinking and tracks the *current* gradient regime instead.
//
// Math
// ----
// Uncentred:
// $$
//   v_{t+1} = \alpha\,v_t + (1 - \alpha)\,g_t^2
// $$
// $$
//   \theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\sqrt{v_{t+1}} + \epsilon}
// $$
//
// Centred — subtract the squared mean to recover the true variance:
// $$
//   \bar g_{t+1} = \alpha\,\bar g_t + (1 - \alpha)\,g_t
// $$
// $$
//   \tilde v_{t+1} = v_{t+1} - \bar g_{t+1}^2
// $$
// $$
//   \theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\sqrt{\tilde v_{t+1}} + \epsilon}
// $$
//
// With momentum $\mu \ne 0$ the normalised gradient is folded into a
// velocity buffer first:
// $$
//   b_{t+1} = \mu\,b_t + \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}, \qquad
//   \theta_{t+1} = \theta_t - \eta\,b_{t+1}
// $$
//
// Parameters
// ----------
// params : vector of TensorImpl
//     Parameters to optimise.
// lr : float, default 1e-2
//     Base step size $\eta$.
// alpha : float, default 0.99
//     EMA decay of the squared-gradient running average.
// eps : float, default 1e-8
//     Numerical stabiliser inside the square root.
// weight_decay : float, default 0.0
//     L2 penalty coefficient.
// momentum : float, default 0.0
//     Polyak momentum $\mu$.  When zero the momentum buffer is
//     skipped entirely.
// centered : bool, default false
//     Track $\bar g$ to recover the true variance.  Better-conditioned
//     in non-stationary regimes at the cost of one extra buffer.
//
// Attributes
// ----------
// square_avg_ : vector of Storage
//     EMA of $g^2$ (always allocated).
// grad_avg_ : vector of Storage
//     EMA of $g$ (allocated only when ``centered == true``).
// moment_buf_ : vector of Storage
//     Momentum buffer (allocated only when ``momentum != 0``).
//
// Notes
// -----
// State buffers are allocated **lazily** — code that only ever uses
// the plain variant does not pay the memory cost of the centred or
// momentum-augmented variants.
//
// References
// ----------
// Tieleman & Hinton, "Lecture 6.5 — RMSProp" (Coursera 2012).
//
// See Also
// --------
// :class:`Adadelta`, :class:`Adagrad`, :class:`Rprop`
class LUCID_API RMSprop : public Optimizer {
public:
    // Construct the optimiser and resize per-slot state vectors.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl
    //     Parameters to optimise.
    // lr : float, default 1e-2
    //     Base step size.
    // alpha : float, default 0.99
    //     EMA decay of the squared-gradient running average.
    // eps : float, default 1e-8
    //     Numerical stabiliser.
    // weight_decay : float, default 0.0
    //     L2 penalty coefficient.
    // momentum : float, default 0.0
    //     Polyak momentum.
    // centered : bool, default false
    //     Use the centred (true-variance) variant.
    RMSprop(std::vector<std::shared_ptr<TensorImpl>> params,
            double lr = 1e-2,
            double alpha = 0.99,
            double eps = 1e-8,
            double weight_decay = 0.0,
            double momentum = 0.0,
            bool centered = false);

    // Set the active learning rate.
    //
    // Parameters
    // ----------
    // lr : float
    //     New base step size.
    void set_lr(double lr) override { lr_ = lr; }

    // Return the current learning rate.
    //
    // Returns
    // -------
    // float
    //     Active value of $\eta$.
    double lr() const override { return lr_; }

    // Return the versioned state-dict identifier.
    //
    // Returns
    // -------
    // str
    //     ``"rmsprop_v1"``.
    std::string state_dict_id() const override { return "rmsprop_v1"; }

protected:
    // Apply one RMSprop update to parameter ``i``.
    //
    // Parameters
    // ----------
    // i : int
    //     Parameter slot index.
    // p : TensorImpl
    //     Parameter tensor (updated in place).
    // g : Storage
    //     Gradient for ``p`` on this step.
    //
    // Notes
    // -----
    // Dispatches to a fused MLX kernel on GPU and a scalar Accelerate
    // loop on CPU; the four variant combinations (plain / centred /
    // momentum / both) share a single branchy path that elides the
    // unused buffer reads.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate per-slot state buffers for parameter ``i``.
    //
    // Parameters
    // ----------
    // i : int
    //     Slot index.
    // p : TensorImpl
    //     Parameter whose shape/dtype/device dictates the buffer layout.
    //
    // Notes
    // -----
    // ``square_avg_[i]`` is always allocated.  ``grad_avg_[i]`` is
    // allocated only when ``centered_`` is true and ``moment_buf_[i]``
    // only when ``momentum_`` is non-zero — see Attributes on
    // :class:`RMSprop`.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, alpha_, eps_, weight_decay_, momentum_;
    bool centered_;
    // Per-parameter running mean-square of gradients.
    std::vector<Storage> square_avg_;
    // Per-parameter running mean of gradients (centered mode only).
    std::vector<Storage> grad_avg_;
    // Per-parameter momentum accumulation buffer.
    std::vector<Storage> moment_buf_;
};

// Rprop optimiser — Resilient Backpropagation (Riedmiller & Braun 1993).
//
// A magnitude-free quasi-Newton method: each parameter carries its own
// step size which grows when the gradient sign is consistent and
// shrinks when it flips.  Because only $\mathrm{sign}(g)$ enters the
// update, Rprop is immune to gradient-magnitude pathologies (vanishing
// or exploding) but is **incompatible with mini-batch noise** — it is
// a batch-mode method.
//
// Math
// ----
// Let $\Delta_t$ be the per-parameter step size and $g_t$ the gradient.
// $$
//   \Delta_{t+1} = \begin{cases}
//     \min(\eta_+\,\Delta_t,\;\Delta_{\max})   & g_t\,g_{t-1} > 0 \\
//     \max(\eta_-\,\Delta_t,\;\Delta_{\min})   & g_t\,g_{t-1} < 0 \\
//     \Delta_t                                & g_t\,g_{t-1} = 0
//   \end{cases}
// $$
// When the sign flips, $g_t$ is additionally **zeroed** to suppress a
// secondary overshoot on the following step.  The parameter update is
// $$
//   \theta_{t+1} = \theta_t - \mathrm{sign}(g_t)\,\Delta_{t+1}
// $$
//
// Parameters
// ----------
// params : vector of TensorImpl
//     Parameters to optimise.
// lr : float, default 1e-2
//     Initial value placed into every element of ``step_size_``.
//     Despite the name this is not used as a global scale.
// eta_minus : float, default 0.5
//     Step-shrink factor $\eta_-$ applied on sign flips.
// eta_plus : float, default 1.2
//     Step-grow factor $\eta_+$ applied on consistent signs.
// step_min : float, default 1e-6
//     Lower clamp $\Delta_{\min}$.
// step_max : float, default 50.0
//     Upper clamp $\Delta_{\max}$.
//
// Attributes
// ----------
// prev_grad_ : vector of Storage
//     Last-step gradient, used to detect sign flips.
// step_size_ : vector of Storage
//     Per-element adaptive step $\Delta$.  Initialised to ``lr_`` on
//     every element rather than to 1.0, so the very first update has a
//     sensible scale.
//
// Notes
// -----
// Rprop assumes deterministic gradients (full-batch).  Applying it to
// mini-batches confuses sign-flip detection because batch noise causes
// spurious sign changes — accepted practice is to either use very
// large batches or switch to RMSprop.
//
// References
// ----------
// Riedmiller & Braun, "A Direct Adaptive Method for Faster
// Backpropagation Learning: The RPROP Algorithm" (ICNN 1993).
//
// See Also
// --------
// :class:`RMSprop` — Hinton's mini-batch-friendly replacement that
// still uses an EMA of $g^2$ but keeps the gradient magnitude in the
// update.
class LUCID_API Rprop : public Optimizer {
public:
    // Construct the optimiser and resize per-slot state vectors.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl
    //     Parameters to optimise.
    // lr : float, default 1e-2
    //     Initial value for every element of ``step_size_``.
    // eta_minus : float, default 0.5
    //     Step-shrink factor.
    // eta_plus : float, default 1.2
    //     Step-grow factor.
    // step_min : float, default 1e-6
    //     Lower step clamp.
    // step_max : float, default 50.0
    //     Upper step clamp.
    Rprop(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-2,
          double eta_minus = 0.5,
          double eta_plus = 1.2,
          double step_min = 1e-6,
          double step_max = 50.0);

    // Set the active learning-rate parameter.
    //
    // Parameters
    // ----------
    // lr : float
    //     New ``lr_`` value.  This is **not** retroactively applied to
    //     existing ``step_size_`` buffers — it only affects fresh
    //     allocations.
    void set_lr(double lr) override { lr_ = lr; }

    // Return the current ``lr_`` parameter.
    //
    // Returns
    // -------
    // float
    //     Latest value passed to :meth:`set_lr` (or the constructor).
    double lr() const override { return lr_; }

    // Return the versioned state-dict identifier.
    //
    // Returns
    // -------
    // str
    //     ``"rprop_v1"``.
    std::string state_dict_id() const override { return "rprop_v1"; }

protected:
    // Apply one Rprop update to parameter ``i``.
    //
    // Parameters
    // ----------
    // i : int
    //     Parameter slot index.
    // p : TensorImpl
    //     Parameter tensor (updated in place).
    // g : Storage
    //     Gradient for ``p`` on this step.  May be zeroed in place
    //     where the sign of ``g`` flipped relative to ``prev_grad_``.
    //
    // Notes
    // -----
    // Drives the per-element step-size adaptation rule above and then
    // applies $-\mathrm{sign}(g)\,\Delta$ to ``p``.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate ``prev_grad_`` and ``step_size_`` for parameter ``i``.
    //
    // Parameters
    // ----------
    // i : int
    //     Slot index.
    // p : TensorImpl
    //     Parameter whose shape/dtype/device dictates the buffer layout.
    //
    // Notes
    // -----
    // ``prev_grad_[i]`` is zero-initialised; ``step_size_[i]`` is
    // filled with ``lr_`` so the first step has a sensible magnitude.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eta_minus_, eta_plus_, step_min_, step_max_;
    // Previous gradient, stored to detect sign changes.
    std::vector<Storage> prev_grad_;
    // Per-element adaptive step sizes.
    std::vector<Storage> step_size_;
};

}  // namespace lucid
