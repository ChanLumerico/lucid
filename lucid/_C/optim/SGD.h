// lucid/_C/optim/SGD.h
//
// Stochastic Gradient Descent and Averaged SGD optimizers.
// Both classes derive from Optimizer and implement their update rules
// on CPU (scalar loop over raw buffers) and GPU (MLX array ops).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// Stochastic gradient descent with optional momentum, Nesterov
// acceleration, and L2 weight decay.
//
// Plain SGD takes the parameter $\theta$ down the negative gradient
// direction at fixed learning rate $\eta$:
// $$
//   \theta_{t+1} = \theta_t - \eta\, g_t.
// $$
//
// With momentum coefficient $\mu > 0$, a per-parameter velocity buffer
// $v$ accumulates a low-pass-filtered gradient (Polyak's heavy ball):
// $$
//   v_{t+1} = \mu\, v_t + (1 - \tau)\, g_t, \qquad
//   \theta_{t+1} = \theta_t - \eta\, v_{t+1},
// $$
// where the dampening coefficient $\tau$ scales the *current* gradient
// when accumulating into $v$.  ``dampening = 0`` recovers classical
// Polyak momentum.
//
// With Nesterov acceleration (Sutskever et al. 2013 reformulation),
// the lookahead form is used:
// $$
//   v_{t+1} = \mu\, v_t + g_t, \qquad
//   \theta_{t+1} = \theta_t - \eta\, (\mu\, v_{t+1} + g_t),
// $$
// which requires ``dampening = 0`` and ``momentum > 0``.
//
// L2 weight decay with coefficient $\lambda$ is applied to the gradient
// *before* the momentum update (coupled weight decay, as in the
// reference framework):
// $$
//   g_t \leftarrow g_t + \lambda\, \theta_t.
// $$
// This is the standard L2 regularisation form; AdamW-style decoupled
// weight decay is *not* used here — use ``AdamW`` for that variant.
//
// Math
// ----
// Full update (momentum branch, no Nesterov):
// $$
//   g_t \leftarrow g_t + \lambda\, \theta_t, \qquad
//   v_{t+1} = \mu\, v_t + (1 - \tau)\, g_t, \qquad
//   \theta_{t+1} = \theta_t - \eta\, v_{t+1}.
// $$
// Plain SGD is the $\mu = 0$ special case where the velocity buffer is
// never allocated.  A parameter's first momentum step starts the buffer
// at its gradient, undamped, as the reference framework does:
// $$
//   v_1 = g_1.
// $$
// The same holds when momentum is switched on part-way through training.
//
// Attributes
// ----------
// lr_ : double
//     Learning rate $\eta$.  Must be non-negative.  Updated by
//     schedulers via ``set_lr``.
// momentum_ : double
//     Momentum coefficient $\mu$.  Zero disables momentum and skips
//     velocity-buffer allocation.
// dampening_ : double
//     Dampening coefficient $\tau$.  Must be zero when ``nesterov_``
//     is true.
// weight_decay_ : double
//     L2 penalty coefficient $\lambda$.
// nesterov_ : bool
//     Enables the Nesterov lookahead form.  Requires ``momentum_ > 0``
//     and ``dampening_ == 0``.
// moment_ : std::vector<Storage>
//     Per-parameter velocity buffers.  Entry $i$ is allocated by
//     ``init_state_slot`` on the first step seen by slot $i$ only when
//     ``momentum_ != 0``; otherwise it remains empty.
//
// Notes
// -----
// CPU and GPU update paths are dispatched inside ``update_one`` based
// on the parameter's device.  The CPU path is a flat scalar loop over
// the raw byte buffer; the GPU path expresses the update as a few MLX
// array ops, which compose with the surrounding lazy graph.
//
// Examples
// --------
// Typical training-loop usage from Python:
//
// >>> opt = lucid.optim.SGD(model.parameters(), lr=0.01,
// ...                       momentum=0.9, weight_decay=1e-4)
// >>> opt.zero_grad()
// >>> loss.backward()
// >>> opt.step()
//
// References
// ----------
// Polyak, "Some methods of speeding up the convergence of iteration
// methods" (1964).
// Sutskever, Martens, Dahl, Hinton, "On the importance of initialization
// and momentum in deep learning" (ICML 2013).
//
// See Also
// --------
// ASGD : averaged SGD with running parameter mean.
// Adam, AdamW : adaptive moment optimisers.
class LUCID_API SGD : public Optimizer {
public:
    // Construct an SGD optimizer with optional momentum and Nesterov.
    //
    // Parameters
    // ----------
    // params : std::vector<std::shared_ptr<TensorImpl>>
    //     Parameters to optimise.  Forwarded to ``Optimizer``.
    // lr : double
    //     Learning rate $\eta$.  Must be non-negative.
    // momentum : double, optional
    //     Momentum coefficient $\mu$ (default ``0.0``).  Zero disables
    //     momentum.
    // dampening : double, optional
    //     Dampening coefficient $\tau$ on the current gradient when
    //     accumulating into the velocity buffer (default ``0.0``).
    //     Must be ``0`` when ``nesterov`` is true.
    // weight_decay : double, optional
    //     L2 regularisation coefficient $\lambda$ (default ``0.0``).
    // nesterov : bool, optional
    //     If true, use the Nesterov lookahead form (default false).
    //     Requires ``momentum > 0`` and ``dampening == 0``.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If ``nesterov`` is requested without positive momentum, or
    //     with a non-zero ``dampening``.
    SGD(std::vector<std::shared_ptr<TensorImpl>> params,
        double lr,
        double momentum = 0.0,
        double dampening = 0.0,
        double weight_decay = 0.0,
        bool nesterov = false);

    // Update the learning rate from a scheduler.
    //
    // Parameters
    // ----------
    // lr : double
    //     New learning rate.
    void set_lr(double lr) override { lr_ = lr; }

    // Check SGD's hyper-parameters against its rules.
    //
    // The single statement of the rules: the constructor and
    // ``set_hyperparams`` both run it, and the Python wrapper runs it on
    // every parameter group it is given.  They are the reference
    // framework's constructor checks.
    //
    // Raises
    // ------
    // InvalidArgument
    //     If ``lr``, ``momentum`` or ``weight_decay`` is negative, or
    //     ``nesterov`` is set without positive momentum and zero dampening.
    static void check_hyperparams(
        double lr, double momentum, double dampening, double weight_decay, bool nesterov);

    // Replace the hyper-parameters between steps.
    //
    // The whole set is checked by ``check_hyperparams`` before any of it is
    // applied, so a rejected call changes nothing.  Momentum switched on
    // mid-run starts each parameter's buffer at its next gradient, as the
    // first momentum step always does (see ``update_one``).
    void set_hyperparams(
        double lr, double momentum, double dampening, double weight_decay, bool nesterov);

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Current momentum coefficient $\mu$.
    double momentum() const { return momentum_; }

    // Current L2 weight-decay coefficient $\lambda$.
    double weight_decay() const { return weight_decay_; }

    // Checkpoint identifier (``"sgd_v1"``).
    std::string state_dict_id() const override { return "sgd_v1"; }

    // Snapshot the per-parameter velocity buffers for checkpointing.
    //
    // Returns
    // -------
    // std::vector<NamedBuffers>
    //     Single-entry list ``[("momentum_buffer", tensors)]`` whose
    //     ``tensors`` runs parallel to ``params_``, or an empty list when no
    //     slot holds a buffer.  A slot contributes a null pointer until its
    //     first momentum step.  A buffer stays after momentum is set to
    //     zero, as the reference framework's state keeps it.
    //
    // See Also
    // --------
    // load_state_buffers : the inverse operation.
    std::vector<NamedBuffers> state_buffers() const override;

    // Restore the velocity buffers from a checkpoint snapshot.
    //
    // Parameters
    // ----------
    // bufs : const std::vector<NamedBuffers>&
    //     The ``"momentum_buffer"`` entry, laid out as ``state_buffers``
    //     lays it out; other names are ignored.  It is restored whatever
    //     the current momentum, as the reference framework restores it.
    //
    // Raises
    // ------
    // std::runtime_error
    //     On any name / shape / dtype / device mismatch.
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;

protected:
    // Apply the SGD update for one parameter slot.
    //
    // Dispatches to either the CPU scalar loop or the MLX GPU path based
    // on the parameter's device.  Implements the plain-SGD, momentum,
    // Nesterov, and weight-decay branches according to the constructor
    // flags.  See the class-level math block for the precise update
    // rule.
    //
    // Parameters
    // ----------
    // slot_idx : std::size_t
    //     Index into ``params_`` and ``moment_``.
    // param : std::shared_ptr<TensorImpl>&
    //     Parameter to update in place.
    // grad : const Storage&
    //     Accumulated gradient for this step.
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Make room for one slot's velocity buffer without allocating it.
    //
    // Parameters
    // ----------
    // slot_idx : std::size_t
    //     Index into ``params_`` and ``moment_``.
    // param : const std::shared_ptr<TensorImpl>&
    //     Parameter occupying the slot (unused).
    //
    // Notes
    // -----
    // The buffer is made by the slot's first momentum step, from that
    // step's gradient, so plain SGD allocates no extra memory.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;
    // Per-parameter velocity buffers; entry i is held from slot i's first
    // momentum step on.
    std::vector<Storage> moment_;
};

// Averaged Stochastic Gradient Descent (Polyak-Ruppert averaging).
//
// Runs SGD with a decaying step size and a decay on the parameter
// itself, while also maintaining a running average $\bar\theta$ (``ax``)
// of the parameter values.  The algorithm is the reference framework's,
// state names and all: ``step``, ``eta``, ``mu`` and ``ax`` per
// parameter.
//
// Math
// ----
// With $t$ the slot's own step count (1 on its first update), $\eta_t$
// and $\mu_t$ the values carried over from the previous step
// ($\eta_1 = \eta$, $\mu_1 = 1$):
// $$
//   g_t \leftarrow g_t + w\, \theta_{t-1}
// $$
// $$
//   \theta_t = \theta_{t-1}\,(1 - \lambda\,\eta_t) - \eta_t\, g_t
// $$
// $$
//   \bar\theta_t = \begin{cases}
//       \theta_t & \mu_t = 1 \\
//       \bar\theta_{t-1} + \mu_t\,(\theta_t - \bar\theta_{t-1}) & \text{otherwise}
//   \end{cases}
// $$
// then, for the next step,
// $$
//   \eta_{t+1} = \frac{\eta}{(1 + \lambda\,\eta\,t)^{\alpha}}, \qquad
//   \mu_{t+1} = \frac{1}{\max(1,\; t - t_0)}.
// $$
// Before $t_0$ the average therefore tracks $\theta$ exactly; after it,
// $\bar\theta$ is the mean of the iterates since $t_0$.
//
// Attributes
// ----------
// lr_ : double
//     Learning rate $\eta$.  Read when ``eta`` is first set and each time
//     it is advanced, so a scheduler's change takes effect from the next
//     step's ``eta``.
// lambd_ : double
//     Decay term $\lambda$ — of the step size and of the parameter.
// alpha_ : double
//     Power of the step-size decay.
// t0_ : double
//     Step at which averaging starts.
// weight_decay_ : double
//     L2 penalty coefficient $w$.
// ax_ : std::vector<Storage>
//     Per-parameter running average $\bar\theta$, zero until the first
//     update copies the parameter in.
// eta_, mu_ : std::vector<double>
//     Per-parameter $\eta_t$ and $\mu_t$, held at the precision they are
//     checkpointed at (``round_to_state_scalar``) so a restored run
//     continues bit-identically.
//
// Notes
// -----
// $t$ is the slot's own update count (``Optimizer::steps_``), so a
// parameter introduced into training late (or temporarily frozen) starts
// its own schedule.
//
// References
// ----------
// Polyak and Juditsky, "Acceleration of stochastic approximation by
// averaging" (SIAM J. Control Optim., 1992).
//
// See Also
// --------
// SGD : the plain instantaneous update.
class LUCID_API ASGD : public Optimizer {
public:
    // Construct an ASGD optimizer.
    //
    // Parameters
    // ----------
    // params : std::vector<std::shared_ptr<TensorImpl>>
    //     Parameters to optimise.
    // lr : double, optional
    //     Learning rate $\eta$ (default ``1e-2``).  Must be non-negative.
    // lambd : double, optional
    //     Decay term $\lambda$ (default ``1e-4``).
    // alpha : double, optional
    //     Power of the step-size decay (default ``0.75``).
    // t0 : double, optional
    //     Step at which averaging starts (default ``1e6``).
    // weight_decay : double, optional
    //     L2 regularisation coefficient (default ``0.0``).
    ASGD(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-2,
         double lambd = 1e-4,
         double alpha = 0.75,
         double t0 = 1e6,
         double weight_decay = 0.0);

    // Update the learning rate from a scheduler.
    void set_lr(double lr) override { lr_ = lr; }

    // Check ASGD's hyper-parameters against its rules — the reference
    // framework's constructor checks, run by the constructor and by
    // ``set_hyperparams``.
    //
    // Raises
    // ------
    // InvalidArgument
    //     If ``lr`` or ``weight_decay`` is negative or NaN.
    static void check_hyperparams(double lr, double weight_decay);

    // Replace the hyper-parameters between steps; nothing is applied when
    // ``check_hyperparams`` rejects them.  ``lr`` reaches the update through
    // the next step's ``eta``, which the reference framework computes at the
    // end of each step.
    void set_hyperparams(double lr, double lambd, double alpha, double t0, double weight_decay);

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Checkpoint identifier (``"asgd_v1"``).
    std::string state_dict_id() const override { return "asgd_v1"; }

    // Snapshot the per-slot state for checkpointing.
    //
    // Returns
    // -------
    // std::vector<NamedBuffers>
    //     ``step`` (0-d I64 per slot), ``eta`` and ``mu`` (0-d, F32 — F64
    //     for an F64 parameter) and ``ax``.  Slots that have not stepped
    //     contribute null entries.
    std::vector<NamedBuffers> state_buffers() const override;

    // Restore the state captured by ``state_buffers``.
    //
    // Raises
    // ------
    // std::runtime_error
    //     On a shape / dtype / device mismatch with the live parameters.
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;

protected:
    // Apply the ASGD update for one parameter slot.
    //
    // Decays and steps $\theta$ with the slot's current $\eta_t$, folds
    // it into $\bar\theta$ with $\mu_t$, then advances $\eta$ and $\mu$
    // for the slot's next step.
    //
    // Parameters
    // ----------
    // i : std::size_t
    //     Slot index into ``params_``, ``ax_``, ``eta_``, ``mu_``.
    // p : std::shared_ptr<TensorImpl>&
    //     Parameter to update in place.
    // g : const Storage&
    //     Accumulated gradient for this step.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate per-slot state on the first observed gradient.
    //
    // ``ax_`` starts at zero, ``eta_`` at the current learning rate and
    // ``mu_`` at one.
    //
    // Parameters
    // ----------
    // i : std::size_t
    //     Slot index.
    // p : const std::shared_ptr<TensorImpl>&
    //     Parameter whose layout dictates the state buffer shapes.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, lambd_, alpha_, t0_, weight_decay_;
    // Per-parameter running averages of the parameter trajectory.
    std::vector<Storage> ax_;
    // Per-parameter step size and averaging weight for the next update.
    std::vector<double> eta_;
    std::vector<double> mu_;
};

}  // namespace lucid
