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
// never allocated.
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
    //     ``tensors`` runs parallel to ``params_``.  Slots without an
    //     allocated velocity (``momentum == 0`` or the slot has never
    //     received a gradient) contribute a null pointer.
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
    //     Must contain exactly one entry whose name is
    //     ``"momentum_buffer"`` and whose tensor list matches the
    //     layout of ``state_buffers``.
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

    // Allocate the velocity buffer for one slot when ``momentum != 0``.
    //
    // Parameters
    // ----------
    // slot_idx : std::size_t
    //     Index into ``params_`` and ``moment_``.
    // param : const std::shared_ptr<TensorImpl>&
    //     Parameter whose shape, dtype and device dictate the velocity
    //     buffer layout.
    //
    // Notes
    // -----
    // When ``momentum_ == 0`` the corresponding ``moment_`` entry is
    // left empty so that plain SGD allocates no extra memory.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;
    // Per-parameter velocity buffers; entry i is active only when momentum != 0.
    std::vector<Storage> moment_;
};

// Averaged Stochastic Gradient Descent (Polyak-Ruppert averaging).
//
// Runs a standard SGD-with-momentum trajectory while also maintaining a
// running average $\bar\theta$ of the parameter values.  Once the step
// counter passes the warm-up threshold ``t0_``, every step blends the
// current parameter into the running average using a decaying mixing
// coefficient $\eta_{\mathrm{avg}}$:
// $$
//   \eta_{\mathrm{avg}} = \frac{1}{\alpha\, t + 1}, \qquad
//   \bar\theta_{t+1} =
//       (1 - \eta_{\mathrm{avg}})\, \bar\theta_t +
//       \eta_{\mathrm{avg}}\, \theta_{t+1} - \lambda\, \bar\theta_t.
// $$
// Before ``t0_`` the average simply tracks $\theta$ ($\bar\theta = \theta$).
//
// The averaged weights $\bar\theta$ generally yield lower-variance
// estimators than the instantaneous parameters, especially for convex
// or near-convex problems, and are typically used at inference time
// after training has finished.
//
// Math
// ----
// Per-parameter update (post warm-up):
// $$
//   g_t \leftarrow g_t + \lambda\, \theta_t,
// $$
// $$
//   v_{t+1} = \mu\, v_t + g_t, \qquad
//   \theta_{t+1} = \theta_t - \eta\, v_{t+1},
// $$
// $$
//   \bar\theta_{t+1} =
//       (1 - \eta_{\mathrm{avg}})\, \bar\theta_t +
//       \eta_{\mathrm{avg}}\, \theta_{t+1} - \lambda\, \bar\theta_t.
// $$
//
// Attributes
// ----------
// lr_ : double
//     Learning rate $\eta$.
// momentum_ : double
//     Momentum coefficient $\mu$.  Zero disables velocity allocation.
// weight_decay_ : double
//     L2 penalty coefficient $\lambda$.
// alpha_ : double
//     Decay exponent controlling the averaging schedule
//     ($\eta_{\mathrm{avg}} = 1 / (\alpha t + 1)$).
// t0_ : double
//     Number of warm-up steps before averaging starts.  Stored as
//     ``double`` to mirror the reference framework's API but compared
//     against integer step counts.
// lambd_ : double
//     Average-side decay coefficient applied to $\bar\theta$ each
//     averaging step.
// moment_ : std::vector<Storage>
//     Per-parameter SGD velocity buffers (active when ``momentum_ != 0``).
// ax_ : std::vector<Storage>
//     Per-parameter running averages of the parameter trajectory.
//     Initialised to a copy of the parameter on the first observed
//     gradient.
// step_ : std::vector<std::int64_t>
//     Per-parameter step counters driving $\eta_{\mathrm{avg}}$.
//
// Notes
// -----
// The step counter is maintained per parameter slot rather than as a
// single global counter so that parameters introduced into training
// late (or temporarily frozen) still see a clean averaging schedule.
//
// References
// ----------
// Polyak and Juditsky, "Acceleration of stochastic approximation by
// averaging" (SIAM J. Control Optim., 1992).
//
// See Also
// --------
// SGD : the underlying instantaneous update.
class LUCID_API ASGD : public Optimizer {
public:
    // Construct an ASGD optimizer.
    //
    // Parameters
    // ----------
    // params : std::vector<std::shared_ptr<TensorImpl>>
    //     Parameters to optimise.
    // lr : double, optional
    //     Learning rate $\eta$ (default ``1e-3``).
    // momentum : double, optional
    //     SGD momentum coefficient $\mu$ (default ``0.0``).
    // weight_decay : double, optional
    //     L2 regularisation coefficient $\lambda$ (default ``0.0``).
    // alpha : double, optional
    //     Decay exponent for the averaging schedule (default ``0.75``).
    // t0 : double, optional
    //     Warm-up step count before averaging engages (default ``1e6``).
    // lambd : double, optional
    //     Average-side decay coefficient (default ``1e-4``).
    ASGD(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-3,
         double momentum = 0.0,
         double weight_decay = 0.0,
         double alpha = 0.75,
         double t0 = 1e6,
         double lambd = 1e-4);

    // Update the learning rate from a scheduler.
    void set_lr(double lr) override { lr_ = lr; }

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Checkpoint identifier (``"asgd_v1"``).
    std::string state_dict_id() const override { return "asgd_v1"; }

protected:
    // Apply the ASGD update for one parameter slot.
    //
    // Performs the standard SGD-with-momentum step on $\theta$, then —
    // once the per-slot step counter has passed ``t0_`` — updates the
    // running average $\bar\theta$ using a $1/(\alpha t + 1)$ schedule
    // attenuated by the average-side decay $\lambda$.
    //
    // Parameters
    // ----------
    // i : std::size_t
    //     Slot index into ``params_``, ``moment_``, ``ax_``, ``step_``.
    // p : std::shared_ptr<TensorImpl>&
    //     Parameter to update in place.
    // g : const Storage&
    //     Accumulated gradient for this step.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate per-slot state on the first observed gradient.
    //
    // Materialises the velocity buffer (only when ``momentum_ != 0``)
    // and the running-average buffer ``ax_`` initialised to a copy of
    // the current parameter.  The per-slot step counter is reset to 0.
    //
    // Parameters
    // ----------
    // i : std::size_t
    //     Slot index.
    // p : const std::shared_ptr<TensorImpl>&
    //     Parameter whose layout dictates the state buffer shapes.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, momentum_, weight_decay_, alpha_, t0_, lambd_;
    // Per-parameter SGD velocity buffers (active when momentum != 0).
    std::vector<Storage> moment_;
    // Per-parameter running averages of the parameter trajectory.
    std::vector<Storage> ax_;
    // Per-parameter step counters used to compute the averaging coefficient.
    std::vector<std::int64_t> step_;
};

}  // namespace lucid
