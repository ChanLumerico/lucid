// lucid/_C/optim/Ada.h
//
// Adaptive-gradient optimisers — Adamax, Adagrad, and Adadelta.
//
// These three optimisers all adapt a *per-parameter* effective step
// size from historical gradient information, but differ in **how** the
// history is summarised:
//
// - :class:`Adamax` keeps an EMA of $g$ and an $\ell_\infty$ envelope of $|g|$
//   (Adam with the infinity norm in place of the L2 norm).
// - :class:`Adagrad` accumulates the *running sum* of $g^2$ without forgetting,
//   producing a monotonically shrinking learning rate.
// - :class:`Adadelta` keeps an EMA of $g^2$ **and** an EMA of squared updates,
//   removing the explicit learning rate altogether.
//
// References
// ----------
// Kingma & Ba, "Adam: A Method for Stochastic Optimization" (ICLR 2015).
// Duchi, Hazan & Singer, "Adaptive Subgradient Methods for Online Learning
//   and Stochastic Optimization" (JMLR 2011).
// Zeiler, "ADADELTA: An Adaptive Learning Rate Method" (arXiv 1212.5701, 2012).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace lucid {

class TensorImpl;

// Adamax optimiser — Adam variant using the infinity norm.
//
// Replaces Adam's $\ell_2$ second-moment estimate with an exponentially
// decayed running maximum, giving a bounded effective step size that is
// well-behaved when occasional gradient components are very large.
//
// Math
// ----
// $$
//   m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
// $$
// $$
//   u_{t+1} = \max(\beta_2 u_t,\; |g_t|)
// $$
// $$
//   \theta_{t+1} = \theta_t - \frac{\eta}{1 - \beta_1^{t+1}} \cdot
//     \frac{m_{t+1}}{u_{t+1} + \epsilon}
// $$
// The second-moment recursion is the limit of an Adam-style $L^p$ EMA
// as $p \to \infty$, hence "max".
//
// Parameters
// ----------
// params : vector of TensorImpl
//     Parameters to optimise; one update buffer per entry.
// lr : float, default 2e-3
//     Base step size $\eta$.
// beta1 : float, default 0.9
//     First-moment EMA decay.
// beta2 : float, default 0.999
//     Decay used inside the running-max envelope.
// eps : float, default 1e-8
//     Numerical stabiliser added to $u$.
// weight_decay : float, default 0.0
//     L2 penalty applied to $g_t$ before the update.
//
// Attributes
// ----------
// m_ : vector of Storage
//     Per-parameter first-moment EMA.
// u_ : vector of Storage
//     Per-parameter $\ell_\infty$ envelope (the "infinity-norm" buffer).
// step_count_ : int
//     Global step counter, used for bias correction of $m$.
//
// Notes
// -----
// Unlike Adam there is **no** bias correction on $u$ — the running max
// is naturally bounded, so dividing it by $1 - \beta_2^{t+1}$ would
// over-amplify the very first updates.
//
// References
// ----------
// Kingma & Ba, "Adam: A Method for Stochastic Optimization", §7.1
// "Adamax" (ICLR 2015).
//
// See Also
// --------
// :class:`Adagrad`, :class:`Adadelta`, :class:`RMSprop`
class LUCID_API Adamax : public Optimizer {
public:
    // Construct the optimiser and resize per-slot state vectors.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl
    //     Parameters to optimise.
    // lr : float, default 2e-3
    //     Base step size.
    // beta1 : float, default 0.9
    //     First-moment EMA decay.
    // beta2 : float, default 0.999
    //     Infinity-norm envelope decay.
    // eps : float, default 1e-8
    //     Numerical stabiliser.
    // weight_decay : float, default 0.0
    //     L2 penalty coefficient.
    Adamax(std::vector<std::shared_ptr<TensorImpl>> params,
           double lr = 2e-3,
           double beta1 = 0.9,
           double beta2 = 0.999,
           double eps = 1e-8,
           double weight_decay = 0.0);

    // Set the active learning rate.
    //
    // Parameters
    // ----------
    // lr : float
    //     New base step size $\eta$ used on subsequent steps.
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
    //     ``"adamax_v1"`` — used by serialisation to validate state-dict layout.
    std::string state_dict_id() const override { return "adamax_v1"; }

protected:
    // Apply one Adamax update to parameter ``i``.
    //
    // Parameters
    // ----------
    // i : int
    //     Index of the parameter in the optimiser's parameter list.
    // p : TensorImpl
    //     Parameter tensor (updated in place).
    // g : Storage
    //     Gradient for ``p`` on this step.
    //
    // Notes
    // -----
    // Advances $m$ via the EMA, refreshes the $\ell_\infty$ envelope $u$,
    // then applies the bias-corrected step in one fused pass.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate the per-slot state buffers for parameter ``i``.
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
    // Both ``m_[i]`` (first moment) and ``u_[i]`` (infinity norm) are
    // zero-initialised.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_;  // Per-parameter first-moment estimates.
    std::vector<Storage> u_;  // Per-parameter infinity-norm estimates.
    std::int64_t step_count_;
};

// Adagrad optimiser — accumulates squared gradients without forgetting.
//
// Maintains a running sum of $g_t^2$ per parameter and scales the
// learning rate by its inverse square root.  The accumulator grows
// monotonically so the effective learning rate decays to zero in the
// limit — well suited to sparse features (NLP, embeddings) but
// typically too aggressive for dense deep-network weights.
//
// Math
// ----
// $$
//   G_{t+1} = G_t + g_t^2
// $$
// $$
//   \theta_{t+1} = \theta_t - \eta \cdot \frac{g_t}{\sqrt{G_{t+1}} + \epsilon}
// $$
//
// Parameters
// ----------
// params : vector of TensorImpl
//     Parameters to optimise.
// lr : float, default 1e-2
//     Base step size $\eta$.
// eps : float, default 1e-10
//     Numerical stabiliser inside the square root.
// weight_decay : float, default 0.0
//     L2 penalty coefficient.
// initial_accumulator_value : float, default 0.0
//     Constant used to seed ``sum_sq_grad_``.  A non-zero value avoids
//     dividing by a near-zero accumulator on the very first step.
//
// Attributes
// ----------
// sum_sq_grad_ : vector of Storage
//     Per-parameter cumulative sum $\sum_{s \le t} g_s^2$.
//
// Notes
// -----
// Because the accumulator never decays, the effective per-parameter
// learning rate $\eta / \sqrt{G_t + \epsilon}$ is **non-increasing** —
// once a coordinate has seen large gradients its step shrinks for the
// rest of training.  This is precisely what makes Adagrad strong on
// sparse problems and weak on long dense-network runs.
//
// References
// ----------
// Duchi, Hazan & Singer, "Adaptive Subgradient Methods for Online
// Learning and Stochastic Optimization" (JMLR 12, 2011).
//
// See Also
// --------
// :class:`Adadelta`, :class:`RMSprop` — both add a forgetting factor
// to address Adagrad's monotonic learning-rate decay.
class LUCID_API Adagrad : public Optimizer {
public:
    // Construct the optimiser and resize per-slot state vectors.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl
    //     Parameters to optimise.
    // lr : float, default 1e-2
    //     Base step size $\eta$.
    // eps : float, default 1e-10
    //     Numerical stabiliser.
    // weight_decay : float, default 0.0
    //     L2 penalty coefficient.
    // initial_accumulator_value : float, default 0.0
    //     Seed value for ``sum_sq_grad_``.
    Adagrad(std::vector<std::shared_ptr<TensorImpl>> params,
            double lr = 1e-2,
            double eps = 1e-10,
            double weight_decay = 0.0,
            double initial_accumulator_value = 0.0);

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
    //     ``"adagrad_v1"``.
    std::string state_dict_id() const override { return "adagrad_v1"; }

protected:
    // Apply one Adagrad update to parameter ``i``.
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
    // Accumulates $g^2$ into ``sum_sq_grad_[i]`` then applies
    // $p \mathrel{-}= \eta \cdot g / (\sqrt{G} + \epsilon)$.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate the squared-gradient accumulator for parameter ``i``.
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
    // When ``initial_accumulator_value_ != 0`` the buffer is filled
    // with that scalar rather than zeros.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, eps_, weight_decay_, initial_accumulator_value_;
    // Per-parameter cumulative sum of squared gradients.
    std::vector<Storage> sum_sq_grad_;
};

// Adadelta optimiser — Adagrad without a fixed learning rate.
//
// Replaces Adagrad's monotonically growing accumulator with an EMA of
// squared gradients $E[g^2]$ and additionally maintains an EMA of
// *squared parameter updates* $E[\Delta\theta^2]$.  The ratio of the
// two acts as a self-tuning step size, so ``lr`` only serves as a
// global scale and is conventionally left at 1.0.
//
// Math
// ----
// $$
//   E[g^2]_t = \rho\,E[g^2]_{t-1} + (1 - \rho)\,g_t^2
// $$
// $$
//   \Delta\theta_t = -\,\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}
//                              {\sqrt{E[g^2]_t + \epsilon}}\; g_t
// $$
// $$
//   E[\Delta\theta^2]_t = \rho\,E[\Delta\theta^2]_{t-1}
//     + (1 - \rho)\,\Delta\theta_t^2
// $$
// $$
//   \theta_{t+1} = \theta_t + \eta\,\Delta\theta_t
// $$
//
// Parameters
// ----------
// params : vector of TensorImpl
//     Parameters to optimise.
// lr : float, default 1.0
//     Global scale on $\Delta\theta_t$.  Typically kept at 1.0.
// rho : float, default 0.9
//     EMA decay $\rho$ shared by both running averages.
// eps : float, default 1e-6
//     Numerical stabiliser inside both square roots.
// weight_decay : float, default 0.0
//     L2 penalty coefficient.
//
// Attributes
// ----------
// sq_avg_ : vector of Storage
//     Per-parameter EMA of $g^2$.
// accumulated_update_ : vector of Storage
//     Per-parameter EMA of $(\Delta\theta)^2$.
//
// Notes
// -----
// Adadelta's unit-matching argument: the ratio
// $\sqrt{E[\Delta\theta^2]} / \sqrt{E[g^2]}$ has the same units as
// $\theta$ (rather than $\theta / g$), which is the original
// motivation for the method.
//
// References
// ----------
// Zeiler, "ADADELTA: An Adaptive Learning Rate Method"
// (arXiv:1212.5701, 2012).
//
// See Also
// --------
// :class:`Adagrad`, :class:`RMSprop`
class LUCID_API Adadelta : public Optimizer {
public:
    // Construct the optimiser and resize per-slot state vectors.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl
    //     Parameters to optimise.
    // lr : float, default 1.0
    //     Global scale on the unit-matched update.
    // rho : float, default 0.9
    //     EMA decay shared by both running averages.
    // eps : float, default 1e-6
    //     Numerical stabiliser.
    // weight_decay : float, default 0.0
    //     L2 penalty coefficient.
    Adadelta(std::vector<std::shared_ptr<TensorImpl>> params,
             double lr = 1.0,
             double rho = 0.9,
             double eps = 1e-6,
             double weight_decay = 0.0);

    // Set the active learning rate (global scale).
    //
    // Parameters
    // ----------
    // lr : float
    //     New scale factor applied to the unit-matched update.
    void set_lr(double lr) override { lr_ = lr; }

    // Return the current learning-rate scale.
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
    //     ``"adadelta_v1"``.
    std::string state_dict_id() const override { return "adadelta_v1"; }

protected:
    // Apply one Adadelta update to parameter ``i``.
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
    // Both running averages are advanced and the unit-matched step is
    // applied in a single fused pass.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate the two per-slot EMA buffers for parameter ``i``.
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
    // Both ``sq_avg_[i]`` and ``accumulated_update_[i]`` are
    // zero-initialised.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, rho_, eps_, weight_decay_;
    // Per-parameter running average of squared gradients.
    std::vector<Storage> sq_avg_;
    // Per-parameter running average of squared parameter updates.
    std::vector<Storage> accumulated_update_;
};

}  // namespace lucid
