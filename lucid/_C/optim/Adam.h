// lucid/_C/optim/Adam.h
//
// Adam-family optimizers: Adam, AdamW, NAdam, and RAdam.  All four
// maintain first-moment ($m$) and second-moment ($v$) estimates for
// each parameter and share a common low-level kernel (``adam_step_cpu``
// / ``adam_step_gpu``) parameterised by flags and scalar broadcasts.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "../api.h"
#include "../core/Dtype.h"
#include "../core/Storage.h"
#include "Optimizer.h"

namespace mlx::core {
class array;
}  // namespace mlx::core

namespace lucid {

class TensorImpl;

// Per-step scalar broadcast cache for Adam-family GPU updates.
//
// Holds the eight broadcast scalars ($\beta_1$, $1-\beta_1$, $\beta_2$,
// $1-\beta_2$, $\epsilon$, $\eta$, $1/(1-\beta_1^t)$, $1/(1-\beta_2^t)$)
// plus the optional weight-decay scalar and the AdamW decay factor as
// pre-allocated MLX scalar arrays.  All entries are rebuilt **once per
// optimiser step** at the first parameter and reused across the
// remaining ~64 parameters of a typical ResNet-18 / transformer step.
//
// Notes
// -----
// Without this cache, ``adam_step_gpu`` rebuilt ``params.size() * 8``
// MLX scalar arrays per call (~520 allocations for ResNet-18).  Each
// MLX array construction has constant overhead, and the bias-correction
// factors don't depend on which parameter is being updated, so the
// scalars are pure step-level invariants.
//
// The cache invalidates when ``step_count_`` advances, the param dtype
// changes (mixed-precision case), or when ``set_lr`` mutates $\eta$.
//
// Attributes
// ----------
// valid : bool
//     ``true`` when the cached arrays match the current step / dtype.
// dt : Dtype
//     Element type the arrays were built for.
// for_step : int64
//     Step number the arrays were built at.
// b1, omb1, b2, omb2 : mlx array (scalar)
//     $\beta_1$, $1 - \beta_1$, $\beta_2$, $1 - \beta_2$ broadcasts.
// eps_a, lr_a, wd_a : mlx array (scalar)
//     $\epsilon$, $\eta$, $\lambda$ broadcasts.  ``wd_a`` is present
//     only when ``weight_decay != 0``.
// inv_bc1, inv_bc2 : mlx array (scalar)
//     $1 / (1 - \beta_1^t)$ and $1 / (1 - \beta_2^t)$.
// wd_factor : mlx array (scalar)
//     $(1 - \eta \lambda)$, the AdamW multiplicative decay factor.
struct AdamScalarCache {
    bool valid = false;
    Dtype dt = Dtype::F32;
    std::int64_t for_step = 0;
    std::unique_ptr<mlx::core::array> b1;
    std::unique_ptr<mlx::core::array> omb1;
    std::unique_ptr<mlx::core::array> b2;
    std::unique_ptr<mlx::core::array> omb2;
    std::unique_ptr<mlx::core::array> eps_a;
    std::unique_ptr<mlx::core::array> lr_a;
    std::unique_ptr<mlx::core::array> wd_a;       // present iff weight_decay != 0
    std::unique_ptr<mlx::core::array> inv_bc1;
    std::unique_ptr<mlx::core::array> inv_bc2;
    std::unique_ptr<mlx::core::array> wd_factor;  // (1 - lr*wd) for AdamW

    // Default-construct an empty (invalid) cache.
    AdamScalarCache();

    // Out-of-line destructor — ``mlx::core::array`` is only forward-
    // declared in this header, so ``unique_ptr`` cannot inline-destroy
    // it.  The definition lives in ``Adam.cpp``.
    ~AdamScalarCache();

    AdamScalarCache(const AdamScalarCache&) = delete;
    AdamScalarCache& operator=(const AdamScalarCache&) = delete;
};

// Adam optimiser — adaptive moment estimation (Kingma & Ba, 2014).
//
// Maintains exponential moving averages of the gradient and squared
// gradient for each parameter, then applies a bias-corrected step
// scaled element-wise by the inverse second-moment $\sqrt{\hat v}$.
//
// Weight decay (when non-zero) is added to the gradient before the
// moment update — i.e. the **L2-regularisation form**, NOT the
// decoupled AdamW form.  Use ``AdamW`` for decoupled decay.
//
// Math
// ----
// Per parameter, with global step counter $t = $ ``step_count_``:
//
// $$
//   g_t \leftarrow g_t + \lambda\, \theta_t \quad
//   \text{(if } \lambda > 0\text{)}
// $$
//
// $$
//   m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t, \qquad
//   v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2
// $$
//
// $$
//   \hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad
//   \hat v_t = \frac{v_t}{1 - \beta_2^t}
// $$
//
// $$
//   \theta_{t+1} = \theta_t - \eta \cdot
//       \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
// $$
//
// Notes
// -----
// ``step_count_`` is a global counter incremented once per ``step()``
// call regardless of how many parameters are updated, so all parameters
// share the same bias-correction factor within a step.  This matches
// reference-framework semantics.
//
// AMSGrad (Reddi et al., 2018) is declared via the ``amsgrad`` ctor
// flag but not yet implemented — passing ``true`` raises
// ``not_implemented`` at construction time.
//
// Attributes
// ----------
// lr_ : float
//     Step size $\eta$.  Default ``1e-3``.
// beta1_, beta2_ : float
//     EMA decay rates for the first and second moments.  Defaults
//     ``0.9`` and ``0.999`` from the original paper.
// eps_ : float
//     Denominator stabiliser $\epsilon$.  Default ``1e-8``.
// weight_decay_ : float
//     L2 penalty coefficient $\lambda$ added to the gradient.  Set to
//     ``0`` to disable.
// amsgrad_ : bool
//     Reserved — see Notes.  Currently must be ``false``.
// step_count_ : int64
//     Global step counter; bias-correction factors are functions of
//     this.  Persisted in ``state_dict`` under ``state_step``.
// m_, v_ : Storage[]
//     Per-parameter first-moment and second-moment buffers, matching
//     each parameter's shape and dtype.
// scalar_cache_ : AdamScalarCache
//     Per-step broadcast cache for the GPU kernel.
//
// References
// ----------
// Kingma & Ba, "Adam: A Method for Stochastic Optimization"
//   (ICLR 2015).
// Reddi, Kale & Kumar, "On the Convergence of Adam and Beyond"
//   (ICLR 2018) — AMSGrad variant.
//
// See Also
// --------
// AdamW : decoupled-weight-decay variant.
// NAdam : Nesterov-momentum variant.
// RAdam : variance-rectified variant.
class LUCID_API Adam : public Optimizer {
public:
    // Construct an Adam optimiser bound to ``params``.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl shared pointers
    //     Parameters to optimise.  One ``m`` and one ``v`` buffer is
    //     allocated per parameter on first ``step()`` call.
    // lr : float, optional
    //     Step size $\eta$.  Default ``1e-3``.
    // beta1, beta2 : float, optional
    //     First- and second-moment EMA decay rates.  Defaults ``0.9``
    //     and ``0.999``.
    // eps : float, optional
    //     Denominator stabiliser $\epsilon$.  Default ``1e-8``.
    // weight_decay : float, optional
    //     L2 penalty coefficient $\lambda$.  Default ``0``.
    // amsgrad : bool, optional
    //     Reserved.  Must be ``false`` — see class Notes.
    //
    // Raises
    // ------
    // not_implemented
    //     If ``amsgrad == true``.
    Adam(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0,
         bool amsgrad = false);

    // Set the learning rate and invalidate the scalar cache.
    //
    // Parameters
    // ----------
    // lr : float
    //     New step size $\eta$.
    //
    // Notes
    // -----
    // The 3.4 perf pass folds $\eta$ (and $1 - \eta \lambda$ for
    // AdamW) into ``AdamScalarCache``; mutating $\eta$ here flips
    // ``scalar_cache_.valid = false`` so the cached MLX scalars are
    // rebuilt on the next step.
    void set_lr(double lr) override {
        lr_ = lr;
        scalar_cache_.valid = false;
    }

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // First-moment EMA decay rate $\beta_1$.
    double beta1() const { return beta1_; }

    // Second-moment EMA decay rate $\beta_2$.
    double beta2() const { return beta2_; }

    // Denominator stabiliser $\epsilon$.
    double eps() const { return eps_; }

    // Schema identifier for ``state_dict`` serialisation.
    //
    // Returns
    // -------
    // str
    //     ``"adam_v1"``.
    std::string state_dict_id() const override { return "adam_v1"; }

    // Export per-parameter state (``exp_avg`` = $m$, ``exp_avg_sq`` = $v$).
    //
    // Returns
    // -------
    // vector of NamedBuffers
    //     One entry per parameter, each containing the named buffers
    //     ``exp_avg`` and ``exp_avg_sq``.
    std::vector<NamedBuffers> state_buffers() const override;

    // Restore per-parameter state from a previous ``state_buffers``.
    //
    // Parameters
    // ----------
    // bufs : vector of NamedBuffers
    //     Must have the same length as the parameter list and contain
    //     ``exp_avg`` / ``exp_avg_sq`` keys.
    //
    // Raises
    // ------
    // runtime_error
    //     On length / key / dtype / shape mismatch.
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;

    // Global step counter $t$ used for bias correction.
    std::int64_t step_count() const override { return step_count_; }

    // Set the global step counter (used by checkpoint loading).
    void set_step_count(std::int64_t s) override { step_count_ = s; }

protected:
    // Apply the standard Adam update to a single parameter.
    //
    // Parameters
    // ----------
    // slot_idx : size_t
    //     Index into ``m_`` / ``v_``.
    // param : TensorImpl shared pointer
    //     In-place parameter to update.
    // grad : Storage
    //     Gradient $g_t$ matching ``param`` shape / dtype / device.
    //
    // Notes
    // -----
    // Dispatches to ``adam_step_gpu`` (GPU stream) or ``adam_step_cpu``
    // (CPU stream) with ``decoupled_wd = false``.
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Allocate zero-initialised $m$ and $v$ buffers for ``slot_idx``.
    //
    // Parameters
    // ----------
    // slot_idx : size_t
    //     Index into ``m_`` / ``v_``.
    // param : TensorImpl shared pointer
    //     Reference parameter — buffers inherit its shape / dtype.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    bool amsgrad_;
    // Global step counter; bias correction factors are functions of this.
    std::int64_t step_count_;

    std::vector<Storage> m_;  // Per-parameter first-moment estimates.
    std::vector<Storage> v_;  // Per-parameter second-moment estimates.

    // 3.4 perf: see AdamScalarCache documentation above.
    AdamScalarCache scalar_cache_;
};

// AdamW optimiser — Adam with decoupled weight decay
// (Loshchilov & Hutter, 2019).
//
// Decouples the L2 penalty from the adaptive moment estimation: the
// weight-decay term is applied directly to the parameter, NOT folded
// into the gradient $g_t$.  This restores the rotation-invariance lost
// by Adam's adaptive learning rate when L2 is added to $g$, and is
// strongly preferred over Adam + L2 for transformer training.
//
// Math
// ----
// $$
//   m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t, \qquad
//   v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2
// $$
//
// $$
//   \hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad
//   \hat v_t = \frac{v_t}{1 - \beta_2^t}
// $$
//
// $$
//   \theta_{t+1} = (1 - \eta \lambda)\, \theta_t - \eta \cdot
//       \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
// $$
//
// Notes
// -----
// The implementation realises the decay as a multiplicative pre-scale
// of $\theta_t$ by $(1 - \eta \lambda)$ (cached in
// ``AdamScalarCache::wd_factor``), then performs the standard Adam
// step.  Equivalent to the additive form above but avoids a separate
// MLX op invocation per parameter.
//
// Attributes
// ----------
// lr_ : float
//     Step size $\eta$.  Default ``1e-3``.
// beta1_, beta2_ : float
//     EMA decay rates.  Defaults ``0.9`` / ``0.999``.
// eps_ : float
//     Denominator stabiliser.  Default ``1e-8``.
// weight_decay_ : float
//     Decoupled penalty coefficient $\lambda$.  Default ``1e-2`` —
//     larger than Adam's because it is no longer scaled by the
//     adaptive denominator.
// step_count_ : int64
//     Global step counter.
// m_, v_ : Storage[]
//     Per-parameter first- and second-moment buffers.
// scalar_cache_ : AdamScalarCache
//     Per-step broadcast cache, including ``wd_factor``.
//
// References
// ----------
// Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
//   (ICLR 2019).
// Kingma & Ba, "Adam: A Method for Stochastic Optimization"
//   (ICLR 2015).
//
// See Also
// --------
// Adam : L2-regularisation variant.
class LUCID_API AdamW : public Optimizer {
public:
    // Construct an AdamW optimiser bound to ``params``.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl shared pointers
    //     Parameters to optimise.
    // lr : float, optional
    //     Step size $\eta$.  Default ``1e-3``.
    // beta1, beta2 : float, optional
    //     First- and second-moment EMA decay rates.  Defaults ``0.9``
    //     and ``0.999``.
    // eps : float, optional
    //     Denominator stabiliser $\epsilon$.  Default ``1e-8``.
    // weight_decay : float, optional
    //     Decoupled penalty coefficient $\lambda$.  Default ``1e-2``.
    AdamW(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 1e-2);

    // Set the learning rate and invalidate the scalar cache.
    //
    // Notes
    // -----
    // Same rationale as ``Adam::set_lr`` — additionally invalidates
    // the cached AdamW ``wd_factor`` = $(1 - \eta \lambda)$ scalar.
    void set_lr(double lr) override {
        lr_ = lr;
        scalar_cache_.valid = false;
    }

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Schema identifier for ``state_dict`` serialisation.
    //
    // Returns
    // -------
    // str
    //     ``"adamw_v1"``.
    std::string state_dict_id() const override { return "adamw_v1"; }

    // Export per-parameter state (``exp_avg`` and ``exp_avg_sq``).
    std::vector<NamedBuffers> state_buffers() const override;

    // Restore per-parameter state.  See ``Adam::load_state_buffers``.
    void load_state_buffers(const std::vector<NamedBuffers>& bufs) override;

    // Global step counter $t$.
    std::int64_t step_count() const override { return step_count_; }

    // Set the global step counter.
    void set_step_count(std::int64_t s) override { step_count_ = s; }

protected:
    // Apply the decoupled-weight-decay Adam update.
    //
    // Notes
    // -----
    // Dispatches to ``adam_step_{cpu,gpu}`` with ``decoupled_wd = true``
    // so the kernel applies $\theta \leftarrow (1 - \eta \lambda)
    // \theta$ before the Adam step.
    void update_one(std::size_t slot_idx,
                    std::shared_ptr<TensorImpl>& param,
                    const Storage& grad) override;

    // Allocate zero-initialised $m$ and $v$ buffers for ``slot_idx``.
    void init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) override;

private:
    double lr_;
    double beta1_, beta2_, eps_;
    double weight_decay_;
    std::int64_t step_count_;

    std::vector<Storage> m_;
    std::vector<Storage> v_;

    // 3.4 perf: see AdamScalarCache documentation above.
    AdamScalarCache scalar_cache_;
};

// NAdam optimiser — Adam with Nesterov momentum (Dozat, 2016).
//
// Replaces the bias-corrected first-moment term with a Nesterov
// look-ahead that blends the current gradient $g_t$ and the running
// moment $m_t$.  The momentum coefficient $\mu_t$ decays each step,
// and its running product $\prod_{s \le t} \mu_s$ replaces the
// $1 - \beta_1^t$ bias correction.
//
// Math
// ----
// $$
//   \mu_t = \beta_1 \left( 1 - \tfrac{1}{2} \cdot 0.96^{t \psi} \right),
//   \qquad
//   \mu_{t+1} = \beta_1 \left( 1 - \tfrac{1}{2} \cdot 0.96^{(t+1)\psi} \right)
// $$
//
// where $\psi$ is ``momentum_decay``.  Let
// $\Pi_t = \prod_{s=1}^{t} \mu_s$ and $\Pi_{t+1} = \Pi_t \cdot \mu_{t+1}$.
//
// $$
//   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad
//   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
// $$
//
// $$
//   d_t = \sqrt{v_t / (1 - \beta_2^t)} + \epsilon
// $$
//
// $$
//   \theta_{t+1} = \theta_t
//     - \eta\, \frac{(1 - \mu_t)}{1 - \Pi_t} \cdot \frac{g_t}{d_t}
//     - \eta\, \frac{\mu_{t+1}}{1 - \Pi_{t+1}} \cdot \frac{m_t}{d_t}
// $$
//
// Notes
// -----
// ``mu_product_`` is a per-parameter ``double``, not a ``Storage``,
// because it is a single scalar accumulator rather than a tensor.
// Weight decay (if non-zero) is added to $g_t$ in the standard L2 form
// — there is no published "decoupled NAdam".
//
// Attributes
// ----------
// lr_ : float
//     Step size $\eta$.  Default ``2e-3`` (paper recommendation).
// beta1_, beta2_, eps_, weight_decay_ : float
//     As in ``Adam``.
// momentum_decay_ : float
//     Decay constant $\psi$ in the $\mu_t$ schedule.  Default
//     ``4e-3``.
// m_, v_ : Storage[]
//     Per-parameter first- and second-moment buffers.
// mu_product_ : double[]
//     Running product $\Pi_t$ of all momentum coefficients seen so far,
//     one scalar per parameter.
// step_count_ : int64
//     Global step counter.
//
// References
// ----------
// Dozat, "Incorporating Nesterov Momentum into Adam"
//   (ICLR Workshop, 2016).
//
// See Also
// --------
// Adam, AdamW, RAdam.
class LUCID_API NAdam : public Optimizer {
public:
    // Construct a NAdam optimiser bound to ``params``.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl shared pointers
    //     Parameters to optimise.
    // lr : float, optional
    //     Step size $\eta$.  Default ``2e-3``.
    // beta1, beta2 : float, optional
    //     EMA decay rates.  Defaults ``0.9`` / ``0.999``.
    // eps : float, optional
    //     Denominator stabiliser.  Default ``1e-8``.
    // weight_decay : float, optional
    //     L2 penalty coefficient.  Default ``0``.
    // momentum_decay : float, optional
    //     Momentum schedule decay $\psi$.  Default ``4e-3``.
    NAdam(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 2e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0,
          double momentum_decay = 0.004);

    // Set the learning rate.
    void set_lr(double lr) override { lr_ = lr; }

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Schema identifier for ``state_dict`` serialisation.
    //
    // Returns
    // -------
    // str
    //     ``"nadam_v1"``.
    std::string state_dict_id() const override { return "nadam_v1"; }

protected:
    // Compute per-step $\mu_t$ and $\mu_{t+1}$, advance ``mu_product_``,
    // then apply the two-term Nesterov update.
    //
    // Parameters
    // ----------
    // i : size_t
    //     Slot index into ``m_`` / ``v_`` / ``mu_product_``.
    // p : TensorImpl shared pointer
    //     In-place parameter.
    // g : Storage
    //     Gradient $g_t$.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialised $m$ and $v$ buffers; initialise
    // ``mu_product_[i]`` to ``1.0``.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_, momentum_decay_;
    std::vector<Storage> m_, v_;
    // Running product of all per-step momentum coefficients mu_t.
    std::vector<double> mu_product_;
    std::int64_t step_count_;
};

// RAdam optimiser — Rectified Adam (Liu et al., 2019).
//
// Computes a variance-based rectification factor $r_t$ derived from
// the simple moving average length $\rho_t$.  When $\rho_t$ exceeds a
// threshold (the paper uses $5$) the adaptive learning rate is
// well-conditioned and we apply the standard bias-corrected Adam step
// scaled by $r_t$; otherwise we fall back to a plain momentum step
// using only $\hat m_t$, which avoids the early-training divergence
// caused by an under-sampled second moment.
//
// Math
// ----
// $$
//   \rho_\infty = \frac{2}{1 - \beta_2} - 1, \qquad
//   \rho_t = \rho_\infty - \frac{2 t\, \beta_2^t}{1 - \beta_2^t}
// $$
//
// If $\rho_t > 5$:
//
// $$
//   r_t = \sqrt{
//     \frac{(\rho_t - 4)(\rho_t - 2)\,\rho_\infty}
//          {(\rho_\infty - 4)(\rho_\infty - 2)\,\rho_t}
//   }
// $$
//
// $$
//   \theta_{t+1} = \theta_t - \eta\, r_t \cdot
//       \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
// $$
//
// otherwise (fallback to plain momentum):
//
// $$
//   \theta_{t+1} = \theta_t - \eta\, \hat m_t
// $$
//
// Notes
// -----
// $\rho_t$ and $r_t$ depend only on $t$ and $\beta_2$, so they are
// computed once per step outside the per-parameter loop.  Weight
// decay (if non-zero) is added to $g_t$ in the standard L2 form.
//
// Attributes
// ----------
// lr_ : float
//     Step size $\eta$.  Default ``1e-3``.
// beta1_, beta2_, eps_, weight_decay_ : float
//     As in ``Adam``.
// m_, v_ : Storage[]
//     Per-parameter first- and second-moment buffers.
// step_count_ : int64
//     Global step counter.
//
// References
// ----------
// Liu, Jiang, He, Chen, Liu, Gao & Han, "On the Variance of the
//   Adaptive Learning Rate and Beyond" (ICLR 2020, arXiv 2019).
//
// See Also
// --------
// Adam, AdamW, NAdam.
class LUCID_API RAdam : public Optimizer {
public:
    // Construct an RAdam optimiser bound to ``params``.
    //
    // Parameters
    // ----------
    // params : vector of TensorImpl shared pointers
    //     Parameters to optimise.
    // lr : float, optional
    //     Step size $\eta$.  Default ``1e-3``.
    // beta1, beta2 : float, optional
    //     EMA decay rates.  Defaults ``0.9`` / ``0.999``.
    // eps : float, optional
    //     Denominator stabiliser.  Default ``1e-8``.
    // weight_decay : float, optional
    //     L2 penalty coefficient.  Default ``0``.
    RAdam(std::vector<std::shared_ptr<TensorImpl>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0);

    // Set the learning rate.
    void set_lr(double lr) override { lr_ = lr; }

    // Current learning rate $\eta$.
    double lr() const override { return lr_; }

    // Schema identifier for ``state_dict`` serialisation.
    //
    // Returns
    // -------
    // str
    //     ``"radam_v1"``.
    std::string state_dict_id() const override { return "radam_v1"; }

protected:
    // Compute $\rho_t$ and $r_t$; apply the rectified adaptive update
    // when $\rho_t > 5$, else fall back to a plain momentum update.
    //
    // Parameters
    // ----------
    // i : size_t
    //     Slot index into ``m_`` / ``v_``.
    // p : TensorImpl shared pointer
    //     In-place parameter.
    // g : Storage
    //     Gradient $g_t$.
    void update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& g) override;

    // Allocate zero-initialised $m$ and $v$ buffers for this slot.
    void init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) override;

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Storage> m_, v_;
    std::int64_t step_count_;
};

}  // namespace lucid
