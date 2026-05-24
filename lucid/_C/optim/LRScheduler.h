// lucid/_C/optim/LRScheduler.h
//
// Learning-rate schedule classes.
//
// Every scheduler is bound to an :class:`Optimizer` at construction
// time, captures its initial learning rate as ``base_lr_``, and
// overrides :meth:`compute_lr_at` to return the lr at a given epoch.
// Calling :meth:`step` advances the internal epoch counter then
// pushes the new lr into the optimiser.
//
// :class:`ReduceLROnPlateau` is **not** a :class:`LRScheduler`
// subclass — its :meth:`step` takes a scalar *metric* instead of
// advancing a fixed counter, so it is a separate standalone class.
//
// References
// ----------
// Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm
//   Restarts" (ICLR 2017).
// Smith, "Cyclical Learning Rates for Training Neural Networks"
//   (WACV 2017, arXiv:1506.01186).
// Vaswani et al., "Attention Is All You Need" (NeurIPS 2017).

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "../api.h"

namespace lucid {

class Optimizer;

// Base class for epoch-driven learning-rate schedules.
//
// Holds a reference to the target optimiser, snapshots its initial
// learning rate, and exposes a templated update loop driven by the
// virtual :meth:`compute_lr_at` hook.
//
// The internal counter ``epoch_`` is 0-indexed and is **incremented
// before** ``compute_lr_at`` is consulted, so the first call to
// :meth:`step` queries epoch 1 (matching the reference framework's
// convention).
//
// Attributes
// ----------
// opt_ : Optimizer&
//     Optimiser whose learning rate is driven by this scheduler.
// base_lr_ : float
//     Learning rate captured from ``opt_`` at construction time.
// epoch_ : int
//     0-indexed epoch counter.
//
// Notes
// -----
// Copy-constructed and copy-assigned by design — the bound optimiser
// reference makes a copied scheduler ambiguous.
//
// See Also
// --------
// :class:`StepLR`, :class:`MultiStepLR`, :class:`ExponentialLR`,
// :class:`CosineAnnealingLR`, :class:`CyclicLR`, :class:`NoamScheduler`,
// :class:`LambdaLR`, :class:`ReduceLROnPlateau`.
class LUCID_API LRScheduler {
public:
    // Construct a scheduler bound to ``opt`` and snapshot ``base_lr_``.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Optimiser whose learning rate this scheduler will drive.
    //
    // Notes
    // -----
    // ``base_lr_`` is read from ``opt.lr()`` *once* at construction —
    // later external changes to the optimiser's lr are ignored.
    explicit LRScheduler(Optimizer& opt);

    // Virtual destructor — ensures derived ``StepLR`` / ``CosineAnnealingLR``
    // / etc. destructors run when held by a base-class pointer.  Defaulted;
    // ``opt_`` is a reference and ``epoch_`` / ``base_lr_`` are PODs, so
    // nothing to clean up at this layer.
    virtual ~LRScheduler() = default;

    LRScheduler(const LRScheduler&) = delete;
    LRScheduler& operator=(const LRScheduler&) = delete;

    // Advance the epoch counter by one and push the new lr.
    //
    // Notes
    // -----
    // Idiomatic placement is **after** the optimiser's own
    // :meth:`step` inside the training loop so the next iteration sees
    // the freshly computed lr.
    void step();

    // Jump the epoch counter to ``epoch`` and refresh the optimiser lr.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Absolute epoch index to set.
    //
    // Notes
    // -----
    // Unlike :meth:`step` this does **not** pre-increment — useful when
    // restoring from a checkpoint where the saved epoch is the value
    // to resume from.
    void set_epoch(std::int64_t epoch);

    // Return the current epoch counter.
    //
    // Returns
    // -------
    // int
    //     Value of ``epoch_``.
    std::int64_t epoch() const { return epoch_; }

protected:
    // Compute the learning rate at the given epoch.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Epoch index queried by :meth:`step` / :meth:`set_epoch`
    //     after they have updated ``epoch_``.
    //
    // Returns
    // -------
    // float
    //     Learning rate for that epoch.
    virtual double compute_lr_at(std::int64_t epoch) const = 0;

    Optimizer& opt_;
    double base_lr_;  // lr captured from the optimizer at construction.
    std::int64_t epoch_;
};

// Step decay — multiply lr by $\gamma$ every ``step_size`` epochs.
//
// Math
// ----
// $$
//   \eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
// $$
// where $s$ is ``step_size`` and $\eta_0$ is ``base_lr_``.
//
// Attributes
// ----------
// step_size_ : int
//     Period in epochs between successive decays.
// gamma_ : float
//     Multiplicative decay factor.
//
// References
// ----------
// Classical staircase schedule used since AlexNet (Krizhevsky 2012).
class LUCID_API StepLR : public LRScheduler {
public:
    // Build a step-decay schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // step_size : int
    //     Period in epochs.
    // gamma : float, default 0.1
    //     Multiplicative decay factor.
    StepLR(Optimizer& opt, std::int64_t step_size, double gamma = 0.1);

protected:
    // Evaluate the step-decay formula at ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     ``base_lr_ * gamma_ ** (epoch / step_size_)``.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t step_size_;
    double gamma_;
};

// Geometric decay — multiply lr by $\gamma$ every epoch.
//
// Math
// ----
// $$
//   \eta_t = \eta_0 \cdot \gamma^{t}
// $$
//
// Attributes
// ----------
// gamma_ : float
//     Per-epoch multiplicative factor (typically slightly below 1).
class LUCID_API ExponentialLR : public LRScheduler {
public:
    // Build an exponential decay schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // gamma : float
    //     Per-epoch multiplicative factor.
    ExponentialLR(Optimizer& opt, double gamma);

protected:
    // Evaluate $\eta_0 \gamma^t$ at ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     ``base_lr_ * gamma_ ** epoch``.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double gamma_;
};

// Milestone decay — multiply lr by $\gamma$ at each passed milestone.
//
// Math
// ----
// $$
//   \eta_t = \eta_0 \cdot \gamma^{\,|\{m \in M : m \le t\}|}
// $$
// where $M$ is the sorted milestone set.
//
// Attributes
// ----------
// milestones_ : vector of int
//     Epochs at which to apply the decay, sorted ascending at construction.
// gamma_ : float
//     Multiplicative decay factor applied once per passed milestone.
//
// Notes
// -----
// Duplicate milestones decay the lr *twice* at the same epoch — this
// matches the reference framework's behaviour and is rarely useful in
// practice.
class LUCID_API MultiStepLR : public LRScheduler {
public:
    // Build a milestone decay schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // milestones : vector of int
    //     Epochs at which to decay.  Sorted ascending internally.
    // gamma : float, default 0.1
    //     Multiplicative decay factor per milestone.
    MultiStepLR(Optimizer& opt, std::vector<std::int64_t> milestones, double gamma = 0.1);

protected:
    // Evaluate the milestone decay rule at ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     ``base_lr_ * gamma_ ** (number of passed milestones)``.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::vector<std::int64_t> milestones_;  // Sorted ascending at construction.
    double gamma_;
};

// Cosine-annealing schedule from $\eta_{\max}$ down to $\eta_{\min}$
// over $T_{\max}$ epochs, following half a cosine wave.
//
// Smoothly anneals the lr so training spends the early epochs at high
// lr (exploration) and the late epochs at low lr (convergence).
// Often combined with warm restarts (SGDR) for cyclic re-exploration.
//
// Math
// ----
// $$
//   \eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})
//     \left(1 + \cos\!\left(\tfrac{t}{T_{\max}}\,\pi\right)\right)
// $$
// for $t \in [0, T_{\max}]$; saturates at $\eta_{\min}$ for $t > T_{\max}$.
//
// Attributes
// ----------
// T_max_ : int
//     Cycle length in optimiser epochs.
// eta_min_ : float
//     Floor of the schedule.  Defaults to 0.
//
// References
// ----------
// Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm
// Restarts" (ICLR 2017) — Lucid does not yet implement the
// warm-restart variant.
class LUCID_API CosineAnnealingLR : public LRScheduler {
public:
    // Build a cosine-annealing schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // T_max : int
    //     Number of epochs per half-cycle.
    // eta_min : float, default 0.0
    //     Floor of the schedule.
    CosineAnnealingLR(Optimizer& opt, std::int64_t T_max, double eta_min = 0.0);

protected:
    // Evaluate the cosine schedule at ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     Annealed learning rate.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t T_max_;
    double eta_min_;
};

// User-defined schedule driven by a callable.
//
// Math
// ----
// $$
//   \eta_t = \eta_0 \cdot \texttt{lr\_lambda}(t)
// $$
//
// Parameters
// ----------
// opt : Optimizer
//     Target optimiser.
// lr_lambda : callable[int -> float]
//     Function returning the multiplicative factor for a given epoch.
//
// Attributes
// ----------
// lr_lambda_ : ``std::function<double(std::int64_t)>``
//     Stored callable.
//
// Notes
// -----
// Useful for warm-up + decay composites that are not worth giving a
// dedicated subclass.
class LUCID_API LambdaLR : public LRScheduler {
public:
    // Build a callable-driven schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // lr_lambda : callable[int -> float]
    //     Returns the multiplicative factor on ``base_lr_`` for a given epoch.
    LambdaLR(Optimizer& opt, std::function<double(std::int64_t)> lr_lambda);

protected:
    // Evaluate ``base_lr_ * lr_lambda_(epoch)``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     Scheduled learning rate.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::function<double(std::int64_t)> lr_lambda_;
};

// Metric-driven plateau detector — reduce lr when progress stalls.
//
// **Not** a :class:`LRScheduler` subclass: :meth:`step` takes a scalar
// metric instead of an epoch index.  Tracks the best metric seen so
// far and counts consecutive epochs that fail to improve beyond a
// threshold.  Once ``patience`` bad epochs accumulate the optimiser's
// lr is multiplied by ``factor`` (subject to ``min_lr``), after which
// a ``cooldown``-step grace period suppresses further reductions to
// let the model settle into the new regime.
//
// Parameters
// ----------
// opt : Optimizer
//     Target optimiser.
// mode : Mode, default Mode::Min
//     ``Min`` to minimise the metric, ``Max`` to maximise.
// factor : float, default 0.1
//     Multiplicative lr reduction.
// patience : int, default 10
//     Bad-epoch tolerance before reducing.
// threshold : float, default 1e-4
//     Minimum improvement to count as progress (interpretation depends
//     on ``threshold_mode``).
// threshold_mode : ThresholdMode, default ThresholdMode::Rel
//     ``Rel`` — improvement measured as a fraction of ``best_``;
//     ``Abs`` — improvement measured in raw units.
// cooldown : int, default 0
//     Steps to wait after each reduction before resuming bad-epoch counting.
// min_lr : float, default 0.0
//     Floor of the lr — reductions saturate here.
// eps : float, default 1e-8
//     Guard against reductions whose absolute lr change is below this.
//
// Attributes
// ----------
// best_ : float
//     Best metric value observed so far.
// num_bad_epochs_ : int
//     Consecutive non-improving epochs.
// cooldown_counter_ : int
//     Remaining grace steps after the most recent reduction.
// last_lr_ : float
//     lr applied at the last call to :meth:`step`.
//
// Notes
// -----
// The improvement test is implemented via :meth:`is_better` which
// branches on ``mode_`` and ``threshold_mode_``.
//
// See Also
// --------
// :class:`CosineAnnealingLR` — deterministic alternative when the loss
// curve is too noisy to drive a plateau heuristic.
class LUCID_API ReduceLROnPlateau {
public:
    // Whether the watched metric should be minimised or maximised.
    enum class Mode { Min, Max };

    // Whether the improvement threshold is interpreted relatively or
    // absolutely.
    enum class ThresholdMode { Rel, Abs };

    // Construct the plateau detector.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // mode : Mode, default Mode::Min
    //     Whether to minimise or maximise the metric.
    // factor : float, default 0.1
    //     Multiplicative lr reduction.
    // patience : int, default 10
    //     Bad-epoch tolerance.
    // threshold : float, default 1e-4
    //     Improvement threshold.
    // threshold_mode : ThresholdMode, default ThresholdMode::Rel
    //     Relative vs absolute interpretation of ``threshold``.
    // cooldown : int, default 0
    //     Grace period after a reduction.
    // min_lr : float, default 0.0
    //     Floor of the lr.
    // eps : float, default 1e-8
    //     Minimum lr delta to actually apply a reduction.
    ReduceLROnPlateau(Optimizer& opt,
                      Mode mode = Mode::Min,
                      double factor = 0.1,
                      std::int64_t patience = 10,
                      double threshold = 1e-4,
                      ThresholdMode threshold_mode = ThresholdMode::Rel,
                      std::int64_t cooldown = 0,
                      double min_lr = 0.0,
                      double eps = 1e-8);

    // Evaluate ``metric``, update counters, and reduce lr if needed.
    //
    // Parameters
    // ----------
    // metric : float
    //     Latest value of the watched metric (e.g. validation loss).
    //
    // Notes
    // -----
    // No-op while the cooldown counter is positive.  Once ``patience``
    // bad epochs accumulate, the optimiser's lr is multiplied by
    // ``factor_`` (clamped at ``min_lr_``) and the cooldown begins.
    void step(double metric);

    // Return the lr applied at the most recent :meth:`step`.
    //
    // Returns
    // -------
    // float
    //     ``last_lr_``.
    double last_lr() const { return last_lr_; }

    // Return the running count of consecutive non-improving epochs.
    //
    // Returns
    // -------
    // int
    //     ``num_bad_epochs_``.
    std::int64_t num_bad_epochs() const { return num_bad_epochs_; }

private:
    // Test whether ``metric`` improves over ``best_``.
    //
    // Parameters
    // ----------
    // metric : float
    //     Candidate metric value.
    //
    // Returns
    // -------
    // bool
    //     True when ``metric`` beats ``best_`` by at least ``threshold_``,
    //     interpreted according to ``threshold_mode_``.
    bool is_better(double metric) const;

    Optimizer& opt_;
    Mode mode_;
    double factor_;
    std::int64_t patience_;
    double threshold_;
    ThresholdMode threshold_mode_;
    std::int64_t cooldown_;
    double min_lr_;
    // eps guards against reducing lr when the change would be negligible.
    double eps_;

    double best_;
    std::int64_t num_bad_epochs_;
    std::int64_t cooldown_counter_;
    double last_lr_;
};

// Triangular cyclic learning-rate schedule (Smith, 2015).
//
// Oscillates between ``base_lr`` and ``max_lr`` over cycles of length
// ``total_size_ = step_size_up + step_size_down``.  Each cycle linearly
// climbs to ``max_lr`` over ``step_size_up`` epochs then descends to
// ``base_lr`` over ``step_size_down`` epochs.  Three amplitude profiles
// are supported via :type:`Mode`.
//
// Math
// ----
// Let $c = t / T$ be the cycle index, $\phi$ the within-cycle phase,
// and $A_c$ the cycle amplitude.  Then
// $$
//   \eta_t = \eta_{\min} + A_c \cdot \mathrm{tri}(\phi)
// $$
// where ``Triangular`` uses $A_c = (\eta_{\max} - \eta_{\min})$ for every
// cycle, ``Triangular2`` halves it as $A_c = (\eta_{\max} - \eta_{\min})/2^c$,
// and ``ExpRange`` decays it as $(\eta_{\max} - \eta_{\min})\,\gamma^t$.
//
// Parameters
// ----------
// opt : Optimizer
//     Target optimiser.
// base_lr : float
//     Lower envelope of the triangle (distinct from :attr:`base_lr_`
//     in :class:`LRScheduler`, which captures the optimiser's initial lr).
// max_lr : float
//     Upper envelope.
// step_size_up : int
//     Epochs for the ramp from ``base_lr`` to ``max_lr``.
// step_size_down : int, default 0
//     Epochs for the descent.  Zero ⇒ symmetric triangle of width
//     ``2 * step_size_up``.
// mode : Mode, default Mode::Triangular
//     Amplitude profile.
// gamma : float, default 1.0
//     Per-step decay used only by ``ExpRange``.
//
// Attributes
// ----------
// base_lr_cyc_ : float
//     Lower envelope.
// max_lr_ : float
//     Upper envelope.
// step_size_up_ : int
//     Ramp length.
// total_size_ : int
//     Cycle length.
// mode_ : Mode
// gamma_ : float
//
// References
// ----------
// Smith, "Cyclical Learning Rates for Training Neural Networks"
// (WACV 2017, arXiv:1506.01186).
class LUCID_API CyclicLR : public LRScheduler {
public:
    // Amplitude profile selector.
    enum class Mode { Triangular, Triangular2, ExpRange };

    // Build a cyclic schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // base_lr : float
    //     Lower envelope.
    // max_lr : float
    //     Upper envelope.
    // step_size_up : int
    //     Epochs for the upward ramp.
    // step_size_down : int, default 0
    //     Epochs for the downward ramp.  Zero ⇒ symmetric.
    // mode : Mode, default Mode::Triangular
    //     Amplitude profile.
    // gamma : float, default 1.0
    //     Decay used only by ``Mode::ExpRange``.
    CyclicLR(Optimizer& opt,
             double base_lr,
             double max_lr,
             std::int64_t step_size_up,
             std::int64_t step_size_down = 0,
             Mode mode = Mode::Triangular,
             double gamma = 1.0);

protected:
    // Evaluate the cyclic schedule at ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current epoch.
    //
    // Returns
    // -------
    // float
    //     Scheduled learning rate.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double base_lr_cyc_, max_lr_;
    std::int64_t step_size_up_, total_size_;
    Mode mode_;
    double gamma_;
};

// Noam schedule — the Transformer "warm-up then $1/\sqrt{t}$ decay" rule
// (Vaswani et al., 2017).
//
// Linearly ramps the lr during the first ``warmup_steps`` then decays
// it as the inverse square root of the step count.  Scaled by
// $d_\text{model}^{-1/2}$ so that the peak lr matches the convention
// used in the original Transformer paper.
//
// Math
// ----
// $$
//   \eta_t = f \cdot d_\text{model}^{-1/2}
//     \cdot \min\!\bigl(t^{-1/2},\; t \cdot w^{-3/2}\bigr)
// $$
// where $w$ is ``warmup_steps``, $f$ is ``factor``, and $t$ is the
// step count (``compute_lr_at`` uses ``epoch`` directly as $t$).
//
// Attributes
// ----------
// model_size_ : int
//     Transformer hidden width $d_\text{model}$.
// warmup_steps_ : int
//     Number of linear-ramp steps before the inverse-sqrt decay kicks in.
// factor_ : float
//     Global scale on the schedule.
//
// Notes
// -----
// Despite the ``LRScheduler`` base, Noam is conceptually a *step*
// schedule rather than an *epoch* schedule — call :meth:`step` once
// per optimiser iteration, not once per epoch.
//
// References
// ----------
// Vaswani et al., "Attention Is All You Need" §5.3 (NeurIPS 2017).
class LUCID_API NoamScheduler : public LRScheduler {
public:
    // Build a Noam (Transformer) schedule.
    //
    // Parameters
    // ----------
    // opt : Optimizer
    //     Target optimiser.
    // model_size : int
    //     Transformer hidden width $d_\text{model}$.
    // warmup_steps : int
    //     Linear-ramp length before the inverse-sqrt decay starts.
    // factor : float, default 1.0
    //     Global scale on the schedule.
    NoamScheduler(Optimizer& opt,
                  std::int64_t model_size,
                  std::int64_t warmup_steps,
                  double factor = 1.0);

protected:
    // Evaluate the Noam formula at step ``epoch``.
    //
    // Parameters
    // ----------
    // epoch : int
    //     Current step count (Noam treats ``epoch`` as a step counter).
    //
    // Returns
    // -------
    // float
    //     Scheduled learning rate.
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t model_size_;
    std::int64_t warmup_steps_;
    double factor_;
};

}  // namespace lucid
