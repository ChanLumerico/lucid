// lucid/_C/optim/LRScheduler.h
//
// Learning-rate schedule classes. Each scheduler is attached to an
// Optimizer at construction time, captures the initial learning rate
// as base_lr_, and overrides compute_lr_at(epoch) to return the
// desired learning rate at that epoch. Calling step() advances the
// internal epoch counter and pushes the new lr into the optimizer.
//
// ReduceLROnPlateau is not a LRScheduler subclass because its
// step() takes a metric argument rather than advancing a fixed epoch
// counter; it is a separate, standalone class.

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "../api.h"

namespace lucid {

class Optimizer;

// Base class for epoch-driven learning-rate schedules.
//
// Each derived class implements compute_lr_at(epoch) which maps a
// non-negative epoch index to a learning rate. epoch_ is 0-indexed
// and is incremented by step() before calling compute_lr_at, so the
// first call to step() queries epoch 1.
class LUCID_API LRScheduler {
public:
    // Capture the optimizer and record base_lr_ from its current lr.
    explicit LRScheduler(Optimizer& opt);
    virtual ~LRScheduler() = default;

    LRScheduler(const LRScheduler&) = delete;
    LRScheduler& operator=(const LRScheduler&) = delete;

    // Advance the epoch counter by one and update optimizer lr.
    void step();

    // Jump the epoch counter to the given value and update optimizer lr
    // immediately without incrementing first.
    void set_epoch(std::int64_t epoch);
    std::int64_t epoch() const { return epoch_; }

protected:
    // Return the learning rate at the given epoch. Called by step() and
    // set_epoch() after updating epoch_.
    virtual double compute_lr_at(std::int64_t epoch) const = 0;

    Optimizer& opt_;
    double base_lr_;    // lr captured from the optimizer at construction.
    std::int64_t epoch_;
};

// Multiply lr by gamma every step_size_ epochs.
// lr = base_lr * gamma^(epoch / step_size)
class LUCID_API StepLR : public LRScheduler {
public:
    StepLR(Optimizer& opt, std::int64_t step_size, double gamma = 0.1);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t step_size_;
    double gamma_;
};

// Multiply lr by gamma every epoch.
// lr = base_lr * gamma^epoch
class LUCID_API ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& opt, double gamma);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double gamma_;
};

// Multiply lr by gamma each time a milestone epoch is reached.
// Milestones are sorted at construction; each passed milestone
// multiplies once so lr = base_lr * gamma^(number of passed milestones).
class LUCID_API MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& opt, std::vector<std::int64_t> milestones, double gamma = 0.1);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::vector<std::int64_t> milestones_; // Sorted ascending at construction.
    double gamma_;
};

// Cosine annealing schedule from base_lr down to eta_min over T_max epochs.
// lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
class LUCID_API CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& opt, std::int64_t T_max, double eta_min = 0.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t T_max_;
    double eta_min_;
};

// User-defined schedule via a callable.
// lr = base_lr * lr_lambda(epoch)
// The lambda receives the current epoch index and returns a multiplicative
// factor applied to base_lr_.
class LUCID_API LambdaLR : public LRScheduler {
public:
    LambdaLR(Optimizer& opt, std::function<double(std::int64_t)> lr_lambda);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::function<double(std::int64_t)> lr_lambda_;
};

// Reduce learning rate when a monitored metric stops improving.
//
// Not a LRScheduler subclass; step() takes a scalar metric value
// instead of an epoch index. Tracks the best metric seen so far and
// counts consecutive bad epochs (epochs that do not improve beyond
// threshold relative to best). After patience_ bad epochs the lr is
// reduced by factor_. A cooldown period after each reduction suppresses
// further reductions for cooldown_ steps to let the model settle.
class LUCID_API ReduceLROnPlateau {
public:
    enum class Mode { Min, Max };
    enum class ThresholdMode { Rel, Abs };

    ReduceLROnPlateau(Optimizer& opt,
                      Mode mode = Mode::Min,
                      double factor = 0.1,
                      std::int64_t patience = 10,
                      double threshold = 1e-4,
                      ThresholdMode threshold_mode = ThresholdMode::Rel,
                      std::int64_t cooldown = 0,
                      double min_lr = 0.0,
                      double eps = 1e-8);

    // Evaluate the metric, update internal counters, and reduce lr if
    // patience has been exceeded outside a cooldown period.
    void step(double metric);

    double last_lr() const { return last_lr_; }
    std::int64_t num_bad_epochs() const { return num_bad_epochs_; }

private:
    // Return true if metric represents a meaningful improvement over best_.
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
// Oscillates between base_lr_cyc_ and max_lr_ over cycles of length
// total_size_ = step_size_up_ + step_size_down_. Three modes:
//   Triangular:  constant amplitude every cycle.
//   Triangular2: amplitude halved each cycle (scale = 1 / 2^cycle).
//   ExpRange:    amplitude decayed by gamma^epoch each cycle.
class LUCID_API CyclicLR : public LRScheduler {
public:
    enum class Mode { Triangular, Triangular2, ExpRange };

    CyclicLR(Optimizer& opt,
             double base_lr,
             double max_lr,
             std::int64_t step_size_up,
             std::int64_t step_size_down = 0,
             Mode mode = Mode::Triangular,
             double gamma = 1.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    double base_lr_cyc_, max_lr_;
    std::int64_t step_size_up_, total_size_;
    Mode mode_;
    double gamma_;
};

// Transformer learning-rate schedule (Vaswani et al., 2017 "Attention Is All You Need").
//
// Increases lr linearly during warmup_steps_ then decays as the inverse
// square root of the step count:
//   lr = factor * model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
// The epoch argument to compute_lr_at is used directly as the step count.
class LUCID_API NoamScheduler : public LRScheduler {
public:
    NoamScheduler(Optimizer& opt,
                  std::int64_t model_size,
                  std::int64_t warmup_steps,
                  double factor = 1.0);

protected:
    double compute_lr_at(std::int64_t epoch) const override;

private:
    std::int64_t model_size_;
    std::int64_t warmup_steps_;
    double factor_;
};

}  // namespace lucid
